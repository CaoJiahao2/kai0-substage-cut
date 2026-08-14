"""
Stage Segmentation Inference and Evaluation Script for kai0.

This script:
1. Loads the Advantage Estimator from checkpoint/value_model/20000
2. Runs inference on episodes from data_infer/ (LeRobot format)
3. Uses absolute_value trajectory to segment videos into stages (equal intervals)
4. Compares predictions with GT annotations from data/cloth_all/annotation/json/
5. Calculates segmentation metrics

Usage:
    python infer_stage_segmentation.py --num_episodes 100 --output_dir outputs/eval_折叠纸箱_round1
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stage_advantage.annotation.evaluator import SimpleValueEvaluator
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import pyarrow.parquet as pq


# Video key mapping: feature name in LeRobot -> evaluator camera name
VIDEO_KEYS = {
    'observation.images.cam_high': 'top',
    'observation.images.cam_left_wrist': 'left',
    'observation.images.cam_right_wrist': 'right',
}
CAMERA_REVERSE = {
    'top': 'observation.images.cam_high',
    'left': 'observation.images.cam_left_wrist',
    'right': 'observation.images.cam_right_wrist',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Stage Segmentation Inference and Evaluation')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes to evaluate')
    parser.add_argument('--output_dir', type=str, default='outputs/eval_折叠纸箱_round1', help='Output directory')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoint/value_model/20000', help='Checkpoint directory')
    parser.add_argument('--config_name', type=str, default='ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD', help='Model config name')
    parser.add_argument('--model_name', type=str, default='KAI0', choices=['KAI0', 'PI06'], help='Model variant')
    parser.add_argument('--data_dir', type=str, default='data_infer', help='LeRobot dataset directory (with data/, videos/, meta/ subdirs)')
    parser.add_argument('--gt_dir', type=str, default='data_infer/annotation/json', help='GT annotation JSON directory')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for video loading')
    parser.add_argument('--batch_size', type=int, default=160, help='Batch size for inference')
    parser.add_argument('--prompt', type=str, default='折叠纸箱 (Fold the carton)', help='Task prompt')
    parser.add_argument('--relative_interval', type=int, default=50, help='Frame interval for relative advantage')
    parser.add_argument('--start_episode', type=int, default=0, help='Start episode index')
    return parser.parse_args()


def load_gt_annotations(gt_dir: str, episode_indices: List[int]) -> Dict[int, Dict]:
    """
    Load GT annotations from json files.

    Returns dict: episode_idx -> {action_config: [{start_frame, end_frame, action_text, skill}, ...]}
    """
    gt_data = {}
    gt_path = Path(gt_dir)

    for chunk_dir in gt_path.iterdir():
        if not chunk_dir.is_dir():
            continue
        for json_file in chunk_dir.glob('episode_*.json'):
            # Parse episode index from filename
            stem = json_file.stem  # e.g., "episode_000000"
            idx_str = stem.replace('episode_', '')
            try:
                episode_idx = int(idx_str)
            except ValueError:
                continue

            if episode_idx not in episode_indices:
                continue

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'label_info' in data and 'action_config' in data['label_info']:
                gt_data[episode_idx] = {
                    'action_config': data['label_info']['action_config'],
                    'frame_count': data.get('frame_count', 0),
                }

    return gt_data


def compute_predicted_stages(absolute_values: np.ndarray, num_stages: int) -> np.ndarray:
    """
    Compute predicted stage IDs based on absolute_value trajectory.

    Uses equal-interval segmentation:
    - 5 stages: [0.0-0.2)=stage0, [0.2-0.4)=stage1, [0.4-0.6)=stage2, [0.6-0.8)=stage3, [0.8-1.0]=stage4

    Args:
        absolute_values: Array of absolute_value predictions (0-1 normalized)
        num_stages: Number of stages to segment into

    Returns:
        Array of stage IDs (0 to num_stages-1) for each frame
    """
    # Normalize absolute_values to [0, 1] if not already
    if absolute_values.max() > 1.0:
        absolute_values = (absolute_values - absolute_values.min()) / (absolute_values.max() - absolute_values.min() + 1e-8)

    # Equal interval boundaries
    boundaries = np.linspace(0, 1, num_stages + 1)  # e.g., [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Assign stage ID based on which interval each frame falls into
    stage_ids = np.zeros(len(absolute_values), dtype=np.int64)
    for i in range(1, num_stages):
        stage_ids[absolute_values >= boundaries[i]] = i

    return stage_ids


def compute_gt_stages(action_config: List[Dict], total_frames: int, num_stages: int) -> np.ndarray:
    """
    Compute GT stage IDs from action_config annotations.

    Args:
        action_config: List of {start_frame, end_frame, ...} from GT
        total_frames: Total number of frames in the episode
        num_stages: Number of stages (derived from GT annotation count)

    Returns:
        Array of stage IDs for each frame
    """
    stage_ids = np.zeros(total_frames, dtype=np.int64)

    # If GT has fewer annotations than num_stages, use GT's actual stage count
    if len(action_config) < num_stages:
        num_stages = len(action_config)

    for stage_idx, action in enumerate(action_config):
        start = action['start_frame']
        end = action['end_frame']
        # Handle edge case where end might be exclusive
        end = min(end, total_frames)
        stage_ids[start:end] = stage_idx

    return stage_ids


def calculate_segmentation_metrics(
    pred_stages: np.ndarray,
    gt_stages: np.ndarray,
    pred_boundaries: List[int],
    gt_boundaries: List[int]
) -> Dict[str, Any]:
    """
    Calculate segmentation metrics.

    Metrics:
    - Boundary accuracy: % of predicted boundaries within threshold of GT boundaries
    - Stage IoU: Intersection-over-Union for each stage
    - Frame-level accuracy: % of frames with correct stage assignment
    """
    metrics = {}

    # Frame-level accuracy
    metrics['frame_accuracy'] = float(np.mean(pred_stages == gt_stages))
    metrics['frame_error'] = 1.0 - metrics['frame_accuracy']

    # Number of stages
    metrics['num_pred_stages'] = int(pred_stages.max() + 1)
    metrics['num_gt_stages'] = int(gt_stages.max() + 1)

    # Stage IoU
    all_stages = set(pred_stages) | set(gt_stages)
    ious = []
    for stage_id in all_stages:
        pred_mask = (pred_stages == stage_id)
        gt_mask = (gt_stages == stage_id)
        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask | gt_mask)
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if stage_id not in pred_stages and stage_id not in gt_stages else 0.0
        ious.append(iou)
    metrics['mean_iou'] = float(np.mean(ious)) if ious else 0.0

    # Boundary metrics
    BOUNDARY_THRESHOLD = 10  # frames

    pred_boundary_set = set(pred_boundaries)
    gt_boundary_set = set(gt_boundaries)

    # Precision: how many predicted boundaries are close to GT
    if len(pred_boundary_set) > 0:
        true_positives = 0
        for pb in pred_boundary_set:
            for gb in gt_boundary_set:
                if abs(pb - gb) <= BOUNDARY_THRESHOLD:
                    true_positives += 1
                    break
        metrics['boundary_precision'] = true_positives / len(pred_boundary_set)
    else:
        metrics['boundary_precision'] = 0.0

    # Recall: how many GT boundaries are captured by predictions
    if len(gt_boundary_set) > 0:
        true_positives = 0
        for gb in gt_boundary_set:
            for pb in pred_boundary_set:
                if abs(pb - gb) <= BOUNDARY_THRESHOLD:
                    true_positives += 1
                    break
        metrics['boundary_recall'] = true_positives / len(gt_boundary_set)
    else:
        metrics['boundary_recall'] = 0.0

    # F1
    if metrics['boundary_precision'] + metrics['boundary_recall'] > 0:
        metrics['boundary_f1'] = 2 * metrics['boundary_precision'] * metrics['boundary_recall'] / (
            metrics['boundary_precision'] + metrics['boundary_recall']
        )
    else:
        metrics['boundary_f1'] = 0.0

    # Boundary error (mean absolute error to nearest GT boundary)
    boundary_errors = []
    for pb in pred_boundary_set:
        min_error = min(abs(pb - gb) for gb in gt_boundary_set) if gt_boundary_set else float('inf')
        boundary_errors.append(min_error)
    metrics['mean_boundary_error'] = float(np.mean(boundary_errors)) if boundary_errors else 0.0

    return metrics


def find_boundaries(stage_ids: np.ndarray) -> List[int]:
    """Find frame indices where stage changes (boundaries)."""
    boundaries = []
    for i in range(1, len(stage_ids)):
        if stage_ids[i] != stage_ids[i-1]:
            boundaries.append(i)
    return boundaries


def run_inference(
    evaluator: SimpleValueEvaluator,
    repo_id: Path,
    episode_idx: int,
    prompt: str,
    batch_size: int,
    relative_interval: int,
    use_2timestep: bool
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Run inference on a single episode.

    Returns:
        (absolute_values, results_dict)
    """
    dataset_metadata = lerobot_dataset.LeRobotDatasetMetadata(repo_id=str(repo_id))

    parquet_file = repo_id / dataset_metadata.data_path.format(
        episode_chunk=episode_idx // dataset_metadata.chunks_size,
        episode_index=episode_idx
    )

    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

    # Read frame range from parquet
    table = pq.read_table(parquet_file)
    frame_indices = table['frame_index'].to_pylist()
    min_frame_index = frame_indices[0]
    max_frame_index = frame_indices[-1]
    num_frames = len(frame_indices)

    # Get video paths - use full feature names as video keys
    video_paths = {}
    for video_key in VIDEO_KEYS.keys():
        video_path = repo_id / dataset_metadata.video_path.format(
            episode_chunk=episode_idx // dataset_metadata.chunks_size,
            episode_index=episode_idx,
            video_key=video_key
        )
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_paths[VIDEO_KEYS[video_key]] = video_path

    # Run inference
    if use_2timestep:
        results = evaluator.evaluate_video_2timesteps_advantages(
            video_paths=(video_paths['top'], video_paths['left'], video_paths['right']),
            prompt=prompt,
            batch_size=batch_size,
            frame_interval=1,
            relative_interval=relative_interval,
            min_frame_index=min_frame_index,
            max_frame_index=max_frame_index,
            prefetch=True,
        )
    else:
        results = evaluator.evaluate_video_1timestep_advantage(
            video_paths=(video_paths['top'], video_paths['left'], video_paths['right']),
            prompt=prompt,
            batch_size=batch_size,
            frame_interval=1,
            min_frame_index=min_frame_index,
            max_frame_index=max_frame_index,
            prefetch=True,
        )

    # Extract absolute_values
    absolute_values = np.array([r['absolute_value'] for r in results])

    return absolute_values, results


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"Stage Segmentation Evaluation")
    print(f"=" * 60)
    print(f"Model: {args.model_name} ({args.config_name})")
    print(f"Checkpoint: {args.ckpt_dir}")
    print(f"Data: {args.data_dir}")
    print(f"GT: {args.gt_dir}")
    print(f"Episodes: {args.start_episode} to {args.start_episode + args.num_episodes - 1}")
    print(f"Output: {output_dir}")
    print(f"=" * 60)

    # Initialize evaluator
    print("Loading model...")
    evaluator = SimpleValueEvaluator(
        config_name=args.config_name,
        ckpt_dir=args.ckpt_dir,
        num_workers=args.num_workers,
    )

    # Dataset metadata
    repo_id = Path(args.data_dir)
    dataset_metadata = lerobot_dataset.LeRobotDatasetMetadata(repo_id=str(repo_id))

    # Episode indices to evaluate
    episode_indices = list(range(args.start_episode, args.start_episode + args.num_episodes))

    # Load GT annotations
    print("Loading GT annotations...")
    gt_data = load_gt_annotations(args.gt_dir, episode_indices)
    print(f"Found {len(gt_data)} episodes with GT annotations")

    # Filter to episodes that have both data and GT
    valid_episodes = [idx for idx in episode_indices if idx in gt_data]

    # Determine number of stages from GT
    num_stages_list = [len(gt_data[idx]['action_config']) for idx in valid_episodes]
    # Use mode (most common) number of stages, or default to 5
    if num_stages_list:
        num_stages = max(set(num_stages_list), key=num_stages_list.count)
    else:
        num_stages = 5
    print(f"Using {num_stages} stages (from GT mode)")

    # Use 2timestep mode for KAI0, 1timestep for PI06
    use_2timestep = (args.model_name == 'KAI0')

    # Run inference and evaluation
    all_metrics = []
    episode_results = []

    for episode_idx in tqdm(valid_episodes, desc="Evaluating episodes"):
        try:
            # Run inference
            absolute_values, inference_results = run_inference(
                evaluator=evaluator,
                repo_id=repo_id,
                episode_idx=episode_idx,
                prompt=args.prompt,
                batch_size=args.batch_size,
                relative_interval=args.relative_interval,
                use_2timestep=use_2timestep,
            )

            # Get GT stages
            gt_info = gt_data[episode_idx]
            action_config = gt_info['action_config']
            total_frames = len(absolute_values)  # Use actual inference output length

            # Compute predicted stages
            pred_stages = compute_predicted_stages(absolute_values, num_stages)

            # Compute GT stages (pad to match prediction length)
            gt_stages = compute_gt_stages(action_config, total_frames, num_stages)

            # Extend GT stages if needed
            if len(gt_stages) < total_frames:
                gt_stages = np.pad(gt_stages, (0, total_frames - len(gt_stages)), mode='edge')

            # Find boundaries
            pred_boundaries = find_boundaries(pred_stages)
            gt_boundaries = find_boundaries(gt_stages)

            # Calculate metrics
            metrics = calculate_segmentation_metrics(pred_stages, gt_stages, pred_boundaries, gt_boundaries)
            metrics['episode_idx'] = episode_idx
            metrics['num_gt_stages'] = len(action_config)
            metrics['num_pred_stages'] = int(pred_stages.max() + 1)

            all_metrics.append(metrics)
            episode_results.append({
                'episode_idx': episode_idx,
                'metrics': metrics,
                'gt_action_config': action_config,
                'pred_boundaries': pred_boundaries,
                'gt_boundaries': gt_boundaries,
                'total_frames': total_frames,
            })

        except Exception as e:
            print(f"Error processing episode {episode_idx}: {e}")
            continue

    # Shutdown evaluator
    evaluator.shutdown()

    # Aggregate metrics
    if not all_metrics:
        print("No valid results!")
        return

    print("\n" + "=" * 60)
    print("AGGREGATE METRICS")
    print("=" * 60)

    # Compute aggregate statistics
    frame_accuracies = [m['frame_accuracy'] for m in all_metrics]
    mean_ious = [m['mean_iou'] for m in all_metrics]
    boundary_f1s = [m['boundary_f1'] for m in all_metrics]
    boundary_precisions = [m['boundary_precision'] for m in all_metrics]
    boundary_recalls = [m['boundary_recall'] for m in all_metrics]
    mean_boundary_errors = [m['mean_boundary_error'] for m in all_metrics]

    aggregate_metrics = {
        'num_episodes': len(all_metrics),
        'frame_accuracy_mean': float(np.mean(frame_accuracies)),
        'frame_accuracy_std': float(np.std(frame_accuracies)),
        'frame_accuracy_min': float(np.min(frame_accuracies)),
        'frame_accuracy_max': float(np.max(frame_accuracies)),
        'mean_iou_mean': float(np.mean(mean_ious)),
        'mean_iou_std': float(np.std(mean_ious)),
        'boundary_f1_mean': float(np.mean(boundary_f1s)),
        'boundary_f1_std': float(np.std(boundary_f1s)),
        'boundary_precision_mean': float(np.mean(boundary_precisions)),
        'boundary_recall_mean': float(np.mean(boundary_recalls)),
        'mean_boundary_error_mean': float(np.mean(mean_boundary_errors)),
        'mean_boundary_error_std': float(np.std(mean_boundary_errors)),
    }

    # Print aggregate metrics
    print(f"Episodes evaluated: {aggregate_metrics['num_episodes']}")
    print(f"\nFrame-level Accuracy:")
    print(f"  Mean: {aggregate_metrics['frame_accuracy_mean']:.4f} ± {aggregate_metrics['frame_accuracy_std']:.4f}")
    print(f"  Min/Max: {aggregate_metrics['frame_accuracy_min']:.4f} / {aggregate_metrics['frame_accuracy_max']:.4f}")
    print(f"\nMean IoU:")
    print(f"  Mean: {aggregate_metrics['mean_iou_mean']:.4f} ± {aggregate_metrics['mean_iou_std']:.4f}")
    print(f"\nBoundary Metrics:")
    print(f"  F1: {aggregate_metrics['boundary_f1_mean']:.4f} ± {aggregate_metrics['boundary_f1_std']:.4f}")
    print(f"  Precision: {aggregate_metrics['boundary_precision_mean']:.4f}")
    print(f"  Recall: {aggregate_metrics['boundary_recall_mean']:.4f}")
    print(f"  Mean Boundary Error: {aggregate_metrics['mean_boundary_error_mean']:.2f} ± {aggregate_metrics['mean_boundary_error_std']:.2f} frames")

    # Save results
    results = {
        'args': vars(args),
        'aggregate_metrics': aggregate_metrics,
        'per_episode_metrics': all_metrics,
        'episode_results': episode_results,
    }

    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {results_path}")

    # Save per-episode summary CSV
    csv_path = output_dir / 'metrics_summary.csv'
    with open(csv_path, 'w') as f:
        f.write('episode_idx,frame_accuracy,mean_iou,boundary_f1,boundary_precision,boundary_recall,mean_boundary_error,num_gt_stages,num_pred_stages\n')
        for m in all_metrics:
            f.write(f"{m['episode_idx']},{m['frame_accuracy']:.4f},{m['mean_iou']:.4f},"
                   f"{m['boundary_f1']:.4f},{m['boundary_precision']:.4f},{m['boundary_recall']:.4f},"
                   f"{m['mean_boundary_error']:.2f},{m['num_gt_stages']},{m['num_pred_stages']}\n")

    print(f"CSV summary saved to: {csv_path}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
