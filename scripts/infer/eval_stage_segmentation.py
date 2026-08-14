#!/usr/bin/env python3
"""
Stage segmentation inference + evaluation for data_infer (fold-box task).

Pipeline
--------
1. Read the LeRobot dataset from ``--data-dir`` (data_infer format).
2. Run the Advantage Estimator (value model) on every frame of the three camera
   videos (``cam_high / cam_left_wrist / cam_right_wrist``) and collect the
   per-frame ``absolute_value``.
3. Write advantage predictions to ``data_KAI0_20000/`` parquet files (reuses eval.py logic).
4. Segment the ``absolute_value`` trajectory into ``K`` stages, where ``K`` is the
   number of GT segments. Default segmentation is **equal-width** binning over a
   fixed ``[0, 1]`` range.
5. Compare predicted stage vs GT stage (from ``annotation/json/``) and compute
   comprehensive metrics (Stage IoU, Boundary Accuracy, Boundary AUC, Full Overlap).

Usage (inside Docker)::

    cd /kai0
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate kai0
    python scripts/eval_stage_segmentation.py \\
        --data-dir /root/user/code/kai0/data_infer \\
        --ckpt-dir /root/user/code/kai0/value_model/20000 \\
        --num 100 \\
        --output-dir outputs/eval_stage_seg_results

Camera mapping: cam_high → top, cam_left_wrist → left, cam_right_wrist → right
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add evaluator path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "stage_advantage" / "annotation"))

from evaluator import SimpleValueEvaluator

# Camera mapping: data_infer feature name → evaluator position
CAMERA_KEYS = (
    "observation.images.cam_high",      # → top
    "observation.images.cam_left_wrist", # → left
    "observation.images.cam_right_wrist",# → right
)

DEFAULT_PROMPT = "折叠纸箱 (Fold the carton)"

# Boundary tolerance range (in frames) for AUC calculation
BOUNDARY_TOLS = list(range(1, 51))  # 1 to 50 frames


# --------------------------------------------------------------------------- #
# 1. GT Annotation loading
# --------------------------------------------------------------------------- #
def load_gt_annotations(gt_dir: str) -> Dict[int, dict]:
    """Load GT annotations from data_infer/annotation/json/.

    Returns dict: episode_index -> {action_config, frame_count, chunk, episode}
    """
    gt_data = {}
    gt_path = Path(gt_dir)
    for chunk_dir in sorted(gt_path.iterdir()):
        if not chunk_dir.is_dir():
            continue
        for json_file in sorted(chunk_dir.glob("episode_*.json")):
            try:
                with open(json_file, "r") as f:
                    d = json.load(f)
            except Exception:
                continue
            if d.get("status") != "pass":
                continue
            ac = d.get("label_info", {}).get("action_config", [])
            if not ac:
                continue
            ep_idx = d.get("episode_index")
            if ep_idx is None:
                continue
            gt_data[ep_idx] = {
                "action_config": ac,
                "frame_count": d.get("frame_count", 0),
                "chunk": d.get("chunk", str(chunk_dir.name)),
                "episode": d.get("episode", json_file.stem),
            }
    return gt_data


def gt_stages_from_action_config(
    action_config: list[dict], length: int, exclude_mistakes: bool = False
) -> tuple[np.ndarray, int, list[int]]:
    """Return per-frame GT stage (0..K-1), number of stages K, and segment end frames."""
    segs = [s for s in action_config if not (exclude_mistakes and s.get("is_mistake"))]
    segs = sorted(segs, key=lambda s: s.get("start_frame", 0))
    ends = [int(s.get("end_frame", 0)) for s in segs]
    k = len(segs)
    if k == 0:
        return np.zeros(length, dtype=np.int32), 0, []
    ends_arr = np.asarray(ends, dtype=np.int64)
    f = np.arange(length, dtype=np.int64)[:, None]
    completed = np.sum(ends_arr[None, :] <= f, axis=1)
    stages = np.clip(completed, 0, k - 1).astype(np.int32)
    return stages, k, ends


# --------------------------------------------------------------------------- #
# 2. Predicted stages from absolute_value
# --------------------------------------------------------------------------- #
def pred_stages_equal_width(
    abs_vals: np.ndarray, k: int, monotonic: bool = True, lo: float = 0.0, hi: float = 1.0
) -> np.ndarray:
    v = np.asarray(abs_vals, dtype=np.float64)
    if monotonic:
        v = np.maximum.accumulate(v)
    if hi - lo < 1e-9 or k <= 1:
        return np.zeros(len(v), dtype=np.int32)
    pred = np.floor((v - lo) / (hi - lo) * k).astype(np.int32)
    return np.clip(pred, 0, k - 1)


def pred_stages_equal_count(abs_vals: np.ndarray, k: int, monotonic: bool = True) -> np.ndarray:
    v = np.asarray(abs_vals, dtype=np.float64)
    if monotonic:
        v = np.maximum.accumulate(v)
    if len(v) == 0 or k <= 1:
        return np.zeros(len(v), dtype=np.int32)
    qs = np.quantile(v, np.linspace(0.0, 1.0, k + 1))
    pred = np.digitize(v, qs[1:-1]).astype(np.int32)
    return np.clip(pred, 0, k - 1)


# --------------------------------------------------------------------------- #
# 3. Comprehensive Metrics (from ref/ and stage_segment_eval.py)
# --------------------------------------------------------------------------- #
def _change_points(arr: np.ndarray) -> np.ndarray:
    return np.where(np.diff(arr) != 0)[0] + 1


def compute_stage_iou(pred: np.ndarray, gt: np.ndarray, k: int) -> Tuple[Dict[int, float], float]:
    iou_per_stage = {}
    for s in range(k):
        gt_mask = (gt == s)
        pred_mask = (pred == s)
        inter = int((gt_mask & pred_mask).sum())
        union = int((gt_mask | pred_mask).sum())
        iou = inter / union if union > 0 else 0.0
        iou_per_stage[s] = round(iou, 4)
    present = [s for s in range(k) if s in iou_per_stage]
    mean_iou = np.mean([iou_per_stage[s] for s in present]) if present else 0.0
    return iou_per_stage, round(float(mean_iou), 4)


def compute_boundary_metrics(pred: np.ndarray, gt: np.ndarray, length: int) -> Dict:
    pred_changes = _change_points(pred).tolist() if len(pred) > 0 else []
    gt_changes = _change_points(gt).tolist() if len(gt) > 0 else []

    def nearest_mae(src: list, ref: list) -> float:
        if not src:
            return float(length)
        if not ref:
            return 0.0
        ref_arr = np.asarray(ref)
        return float(np.mean(np.min(np.abs(np.asarray(src)[:, None] - ref_arr[None, :]), axis=1)))

    bmae_ptg = nearest_mae(pred_changes, gt_changes)
    bmae_gtp = nearest_mae(gt_changes, pred_changes)

    # Boundary accuracy at tolerances
    tol_acc = {}
    for tol in [10, 20, 30, 50]:
        total = len(gt_changes)
        hits = 0
        if total > 0 and pred_changes:
            for b in gt_changes:
                min_err = np.min(np.abs(np.asarray(pred_changes) - b))
                if min_err <= tol:
                    hits += 1
        tol_acc[f"acc@{tol}f"] = round(hits / total, 4) if total > 0 else 0.0

    # Boundary AUC (0-50 frames)
    auc_accs = []
    for tol in BOUNDARY_TOLS:
        total = len(gt_changes)
        hits = 0
        if total > 0 and pred_changes:
            for b in gt_changes:
                min_err = np.min(np.abs(np.asarray(pred_changes) - b))
                if min_err <= tol:
                    hits += 1
        auc_accs.append(hits / total if total > 0 else 0.0)
    boundary_auc = round(float(np.mean(auc_accs)), 4)

    return {
        "pred_boundaries": pred_changes,
        "gt_boundaries": gt_changes,
        "num_pred_boundaries": len(pred_changes),
        "num_gt_boundaries": len(gt_changes),
        "boundary_mae_pred_to_gt": round(bmae_ptg, 2),
        "boundary_mae_gt_to_pred": round(bmae_gtp, 2),
        "boundary_mae_avg": round((bmae_ptg + bmae_gtp) / 2, 2),
        **tol_acc,
        "boundary_auc": boundary_auc,
    }


def compute_episode_full_overlap(pred: np.ndarray, gt: np.ndarray, k: int) -> Tuple[float, float, float, float]:
    ious = []
    for s in range(k):
        gt_mask = (gt == s)
        pred_mask = (pred == s)
        inter = int((gt_mask & pred_mask).sum())
        union = int((gt_mask | pred_mask).sum())
        iou = inter / union if union > 0 else 0.0
        ious.append(iou)
    min_iou = min(ious) if ious else 0.0
    return (
        round(min_iou, 4),
        1.0 if min_iou >= 0.5 else 0.0,
        1.0 if min_iou >= 0.7 else 0.0,
        1.0 if min_iou >= 0.9 else 0.0,
    )


def compute_frame_metrics(pred: np.ndarray, gt: np.ndarray, k: int) -> Dict:
    acc = float(np.mean(pred == gt)) if len(pred) > 0 else 0.0
    per_stage = {}
    for s in range(k):
        tp = int(np.sum((pred == s) & (gt == s)))
        fp = int(np.sum((pred == s) & (gt != s)))
        fn = int(np.sum((pred != s) & (gt == s)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_stage[s] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(gt == s)),
        }
    return {"frame_accuracy": round(acc, 4), "per_stage": per_stage}


# --------------------------------------------------------------------------- #
# 4. Main pipeline
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Stage segmentation eval on data_infer (fold-box).")
    parser.add_argument("--data-dir", type=str, default="/root/user/code/kai0/data_infer",
                        help="LeRobot dataset root (with data/, videos/, meta/, annotation/json/).")
    parser.add_argument("--gt-dir", type=str, default=None,
                        help="GT annotation JSON dir (default: <data-dir>/annotation/json).")
    parser.add_argument("--ckpt-dir", type=str, default="/root/user/code/kai0/value_model/20000",
                        help="Value-model checkpoint dir (contains model.safetensors).")
    parser.add_argument("--config-name", type=str, default="ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD",
                        help="Training config name of the value model.")
    parser.add_argument("--model-name", type=str, default="KAI0", choices=["KAI0", "PI06"],
                        help="KAI0 (2-timestep) or PI06 (1-timestep).")
    parser.add_argument("--num", type=int, default=100,
                        help="Number of episodes to evaluate (0 = all).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for episode sampling.")
    parser.add_argument("--output-dir", type=str, default="outputs/eval_stage_seg_results",
                        help="Output directory for results.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Task prompt.")
    parser.add_argument("--batch-size", type=int, default=160, help="Batch size for GPU inference.")
    parser.add_argument("--num-workers", type=int, default=32, help="Threads for video loading.")
    parser.add_argument("--relative-interval", type=int, default=50,
                        help="Frames ahead for 2-timestep relative advantage.")
    parser.add_argument("--seg-mode", type=str, default="equal_width", choices=["equal_width", "equal_count"],
                        help="equal_width=divide value range into K equal bins.")
    parser.add_argument("--monotonic", action="store_true", default=True,
                        help="Enforce monotonic non-decreasing absolute_value.")
    parser.add_argument("--no-monotonic", dest="monotonic", action="store_false")
    parser.add_argument("--abs-range", type=float, nargs=2, default=[0.0, 1.0],
                        help="Fixed absolute_value range for equal-width binning.")
    parser.add_argument("--save-parquet", action="store_true", default=False,
                        help="Save advantage predictions to parquet files.")
    parser.add_argument("--save-trajectories", action="store_true",
                        help="Also dump per-frame abs_value/pred/gt to per_episode.json.")
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)  # Do NOT resolve (keeps /kai0/data_infer symlink)
    gt_dir = Path(args.gt_dir) if args.gt_dir else (data_dir / "annotation" / "json")
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    use_2timestep = (args.model_name == "KAI0")

    print("=" * 80)
    print("  STAGE SEGMENTATION INFERENCE + EVALUATION")
    print("=" * 80)
    print(f"  Data:        {data_dir}")
    print(f"  GT:          {gt_dir}")
    print(f"  Checkpoint:  {ckpt_dir}")
    print(f"  Config:      {args.config_name}")
    print(f"  Model:       {args.model_name} ({'2-timestep' if use_2timestep else '1-timestep'})")
    print(f"  Output:      {out_dir}")
    print(f"  Episodes:    {args.num if args.num > 0 else 'ALL'}")
    print(f"  Seg mode:    {args.seg_mode}")
    print("=" * 80)

    # ------------------------------------------------------------------ #
    # Load GT annotations
    # ------------------------------------------------------------------ #
    print("\nLoading GT annotations...")
    gt_data = load_gt_annotations(str(gt_dir))
    all_episode_indices = sorted(gt_data.keys())
    print(f"  Found {len(gt_data)} episodes with valid GT annotations")

    if not all_episode_indices:
        print("ERROR: No GT annotations found!")
        return

    # Select episodes
    if args.num > 0 and args.num < len(all_episode_indices):
        rng = random.Random(args.seed)
        selected = list(all_episode_indices)
        rng.shuffle(selected)
        selected = selected[: args.num]
    else:
        selected = all_episode_indices

    selected.sort()
    print(f"  Selected {len(selected)} episodes for evaluation")

    # ------------------------------------------------------------------ #
    # Initialize evaluator
    # ------------------------------------------------------------------ #
    print("\nLoading model...")
    evaluator = SimpleValueEvaluator(
        config_name=args.config_name,
        ckpt_dir=str(ckpt_dir),
        num_workers=args.num_workers,
    )
    print("  Model loaded successfully")

    # ------------------------------------------------------------------ #
    # Process each episode
    # ------------------------------------------------------------------ #
    all_metrics = []
    max_k = 0

    # If saving parquet, create output directory for it (local writable dir)
    if args.save_parquet:
        parquet_out_dir = out_dir / "advantage_parquet"
        parquet_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Parquet output: {parquet_out_dir}")

    for idx, ep_idx in enumerate(selected):
        gt_info = gt_data[ep_idx]
        chunk = gt_info["chunk"]
        ep_name = gt_info["episode"]
        gt_frame_count = gt_info["frame_count"]

        # Build video paths
        # data_infer/videos/chunk-xxx/observation.images.cam_xxx/episode_xxxxxx.mp4
        video_root = data_dir / "videos" / chunk
        # Use frame_count as max_frame_index (exclusive)
        max_frame_idx = gt_frame_count - 1 if gt_frame_count > 0 else 10**9

        video_paths = tuple(
            str(video_root / cam_key / f"{ep_name}.mp4") for cam_key in CAMERA_KEYS
        )

        # Check videos exist
        if not all(os.path.exists(p) for p in video_paths):
            print(f"  [{idx + 1}/{len(selected)}] ep{ep_idx:06d} missing videos, skip")
            continue

        # Run inference
        try:
            if use_2timestep:
                results = evaluator.evaluate_video_2timesteps_advantages(
                    video_paths=video_paths,
                    prompt=args.prompt,
                    batch_size=args.batch_size,
                    frame_interval=1,
                    relative_interval=args.relative_interval,
                    min_frame_index=0,
                    max_frame_index=max_frame_idx,
                    prefetch=True,
                )
            else:
                results = evaluator.evaluate_video_1timestep_advantage(
                    video_paths=video_paths,
                    prompt=args.prompt,
                    batch_size=args.batch_size,
                    frame_interval=1,
                    min_frame_index=0,
                    max_frame_index=max_frame_idx,
                    prefetch=True,
                )
        except Exception as ex:
            print(f"  [{idx + 1}/{len(selected)}] ep{ep_idx:06d} inference failed: {ex}")
            continue

        # Extract absolute_value
        length = max(r["frame_idx"] for r in results) + 1
        av = np.zeros(length, dtype=np.float64)
        for r in results:
            fi = r["frame_idx"]
            if 0 <= fi < length:
                av[fi] = r["absolute_value"]

        # ------------------------------------------------------------------ #
        # Save advantage predictions to parquet (like eval.py)
        # ------------------------------------------------------------------ #
        if args.save_parquet:
            try:
                import pyarrow.parquet as pq
                import pyarrow as pa

                # Read original parquet
                src_parquet = data_dir / "data" / chunk / f"{ep_name}.parquet"
                if src_parquet.exists():
                    table = pq.read_table(src_parquet)
                    # Build advantage columns
                    adv_dict = {
                        "frame_idx": [r["frame_idx"] for r in results],
                        "relative_advantage": [r["relative_advantage"] for r in results],
                        "absolute_value": [r["absolute_value"] for r in results],
                        "absolute_advantage": [r["absolute_advantage"] for r in results],
                    }
                    adv_table = pa.Table.from_pylist(adv_dict)
                    for col in ["relative_advantage", "absolute_value", "absolute_advantage"]:
                        if col in adv_table.column_names and col not in table.column_names:
                            table = table.append_column(col, adv_table[col])

                    # Write output
                    out_parquet = parquet_out_dir / "data" / chunk / f"{ep_name}.parquet"
                    out_parquet.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(table, out_parquet)
            except Exception as ex:
                print(f"    Warning: parquet save failed: {ex}")

        # ------------------------------------------------------------------ #
        # Stage segmentation and metrics
        # ------------------------------------------------------------------ #
        gt, k, ends = gt_stages_from_action_config(gt_info["action_config"], length)
        if k <= 1:
            print(f"  [{idx + 1}/{len(selected)}] ep{ep_idx:06d} K<=1, skip metrics")
            continue
        max_k = max(max_k, k)

        if args.seg_mode == "equal_width":
            pred = pred_stages_equal_width(av, k, args.monotonic,
                                           lo=args.abs_range[0], hi=args.abs_range[1])
        else:
            pred = pred_stages_equal_count(av, k, args.monotonic)

        # Compute metrics
        frame_metrics = compute_frame_metrics(pred, gt, k)
        boundary_metrics = compute_boundary_metrics(pred, gt, length)
        stage_iou, mean_iou = compute_stage_iou(pred, gt, k)
        min_iou, overlap_05, overlap_07, overlap_09 = compute_episode_full_overlap(pred, gt, k)

        rec = {
            "episode_index": ep_idx,
            "chunk": chunk,
            "episode": ep_name,
            "length": int(length),
            "num_stages": int(k),
            "method": args.model_name,
            "seg_mode": args.seg_mode,
            "monotonic": args.monotonic,
            "frame_accuracy": frame_metrics["frame_accuracy"],
            "stage_iou": stage_iou,
            "mean_iou": mean_iou,
            "boundary_mae_pred_to_gt": boundary_metrics["boundary_mae_pred_to_gt"],
            "boundary_mae_gt_to_pred": boundary_metrics["boundary_mae_gt_to_pred"],
            "boundary_mae_avg": boundary_metrics["boundary_mae_avg"],
            "boundary_acc@10f": boundary_metrics["acc@10f"],
            "boundary_acc@20f": boundary_metrics["acc@20f"],
            "boundary_acc@30f": boundary_metrics["acc@30f"],
            "boundary_acc@50f": boundary_metrics["acc@50f"],
            "boundary_auc": boundary_metrics["boundary_auc"],
            "min_iou": min_iou,
            "overlap_05": overlap_05,
            "overlap_07": overlap_07,
            "overlap_09": overlap_09,
            "per_stage": frame_metrics["per_stage"],
            "gt_segment_ends": ends,
        }
        if args.save_trajectories:
            rec["abs_value"] = av.tolist()
            rec["pred_stages"] = pred.tolist()
            rec["gt_stages"] = gt.tolist()
            rec["pred_boundaries"] = boundary_metrics["pred_boundaries"]
            rec["gt_boundaries"] = boundary_metrics["gt_boundaries"]

        all_metrics.append(rec)

        print(
            f"  [{idx + 1}/{len(selected)}] ep{ep_idx:06d} len={length} K={k} "
            f"acc={frame_metrics['frame_accuracy']:.3f} iou={mean_iou:.3f} "
            f"bMAE={boundary_metrics['boundary_mae_avg']:.1f}f "
            f"B-AUC={boundary_metrics['boundary_auc']:.3f} "
            f"minIoU={min_iou:.3f}"
        )

    # Shutdown evaluator
    if hasattr(evaluator, "shutdown"):
        evaluator.shutdown()

    if not all_metrics:
        print("\nNo episodes evaluated!")
        return

    # ------------------------------------------------------------------ #
    # Aggregate summary
    # ------------------------------------------------------------------ #
    n_eps = len(all_metrics)

    # Frame accuracy
    accs = [m["frame_accuracy"] for m in all_metrics]

    # Stage IoU
    stage_iou_agg = {s: [] for s in range(max_k)}
    for m in all_metrics:
        for s, iou in m["stage_iou"].items():
            if s in stage_iou_agg:
                stage_iou_agg[s].append(iou)
    stage_iou_mean = {str(s): round(np.mean(vals), 4) for s, vals in stage_iou_agg.items() if vals}
    stage_iou_overall = round(np.mean([v for vals in stage_iou_agg.values() for v in vals]), 4) if any(stage_iou_agg.values()) else 0.0

    # Boundary metrics
    boundary_maes = [m["boundary_mae_avg"] for m in all_metrics]
    boundary_aucs = [m["boundary_auc"] for m in all_metrics]
    acc_at = {tol: [m[f"boundary_acc@{tol}f"] for m in all_metrics] for tol in [10, 20, 30, 50]}

    # Episode full overlap
    min_ious = [m["min_iou"] for m in all_metrics]
    overlap_counts = {
        "overlap@0.5": sum(m["overlap_05"] for m in all_metrics),
        "overlap@0.7": sum(m["overlap_07"] for m in all_metrics),
        "overlap@0.9": sum(m["overlap_09"] for m in all_metrics),
    }

    summary = {
        "num_episodes": n_eps,
        "seed": args.seed,
        "abs_range": [float(args.abs_range[0]), float(args.abs_range[1])],
        "seg_mode": args.seg_mode,
        "monotonic": args.monotonic,
        "method": args.model_name,
        "frame_accuracy": {
            "mean": round(float(np.mean(accs)), 4),
            "std": round(float(np.std(accs)), 4),
            "min": round(float(np.min(accs)), 4),
            "max": round(float(np.max(accs)), 4),
            "median": round(float(np.median(accs)), 4),
        },
        "stage_iou": {
            "per_stage_mean": stage_iou_mean,
            "overall_mean": stage_iou_overall,
        },
        "boundary": {
            "mae": {
                "mean": round(float(np.mean(boundary_maes)), 2),
                "std": round(float(np.std(boundary_maes)), 2),
            },
            "acc@10f": round(float(np.mean(acc_at[10])), 4),
            "acc@20f": round(float(np.mean(acc_at[20])), 4),
            "acc@30f": round(float(np.mean(acc_at[30])), 4),
            "acc@50f": round(float(np.mean(acc_at[50])), 4),
            "auc": round(float(np.mean(boundary_aucs)), 4),
        },
        "episode_full_overlap": {
            "min_iou_mean": round(float(np.mean(min_ious)), 4),
            "min_iou_std": round(float(np.std(min_ious)), 4),
            "overlap_rate@0.5": round(overlap_counts["overlap@0.5"] / n_eps, 4),
            "overlap_rate@0.7": round(overlap_counts["overlap@0.7"] / n_eps, 4),
            "overlap_rate@0.9": round(overlap_counts["overlap@0.9"] / n_eps, 4),
        },
    }

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2, ensure_ascii=False)
    json.dump(all_metrics, open(out_dir / "per_episode.json", "w"), indent=2, ensure_ascii=False)

    # Compact CSV
    with open(out_dir / "per_episode.csv", "w") as fcsv:
        header = "episode_index,length,num_stages,frame_accuracy,mean_iou,boundary_mae_avg,boundary_auc,boundary_acc@10f,boundary_acc@20f,boundary_acc@30f,min_iou,overlap_05,overlap_07,overlap_09"
        fcsv.write(header + "\n")
        for m in all_metrics:
            fcsv.write(
                f"{m['episode_index']},{m['length']},{m['num_stages']},"
                f"{m['frame_accuracy']:.4f},{m['mean_iou']:.4f},"
                f"{m['boundary_mae_avg']:.2f},{m['boundary_auc']:.4f},"
                f"{m['boundary_acc@10f']:.4f},{m['boundary_acc@20f']:.4f},{m['boundary_acc@30f']:.4f},"
                f"{m['min_iou']:.4f},{m['overlap_05']},{m['overlap_07']},{m['overlap_09']}\n"
            )

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("  STAGE SEGMENTATION EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\n  Episodes: {n_eps}")
    print(f"  Method: {args.model_name}, Seg: {args.seg_mode}, Monotonic: {args.monotonic}")

    fa = summary["frame_accuracy"]
    print(f"\n  ① Frame-level Accuracy")
    print(f"     Mean: {fa['mean']:.4f} ± {fa['std']:.4f}")
    print(f"     Min/Max: {fa['min']:.4f} / {fa['max']:.4f}")

    print(f"\n  ② Stage IoU (Intersection over Union)")
    print(f"     Per-stage mean: {summary['stage_iou']['per_stage_mean']}")
    print(f"     Overall mean: {summary['stage_iou']['overall_mean']:.4f}")

    b = summary["boundary"]
    print(f"\n  ③ Boundary Accuracy")
    print(f"     @±10f: {b['acc@10f']:.4f}   @±20f: {b['acc@20f']:.4f}")
    print(f"     @±30f: {b['acc@30f']:.4f}   @±50f: {b['acc@50f']:.4f}")
    print(f"     Boundary AUC (0-50f): {b['auc']:.4f}")
    print(f"     Boundary MAE: {b['mae']['mean']:.2f} ± {b['mae']['std']:.2f} frames")

    eo = summary["episode_full_overlap"]
    print(f"\n  ④ Episode Full Overlap (min-IoU across all stages)")
    print(f"     Mean min-IoU: {eo['min_iou_mean']:.4f} ± {eo['min_iou_std']:.4f}")
    print(f"     Overlap @≥0.5: {eo['overlap_rate@0.5']:.4f} ({overlap_counts['overlap@0.5']}/{n_eps})")
    print(f"     Overlap @≥0.7: {eo['overlap_rate@0.7']:.4f} ({overlap_counts['overlap@0.7']}/{n_eps})")
    print(f"     Overlap @≥0.9: {eo['overlap_rate@0.9']:.4f} ({overlap_counts['overlap@0.9']}/{n_eps})")

    print("\n" + "=" * 80)
    print(f"  Results saved to:")
    print(f"    - {out_dir / 'summary.json'}")
    print(f"    - {out_dir / 'per_episode.json'}")
    print(f"    - {out_dir / 'per_episode.csv'}")
    if args.save_parquet:
        print(f"    - {parquet_out_dir}  (advantage predictions)")
    print()


if __name__ == "__main__":
    main()