#!/usr/bin/env python3
"""Stage-segmentation evaluation on annotated cloth_all episodes.

Pipeline
--------
1. Select N (default 100) annotated episodes from
   ``data/cloth_all/annotation/json/chunk-*/episode_*.json`` (``status==pass`` and
   non-empty ``label_info.action_config``). The ``action_config`` segments are the
   human ground-truth (GT) stage boundaries.
2. Run the Advantage Estimator (value model) on every frame of the three camera
   videos (``cam_high / cam_left_wrist / cam_right_wrist``) and collect the
   per-frame ``absolute_value`` (cumulative progress signal).
3. Segment the ``absolute_value`` trajectory into ``K`` stages, where ``K`` is the
   number of GT segments. Default segmentation is **equal-width** binning over a
   fixed ``[0, 1]`` range (``--seg-mode equal_width``, ``--abs-range 0.0 1.0``); for
   ``K=5`` this yields stages 0-0.2 / 0.2-0.4 / 0.4-0.6 / 0.6-0.8 / 0.8-1.0.
   ``equal_count`` does equal-population (quantile) binning instead.
4. Compare predicted stage vs GT stage per frame and compute comprehensive metrics:
   - Stage IoU (per-class intersection over union)
   - Boundary Accuracy @±10f, @±20f, @±30f, @±50f
   - Boundary AUC (area under accuracy-tolerance curve, 0-50 frames)
   - Episode Full Overlap (min-IoU ≥ threshold)

Usage (inside the kai0 GPU env, e.g. uv / conda)::

    uv run python stage_advantage/annotation/stage_segment_eval.py \
        --data-root data/cloth_all \
        --ckpt-dir checkpoint/value_model/20000 \
        --config-name ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
        --num 100 --video-source annotation --method 2timestep \
        --seg-mode equal_width --abs-range 0.0 1.0

Dry-run (no model, validates episode selection + GT parsing only)::

    python3 stage_advantage/annotation/stage_segment_eval.py --dry-run --num 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# Allow `from evaluator import SimpleValueEvaluator` when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# datasight camera keys (in that order: top, left wrist, right wrist).
CAMS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
DEFAULT_PROMPT = "Flatten and fold the cloth."

# Boundary tolerance range (in frames) for AUC calculation
BOUNDARY_TOLS = list(range(1, 51))  # 1 to 50 frames


# --------------------------------------------------------------------------- #
# Episode selection & GT
# --------------------------------------------------------------------------- #
def list_annotated_episodes(data_root: Path, require_pass: bool = True) -> list[dict]:
    """List annotated episodes with valid action_config GT."""
    files = sorted(glob.glob(str(data_root / "annotation/json/chunk-*/episode_*.json")))
    eps: list[dict] = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if require_pass and d.get("status") != "pass":
            continue
        ac = d.get("label_info", {}).get("action_config", [])
        if not ac:
            continue
        eps.append(
            {
                "episode_index": d.get("episode_index"),
                "chunk": d.get("chunk"),
                "episode": d.get("episode"),
                "json_path": f,
                "frame_count": d.get("frame_count"),
                "action_config": ac,
            }
        )
    return eps


def video_paths_for(data_root: Path, chunk: str, episode: str, source: str) -> tuple[str, str, str]:
    """Build (top, left, right) mp4 paths for an episode from the chosen video source."""
    base = data_root / ("annotation/videos" if source == "annotation" else "videos") / chunk
    return tuple(str(base / cam / f"{episode}.mp4") for cam in CAMS)


def gt_stages_from_action_config(
    action_config: list[dict], length: int, exclude_mistakes: bool = False
) -> tuple[np.ndarray, int, list[int]]:
    """Return per-frame GT stage (0..K-1), number of stages K, and segment end frames.

    Stage of frame f = number of segments whose ``end_frame <= f`` (clamped to K-1).
    This yields a monotonic 0..K-1 progression and naturally absorbs lead-in frames,
    gaps and trailing frames into the neighbouring stage.
    """
    segs = [s for s in action_config if not (exclude_mistakes and s.get("is_mistake"))]
    segs = sorted(segs, key=lambda s: s.get("start_frame", 0))
    ends = [int(s.get("end_frame", 0)) for s in segs]
    k = len(segs)
    if k == 0:
        return np.zeros(length, dtype=np.int32), 0, []
    ends_arr = np.asarray(ends, dtype=np.int64)
    # count of segments with end_frame <= f, for all f in [0, length)
    f = np.arange(length, dtype=np.int64)[:, None]
    completed = np.sum(ends_arr[None, :] <= f, axis=1)
    stages = np.clip(completed, 0, k - 1).astype(np.int32)
    return stages, k, ends


# --------------------------------------------------------------------------- #
# Predicted stages from absolute_value
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
    # digitize by inner edges -> 0..k-1
    pred = np.digitize(v, qs[1:-1]).astype(np.int32)
    return np.clip(pred, 0, k - 1)


# --------------------------------------------------------------------------- #
# Comprehensive Metrics (from ref/src/eval/eval_stage_detail.py)
# --------------------------------------------------------------------------- #
def _change_points(arr: np.ndarray) -> np.ndarray:
    """Find indices where stage changes (boundaries)."""
    return np.where(np.diff(arr) != 0)[0] + 1


def compute_stage_iou(pred: np.ndarray, gt: np.ndarray, k: int) -> Tuple[Dict[int, float], float]:
    """Compute per-stage IoU and mean IoU.

    Returns:
        iou_per_stage: dict mapping stage_id to IoU value
        mean_iou: average IoU across all stages
    """
    iou_per_stage = {}
    total_inter = 0
    total_union = 0

    for s in range(k):
        gt_mask = (gt == s)
        pred_mask = (pred == s)
        inter = int((gt_mask & pred_mask).sum())
        union = int((gt_mask | pred_mask).sum())
        iou = inter / union if union > 0 else 0.0
        iou_per_stage[s] = round(iou, 4)
        total_inter += inter
        total_union += union

    # Mean IoU across all stages (per-class)
    present_stages = [s for s in range(k) if s in iou_per_stage]
    mean_iou = np.mean([iou_per_stage[s] for s in present_stages]) if present_stages else 0.0

    return iou_per_stage, round(float(mean_iou), 4)


def compute_boundary_metrics(pred: np.ndarray, gt: np.ndarray, length: int) -> Dict:
    """Compute comprehensive boundary metrics.

    Includes:
    - Boundary MAE (pred to GT, GT to pred)
    - Boundary Accuracy @±10f, @±20f, @±30f, @±50f
    - Boundary AUC (area under accuracy-tolerance curve, 0-50 frames)

    Returns:
        dict with boundary metrics
    """
    pred_changes = _change_points(pred)
    gt_changes = _change_points(gt)

    pred_changes_list = pred_changes.tolist() if len(pred_changes) > 0 else []
    gt_changes_list = gt_changes.tolist() if len(gt_changes) > 0 else []

    # Boundary MAE
    def nearest_mae(src: list, ref: list) -> float:
        if not src:
            return float(length)
        if not ref:
            return 0.0
        ref_arr = np.asarray(ref)
        return float(np.mean(np.min(np.abs(np.asarray(src)[:, None] - ref_arr[None, :]), axis=1)))

    boundary_mae_pred_to_gt = nearest_mae(pred_changes_list, gt_changes_list)
    boundary_mae_gt_to_pred = nearest_mae(gt_changes_list, pred_changes_list)

    # Boundary Accuracy at different tolerances
    tol_acc = {}
    for tol in [10, 20, 30, 50]:
        acc = 0.0
        total = 0
        for b in gt_changes_list:
            if pred_changes_list:
                min_err = np.min(np.abs(np.asarray(pred_changes_list) - b))
                if min_err <= tol:
                    acc += 1
            total += 1
        tol_acc[f"acc@{tol}f"] = round(acc / total, 4) if total > 0 else 0.0

    # Boundary AUC (0-50 frames)
    # For each tolerance t in 1..50, compute accuracy and average
    auc_accs = []
    for tol in BOUNDARY_TOLS:
        acc = 0.0
        total = 0
        for b in gt_changes_list:
            if pred_changes_list:
                min_err = np.min(np.abs(np.asarray(pred_changes_list) - b))
                if min_err <= tol:
                    acc += 1
            total += 1
        auc_accs.append(acc / total if total > 0 else 0.0)

    # AUC = average accuracy across all tolerances (normalized by max tolerance)
    boundary_auc = round(float(np.mean(auc_accs)), 4)

    return {
        "pred_boundaries": pred_changes_list,
        "gt_boundaries": gt_changes_list,
        "num_pred_boundaries": len(pred_changes_list),
        "num_gt_boundaries": len(gt_changes_list),
        "boundary_mae_pred_to_gt": round(boundary_mae_pred_to_gt, 2),
        "boundary_mae_gt_to_pred": round(boundary_mae_gt_to_pred, 2),
        "boundary_mae_avg": round((boundary_mae_pred_to_gt + boundary_mae_gt_to_pred) / 2, 2),
        **tol_acc,
        "boundary_auc": boundary_auc,
    }


def compute_episode_full_overlap(pred: np.ndarray, gt: np.ndarray, k: int) -> Tuple[float, float, float]:
    """Compute episode-level full overlap metrics.

    Returns:
        (min_iou, overlap_05, overlap_07, overlap_09)
        - min_iou: minimum IoU across all stages in this episode
        - overlap_05: 1.0 if min_iou >= 0.5, else 0.0
        - overlap_07: 1.0 if min_iou >= 0.7, else 0.0
        - overlap_09: 1.0 if min_iou >= 0.9, else 0.0
    """
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
    """Compute per-stage precision/recall/F1 and frame-level accuracy."""
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
    return {
        "frame_accuracy": round(acc, 4),
        "per_stage": per_stage,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage segmentation eval on annotated cloth_all episodes.")
    p.add_argument("--data-root", default="data/cloth_all", help="cloth_all dataset root.")
    p.add_argument("--ckpt-dir", default="checkpoint/value_model/20000", help="Value-model checkpoint dir (contains model.safetensors).")
    p.add_argument("--config-name", default="ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD", help="Training config name of the value model.")
    p.add_argument("--num", type=int, default=100, help="Number of episodes to randomly sample (ignored when --all is set).")
    p.add_argument("--all", action="store_true", help="Evaluate ALL valid annotated episodes (ignores --num).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for episode sampling.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split the selected episodes into N shards for parallelism (use with --shard-id).")
    p.add_argument("--shard-id", type=int, default=0,
                   help="0-based shard index to evaluate (requires --num-shards>1).")
    p.add_argument("--abs-range", type=float, nargs=2, default=[0.0, 1.0],
                   metavar=("LO", "HI"), help="Fixed absolute_value range for equal-width binning (default 0.0 1.0).")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--video-source", choices=["annotation", "main"], default="annotation",
                   help="annotation=low-res annotation videos (faster); main=full-res videos.")
    p.add_argument("--method", choices=["1timestep", "2timestep"], default="2timestep",
                   help="2timestep matches a KAI0 checkpoint; 1timestep for PI06.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=16, help="Parallel threads for video loading.")
    p.add_argument("--frame-interval", type=int, default=1, help="1 = every frame.")
    p.add_argument("--relative-interval", type=int, default=50, help="Frames ahead for 2timestep relative advantage.")
    p.add_argument("--seg-mode", choices=["equal_width", "equal_count"], default="equal_width",
                   help="equal_width=divide value range into K equal bins (default '等分').")
    p.add_argument("--monotonic", dest="monotonic", action="store_true", default=True,
                   help="Enforce monotonic non-decreasing absolute_value (cumulative max). Default on.")
    p.add_argument("--no-monotonic", dest="monotonic", action="store_false")
    p.add_argument("--exclude-mistakes", action="store_true", help="Drop is_mistake segments from GT.")
    p.add_argument("--output-dir", default="stage_advantage/annotation/stage_seg_results")
    p.add_argument("--episodes", default=None, help="Comma-separated episode indices to force (overrides --num).")
    p.add_argument("--dry-run", action="store_true", help="Select episodes & parse GT only; no model.")
    p.add_argument("--save-trajectories", action="store_true", help="Also dump per-frame abs_value/pred/gt to per_episode.json.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps = list_annotated_episodes(data_root)
    print(f"valid annotated episodes (pass & non-empty): {len(eps)}")

    if args.episodes:
        want = {int(x) for x in args.episodes.split(",") if x.strip()}
        eps = [e for e in eps if e["episode_index"] in want]
    elif args.all:
        pass  # keep every valid episode
    else:
        rng = random.Random(args.seed)
        rng.shuffle(eps)
        eps = eps[: args.num]
    if args.num_shards > 1:
        n = len(eps)
        chunk = (n + args.num_shards - 1) // args.num_shards
        s, e = args.shard_id * chunk, min(args.shard_id * chunk + chunk, n)
        eps = eps[s:e]
    mode = "ALL" if args.all else f"num={args.num} seed={args.seed}"
    if args.num_shards > 1:
        mode += f" shard={args.shard_id}/{args.num_shards}"
    print(f"selected {len(eps)} episodes (mode={mode}, video-source={args.video_source}, method={args.method}, seg={args.seg_mode})")

    if args.dry_run:
        for e in eps:
            gt, k, ends = gt_stages_from_action_config(e["action_config"], e["frame_count"] or 0, args.exclude_mistakes)
            print(f"ep{e['episode_index']:06d} len={e['frame_count']} K={k} seg_ends={ends}")
        print("dry-run OK")
        return

    # Heavy import: only when actually running inference.
    from evaluator import SimpleValueEvaluator

    evaluator = SimpleValueEvaluator(args.config_name, args.ckpt_dir, num_workers=args.num_workers)

    # Aggregate metrics across all episodes
    all_metrics: list[dict] = []
    # For per-stage metrics aggregation
    max_k = 0

    for i, e in enumerate(eps):
        vp = video_paths_for(data_root, e["chunk"], e["episode"], args.video_source)
        if not all(os.path.exists(p) for p in vp):
            print(f"[{i + 1}/{len(eps)}] ep{e['episode_index']:06d} missing videos, skip")
            continue
        cap_len = (e["frame_count"] or 10**9) - 1
        try:
            if args.method == "2timestep":
                results = evaluator.evaluate_video_2timesteps_advantages(
                    video_paths=vp, prompt=args.prompt, batch_size=args.batch_size,
                    frame_interval=args.frame_interval, relative_interval=args.relative_interval,
                    min_frame_index=0, max_frame_index=cap_len, prefetch=True,
                )
            else:
                results = evaluator.evaluate_video_1timestep_advantage(
                    video_paths=vp, prompt=args.prompt, batch_size=args.batch_size,
                    frame_interval=args.frame_interval, relative_interval=args.relative_interval,
                    min_frame_index=0, max_frame_index=cap_len, prefetch=True,
                )
        except Exception as ex:  # noqa: BLE001
            print(f"[{i + 1}/{len(eps)}] ep{e['episode_index']:06d} inference failed: {ex}")
            continue

        # actual length from returned frame indices
        length = max(r["frame_idx"] for r in results) + 1
        av = np.zeros(length, dtype=np.float64)
        for r in results:
            fi = r["frame_idx"]
            if 0 <= fi < length:
                av[fi] = r["absolute_value"]

        gt, k, ends = gt_stages_from_action_config(e["action_config"], length, args.exclude_mistakes)
        if k <= 1:
            print(f"[{i + 1}/{len(eps)}] ep{e['episode_index']:06d} K<=1, skip metrics")
            continue
        max_k = max(max_k, k)

        if args.seg_mode == "equal_width":
            pred = pred_stages_equal_width(av, k, args.monotonic, lo=args.abs_range[0], hi=args.abs_range[1])
        else:
            pred = pred_stages_equal_count(av, k, args.monotonic)

        # Compute comprehensive metrics
        frame_metrics = compute_frame_metrics(pred, gt, k)
        boundary_metrics = compute_boundary_metrics(pred, gt, length)
        stage_iou, mean_iou = compute_stage_iou(pred, gt, k)
        min_iou, overlap_05, overlap_07, overlap_09 = compute_episode_full_overlap(pred, gt, k)

        rec: dict = {
            "episode_index": e["episode_index"],
            "chunk": e["chunk"],
            "episode": e["episode"],
            "length": int(length),
            "num_stages": int(k),
            "method": args.method,
            "seg_mode": args.seg_mode,
            "monotonic": args.monotonic,
            "video_source": args.video_source,
            # Frame metrics
            "frame_accuracy": frame_metrics["frame_accuracy"],
            # Stage IoU
            "stage_iou": stage_iou,
            "mean_iou": mean_iou,
            # Boundary metrics
            "boundary_mae_pred_to_gt": boundary_metrics["boundary_mae_pred_to_gt"],
            "boundary_mae_gt_to_pred": boundary_metrics["boundary_mae_gt_to_pred"],
            "boundary_mae_avg": boundary_metrics["boundary_mae_avg"],
            "boundary_acc@10f": boundary_metrics["acc@10f"],
            "boundary_acc@20f": boundary_metrics["acc@20f"],
            "boundary_acc@30f": boundary_metrics["acc@30f"],
            "boundary_acc@50f": boundary_metrics["acc@50f"],
            "boundary_auc": boundary_metrics["boundary_auc"],
            # Episode full overlap
            "min_iou": min_iou,
            "overlap_05": overlap_05,
            "overlap_07": overlap_07,
            "overlap_09": overlap_09,
            # Per-stage metrics
            "per_stage": frame_metrics["per_stage"],
            # GT info
            "gt_segment_ends": ends,
        }
        if args.save_trajectories:
            rec["abs_value"] = av.tolist()
            rec["pred_stages"] = pred.tolist()
            rec["gt_stages"] = gt.tolist()
            rec["pred_boundaries"] = boundary_metrics["pred_boundaries"]
            rec["gt_boundaries"] = boundary_metrics["gt_boundaries"]

        all_metrics.append(rec)

        # Print per-episode summary
        print(
            f"[{i + 1}/{len(eps)}] ep{e['episode_index']:06d} len={length} K={k} "
            f"acc={frame_metrics['frame_accuracy']:.3f} iou={mean_iou:.3f} "
            f"bMAE={boundary_metrics['boundary_mae_avg']:.1f}f "
            f"B-AUC={boundary_metrics['boundary_auc']:.3f} "
            f"minIoU={min_iou:.3f}"
        )

    if hasattr(evaluator, "shutdown"):
        evaluator.shutdown()

    if not all_metrics:
        print("no episodes evaluated")
        return

    # --------------------------------------------------------------------------- #
    # Aggregate summary
    # --------------------------------------------------------------------------- #
    n_eps = len(all_metrics)

    # Frame accuracy
    accs = [m["frame_accuracy"] for m in all_metrics]

    # Stage IoU - aggregate per-stage IoU
    stage_iou_agg: dict[int, list[float]] = {s: [] for s in range(max_k)}
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
        "≥0.5": sum(m["overlap_05"] for m in all_metrics),
        "≥0.7": sum(m["overlap_07"] for m in all_metrics),
        "≥0.9": sum(m["overlap_09"] for m in all_metrics),
    }

    summary = {
        "num_episodes": n_eps,
        "seed": args.seed,
        "abs_range": [float(args.abs_range[0]), float(args.abs_range[1])],
        "seg_mode": args.seg_mode,
        "monotonic": args.monotonic,
        "method": args.method,
        "video_source": args.video_source,
        # Frame accuracy
        "frame_accuracy": {
            "mean": round(float(np.mean(accs)), 4),
            "std": round(float(np.std(accs)), 4),
            "min": round(float(np.min(accs)), 4),
            "max": round(float(np.max(accs)), 4),
            "median": round(float(np.median(accs)), 4),
        },
        # Stage IoU
        "stage_iou": {
            "per_stage_mean": stage_iou_mean,
            "overall_mean": stage_iou_overall,
        },
        # Boundary metrics
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
        # Episode full overlap
        "episode_full_overlap": {
            "min_iou_mean": round(float(np.mean(min_ious)), 4),
            "min_iou_std": round(float(np.std(min_ious)), 4),
            "overlap_rate@0.5": round(overlap_counts["≥0.5"] / n_eps, 4),
            "overlap_rate@0.7": round(overlap_counts["≥0.7"] / n_eps, 4),
            "overlap_rate@0.9": round(overlap_counts["≥0.9"] / n_eps, 4),
        },
    }

    # Save results
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

    # --------------------------------------------------------------------------- #
    # Print summary
    # --------------------------------------------------------------------------- #
    print("\n" + "=" * 80)
    print("  STAGE SEGMENTATION EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\n  Episodes: {n_eps}")
    print(f"  Method: {args.method}, Seg: {args.seg_mode}, Monotonic: {args.monotonic}")

    print(f"\n  ① Frame-level Accuracy")
    fa = summary["frame_accuracy"]
    print(f"     Mean: {fa['mean']:.4f} ± {fa['std']:.4f}")
    print(f"     Min/Max: {fa['min']:.4f} / {fa['max']:.4f}")

    print(f"\n  ② Stage IoU (Intersection over Union)")
    print(f"     Per-stage mean: {summary['stage_iou']['per_stage_mean']}")
    print(f"     Overall mean: {summary['stage_iou']['overall_mean']:.4f}")

    print(f"\n  ③ Boundary Accuracy (@±10f / @±20f / @±30f / @±50f)")
    b = summary["boundary"]
    print(f"     @±10f: {b['acc@10f']:.4f}")
    print(f"     @±20f: {b['acc@20f']:.4f}")
    print(f"     @±30f: {b['acc@30f']:.4f}")
    print(f"     @±50f: {b['acc@50f']:.4f}")
    print(f"     Boundary AUC (0-50f): {b['auc']:.4f}")
    print(f"     Boundary MAE: {b['mae']['mean']:.2f} ± {b['mae']['std']:.2f} frames")

    print(f"\n  ④ Episode Full Overlap (min-IoU across all stages)")
    eo = summary["episode_full_overlap"]
    print(f"     Mean min-IoU: {eo['min_iou_mean']:.4f} ± {eo['min_iou_std']:.4f}")
    print(f"     Overlap rate @≥0.5: {eo['overlap_rate@0.5']:.4f} ({overlap_counts['≥0.5']}/{n_eps})")
    print(f"     Overlap rate @≥0.7: {eo['overlap_rate@0.7']:.4f} ({overlap_counts['≥0.7']}/{n_eps})")
    print(f"     Overlap rate @≥0.9: {eo['overlap_rate@0.9']:.4f} ({overlap_counts['≥0.9']}/{n_eps})")

    print("\n" + "=" * 80)
    print(f"\nResults saved to: {out_dir}/")
    print(f"  - summary.json     (overall statistics)")
    print(f"  - per_episode.json (detailed per-episode results)")
    print(f"  - per_episode.csv   (compact CSV format)")
    print()


if __name__ == "__main__":
    main()
