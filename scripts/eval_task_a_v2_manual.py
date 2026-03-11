#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

_INFER_VIZ_PATH = Path(__file__).with_name("infer_dataset_loss_viz.py")
_INFER_VIZ_SPEC = importlib.util.spec_from_file_location("infer_dataset_loss_viz", _INFER_VIZ_PATH)
if _INFER_VIZ_SPEC is None or _INFER_VIZ_SPEC.loader is None:
    raise RuntimeError(f"Failed to load helper module: {_INFER_VIZ_PATH}")
infer_viz = importlib.util.module_from_spec(_INFER_VIZ_SPEC)
_INFER_VIZ_SPEC.loader.exec_module(infer_viz)


def _find_single(pattern: str) -> str:
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


def _load_tasks_map(tasks_jsonl: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with tasks_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            out[int(item["task_index"])] = str(item["task"])
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task_A v2 dataset manually and report mean losses.")
    parser.add_argument("--repo-id", type=Path, default=Path("/home/magiclab/kai0/data/Task_A/base"))
    parser.add_argument("--episodes", default="0,1")
    parser.add_argument("--policy-config", default="pi05_flatten_fold_normal")
    parser.add_argument("--policy-dir", type=Path, default=Path("/home/magiclab/kai0/model/Task_A/mixed_1"))
    parser.add_argument("--urdf-path", type=Path, default=Path("/home/magiclab/projects/mobile_aloha_sim_ros2/aloha_new_description/urdf/aloha_new.urdf"))
    parser.add_argument("--urdf-root", type=Path, default=Path("/home/magiclab/projects/mobile_aloha_sim_ros2"))
    parser.add_argument("--ee-links", default="fl_link8,fr_link8")
    parser.add_argument("--max-frames-per-episode", type=int, default=0, help="0 means full episode.")
    parser.add_argument("--save-csv", type=Path, default=Path("/home/magiclab/kai0/outputs/loss_eval/task_a_ep01_manual.csv"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    episodes = [int(x.strip()) for x in args.episodes.split(",") if x.strip()]
    if not episodes:
        raise ValueError("No episodes selected.")

    repo = args.repo_id.expanduser().resolve()
    logging.info("Loading policy config=%s dir=%s", args.policy_config, args.policy_dir)
    policy = _policy_config.create_trained_policy(_config.get_config(args.policy_config), args.policy_dir.expanduser().resolve())
    logging.info("Policy loaded")
    tasks_map = _load_tasks_map(repo / "meta/tasks.jsonl")

    ee_links = [x.strip() for x in args.ee_links.split(",") if x.strip()]
    fk_solvers, fk_maps, fk_tmp = infer_viz._init_fk(
        dataset_joint_names=[],
        urdf_path=args.urdf_path.expanduser().resolve(),
        urdf_root=args.urdf_root.expanduser().resolve(),
        ee_links=ee_links,
    )
    logging.info("FK enabled=%s", fk_solvers is not None)

    camera_map = [
        ("top_head", "observation.images.top_head"),
        ("hand_left", "observation.images.hand_left"),
        ("hand_right", "observation.images.hand_right"),
    ]

    rows: list[dict[str, float | int | None]] = []
    try:
        for ep in episodes:
            logging.info("Episode %d start", ep)
            parquet_path = _find_single(str(repo / f"data/chunk-*/episode_{ep:06d}.parquet"))
            df = pd.read_parquet(parquet_path).sort_values("frame_index").reset_index(drop=True)

            caps: dict[str, cv2.VideoCapture] = {}
            for _, cam_key in camera_map:
                video_path = _find_single(str(repo / f"videos/chunk-*/{cam_key}/episode_{ep:06d}.mp4"))
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open video: {video_path}")
                caps[cam_key] = cap

            try:
                num_rows = len(df) if args.max_frames_per_episode <= 0 else min(len(df), args.max_frames_per_episode)
                for i in range(num_rows):
                    if i == 0 or (i + 1) % 100 == 0:
                        logging.info("Episode %d frame %d/%d", ep, i + 1, num_rows)
                    r = df.iloc[i]
                    frame_index = int(r["frame_index"])
                    images: dict[str, np.ndarray] = {}
                    for policy_key, cam_key in camera_map:
                        cap = caps[cam_key]
                        ok, frame = cap.read()
                        if not ok:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
                            ok, frame = cap.read()
                            if not ok:
                                raise RuntimeError(f"Failed reading ep={ep} frame={frame_index} cam={cam_key}")
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        images[policy_key] = np.transpose(frame, (2, 0, 1)).astype(np.uint8)

                    obs = {
                        "state": np.asarray(r["observation.state"], dtype=np.float32).reshape(-1),
                        "images": images,
                    }
                    task_idx = int(r["task_index"]) if "task_index" in df.columns else None
                    if task_idx is not None and task_idx in tasks_map:
                        obs["prompt"] = tasks_map[task_idx]

                    out = policy.infer(obs)
                    pred_steps = infer_viz._extract_action_sequence(out.get("actions")) or []
                    pred_vec = pred_steps[0] if pred_steps else None
                    tgt_vec = np.asarray(r["action"], dtype=np.float32).reshape(-1).tolist()

                    loss_l1 = infer_viz._compute_l1(pred_vec, tgt_vec)
                    loss_ee = None
                    if fk_solvers is not None and fk_maps is not None and pred_vec is not None:
                        dists: list[float] = []
                        for link_name, solver in fk_solvers.items():
                            mapping = fk_maps[link_name]
                            pred_pt = infer_viz._vector_to_fk_point(pred_vec, solver, mapping, "rad")
                            tgt_pt = infer_viz._vector_to_fk_point(tgt_vec, solver, mapping, "rad")
                            dist = infer_viz._compute_l2_distance(pred_pt, tgt_pt)
                            if dist is not None:
                                dists.append(float(dist))
                        if dists:
                            loss_ee = float(np.mean(dists))

                    rows.append(
                        {
                            "episode_index": ep,
                            "frame_index": frame_index,
                            "loss_l1": float(loss_l1) if loss_l1 is not None else None,
                            "loss_ee": float(loss_ee) if loss_ee is not None else None,
                        }
                    )
            finally:
                for cap in caps.values():
                    cap.release()
    finally:
        if fk_tmp is not None:
            fk_tmp.cleanup()

    args.save_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.save_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_index", "frame_index", "loss_l1", "loss_ee"])
        writer.writeheader()
        writer.writerows(rows)

    l1_vals = [float(r["loss_l1"]) for r in rows if r["loss_l1"] is not None]
    ee_vals = [float(r["loss_ee"]) for r in rows if r["loss_ee"] is not None]
    print(f"samples={len(rows)}")
    print(f"mean_joint_l1={float(np.mean(l1_vals)) if l1_vals else float('nan')}")
    print(f"mean_ee_l2={float(np.mean(ee_vals)) if ee_vals else float('nan')}")
    print(f"csv={args.save_csv}")


if __name__ == "__main__":
    main()
