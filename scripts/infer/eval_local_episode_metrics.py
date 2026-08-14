#!/usr/bin/env python3
"""Evaluate local Kai0/OpenPI checkpoints on selected dataset episodes.

This script targets the local Kai0 v3 dataset layout used in this repo:
  data/<Task>/base/meta/info.json
  data/<Task>/base/meta/episodes/chunk-*/file-*.parquet
  data/<Task>/base/data/chunk-*/file-*.parquet
  data/<Task>/base/videos/<camera_key>/chunk-*/file-*.mp4

Example:
  uv run python scripts/eval_local_episode_metrics.py \
    --dataset-path data/Task_B/base \
    --checkpoint-dir model/Task_B/mixed_9 \
    --policy-config pi05_flatten_fold_normal \
    --episodes 0,1
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import tempfile
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_lerobot_stub() -> None:
    """OpenPI policy loading imports `lerobot.datasets` in this environment."""
    try:
        if importlib.util.find_spec("lerobot.datasets.lerobot_dataset") is not None:
            return
    except ModuleNotFoundError:
        pass

    lerobot_pkg = sys.modules.setdefault("lerobot", types.ModuleType("lerobot"))
    if not hasattr(lerobot_pkg, "__path__"):
        lerobot_pkg.__path__ = []  # type: ignore[attr-defined]

    datasets_pkg = sys.modules.setdefault("lerobot.datasets", types.ModuleType("lerobot.datasets"))
    if not hasattr(datasets_pkg, "__path__"):
        datasets_pkg.__path__ = []  # type: ignore[attr-defined]

    dataset_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")

    class _DummyLeRobotDataset:
        pass

    dataset_mod.LeRobotDataset = _DummyLeRobotDataset
    sys.modules["lerobot.datasets.lerobot_dataset"] = dataset_mod


_ensure_lerobot_stub()

from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from scripts.viz import infer_dataset_loss_viz as _infer_helpers
from scripts.viz.rerun_dataset_viz import URDFKinematics
from scripts.viz.rerun_dataset_viz import _build_urdf_to_dataset_map
from scripts.viz.rerun_dataset_viz import _joint_map_from_state
from scripts.viz.rerun_dataset_viz import _rewrite_urdf_for_fk


DEFAULT_CAMERA_KEYS = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate selected dataset episodes with a local OpenPI checkpoint.")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Local dataset root, e.g. data/Task_B/base")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Local checkpoint dir, e.g. model/Task_B/mixed_9")
    parser.add_argument("--policy-config", default="pi05_flatten_fold_normal", help="Config name in openpi.training.config")
    parser.add_argument("--episodes", default="0,1", help="Episode list/range, e.g. '0,1' or '0-3'")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--camera-keys", default=",".join(DEFAULT_CAMERA_KEYS), help="Comma-separated dataset camera keys")
    parser.add_argument("--urdf-path", type=Path, default=_infer_helpers.DEFAULT_URDF_PATH)
    parser.add_argument("--urdf-root", type=Path, default=_infer_helpers.DEFAULT_URDF_ROOT)
    parser.add_argument("--ee-links", default="fl_link8,fr_link8", help="Left/right ee links")
    parser.add_argument("--max-frames-per-episode", type=int, default=0, help="0 means full episode")
    parser.add_argument("--log-every", type=int, default=200, help="Print every N frames within each episode")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional summary JSON output path")
    return parser.parse_args()


def _parse_camera_keys(raw: str) -> list[str]:
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    if not keys:
        raise ValueError("--camera-keys is empty.")
    return keys


def _load_norm_stats(checkpoint_dir: Path) -> dict[str, _normalize.NormStats]:
    direct = checkpoint_dir / "norm_stats.json"
    if direct.exists():
        return _normalize.deserialize_json(direct.read_text())

    matches = sorted(checkpoint_dir.glob("assets/**/norm_stats.json"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find norm_stats.json under {checkpoint_dir}. "
            "Expected either checkpoint_dir/norm_stats.json or checkpoint_dir/assets/**/norm_stats.json."
        )
    return _normalize.deserialize_json(matches[0].read_text())


def _load_tasks_map(dataset_root: Path) -> dict[int, str]:
    tasks_parquet = dataset_root / "meta/tasks.parquet"
    if tasks_parquet.exists():
        table = pq.read_table(tasks_parquet).to_pydict()
        if "task_index" in table and "__index_level_0__" in table:
            return {int(k): str(v) for k, v in zip(table["task_index"], table["__index_level_0__"], strict=True)}

    tasks_jsonl = dataset_root / "meta/tasks.jsonl"
    if tasks_jsonl.exists():
        tasks_map: dict[int, str] = {}
        for line in tasks_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            tasks_map[int(item["task_index"])] = str(item["task"])
        return tasks_map

    return {}


def _read_episode_rows(dataset_root: Path) -> list[dict[str, Any]]:
    meta_root = dataset_root / "meta/episodes"
    files = sorted(meta_root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under {meta_root}")

    rows: list[dict[str, Any]] = []
    for file in files:
        table = pq.read_table(file)
        columns = table.to_pydict()
        for row_idx in range(table.num_rows):
            row = {key: value[row_idx] for key, value in columns.items()}
            rows.append(row)
    rows.sort(key=lambda item: int(item["episode_index"]))
    return rows


def _table_to_python_dict(table: pa.Table) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for column_name in table.column_names:
        column = table[column_name]
        assert isinstance(column, pa.ChunkedArray)
        combined = column.combine_chunks()
        if isinstance(combined, pa.FixedSizeListArray):
            values = combined.flatten().to_numpy()
            out[column_name] = values.reshape(len(combined), combined.type.list_size).tolist()
        elif isinstance(combined, pa.ListArray):
            out[column_name] = combined.to_pylist()
        else:
            out[column_name] = combined.to_pylist()
    return out


def _build_file_start_indices(rows: list[dict[str, Any]], camera_keys: list[str], fps: int) -> tuple[dict[tuple[int, int], int], dict[tuple[str, int, int], int]]:
    data_starts: dict[tuple[int, int], int] = {}
    video_starts: dict[tuple[str, int, int], int] = {}
    for row in rows:
        data_key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        data_start = int(row["dataset_from_index"])
        data_starts[data_key] = min(data_starts.get(data_key, data_start), data_start)

        for camera_key in camera_keys:
            chunk_key = f"videos/{camera_key}/chunk_index"
            file_key = f"videos/{camera_key}/file_index"
            from_key = f"videos/{camera_key}/from_timestamp"
            if chunk_key not in row or file_key not in row or from_key not in row:
                raise KeyError(f"Episode metadata missing video keys for camera '{camera_key}'")
            video_key = (camera_key, int(row[chunk_key]), int(row[file_key]))
            video_start = int(round(float(row[from_key]) * fps))
            video_starts[video_key] = min(video_starts.get(video_key, video_start), video_start)

    return data_starts, video_starts


def _data_file_path(dataset_root: Path, info: dict[str, Any], chunk_index: int, file_index: int) -> Path:
    template = info["data_path"]
    return dataset_root / template.format(chunk_index=chunk_index, file_index=file_index)


def _video_file_path(dataset_root: Path, info: dict[str, Any], camera_key: str, chunk_index: int, file_index: int) -> Path:
    template = info["video_path"]
    return dataset_root / template.format(video_key=camera_key, chunk_index=chunk_index, file_index=file_index)


def _frame_iter(video_path: Path, skip_frames: int, max_frames: int) -> Iterator[np.ndarray]:
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "NONE"
    stream.thread_count = 1

    try:
        seen = 0
        yielded = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                if seen < skip_frames:
                    seen += 1
                    continue
                if yielded >= max_frames:
                    return
                yielded += 1
                seen += 1
                yield frame.to_rgb().to_ndarray()
    finally:
        container.close()


def _episode_prompt(row: dict[str, Any], task_map: dict[int, str]) -> str | None:
    tasks = row.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    task_index = row.get("task_index")
    if task_index is not None and int(task_index) in task_map:
        return task_map[int(task_index)]
    return None


def _load_policy(policy_config: str, checkpoint_dir: Path, dataset_path: Path):
    cfg = _config.get_config(policy_config)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=str(dataset_path.resolve())))
    norm_stats = _load_norm_stats(checkpoint_dir)
    return _policy_config.create_trained_policy(
        cfg,
        checkpoint_dir,
        norm_stats=norm_stats,
        default_prompt=getattr(cfg.data, "default_prompt", None),
    )


def _init_fk(urdf_path: Path, urdf_root: Path | None, ee_links: list[str]) -> tuple[URDFKinematics, dict[str, int], tempfile.TemporaryDirectory[str]]:
    tmp = tempfile.TemporaryDirectory(prefix="kai0_eval_fk_")
    fk_urdf = _rewrite_urdf_for_fk(urdf_path, urdf_root, Path(tmp.name))
    kin = URDFKinematics(fk_urdf)
    urdf_to_dataset = _build_urdf_to_dataset_map([], kin.joint_names)
    if not urdf_to_dataset:
        tmp.cleanup()
        raise RuntimeError("Failed to build URDF to dataset joint mapping.")
    for link_name in ee_links:
        if not kin.has_link(link_name):
            tmp.cleanup()
            raise RuntimeError(f"EE link not found in URDF: {link_name}")
    return kin, urdf_to_dataset, tmp


def main() -> None:
    args = _parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    episodes = _infer_helpers._parse_dataset_episodes(args.episodes)
    if not episodes:
        raise ValueError("--episodes resolved to an empty list.")
    camera_keys = _parse_camera_keys(args.camera_keys)
    ee_links = [item.strip() for item in args.ee_links.split(",") if item.strip()]
    if len(ee_links) != 2:
        raise ValueError("--ee-links must contain exactly two comma-separated link names.")

    info = json.loads((dataset_path / "meta/info.json").read_text())
    fps = int(info["fps"])
    task_map = _load_tasks_map(dataset_path)
    episode_rows = _read_episode_rows(dataset_path)
    row_by_episode = {int(row["episode_index"]): row for row in episode_rows}
    data_file_starts, video_file_starts = _build_file_start_indices(episode_rows, camera_keys, fps)

    missing = [epi for epi in episodes if epi not in row_by_episode]
    if missing:
        raise ValueError(f"Episodes not found in metadata: {missing}")

    policy = _load_policy(args.policy_config, checkpoint_dir, dataset_path)
    kin, urdf_to_dataset, fk_tmp = _init_fk(
        urdf_path=args.urdf_path.expanduser().resolve(),
        urdf_root=args.urdf_root.expanduser().resolve() if args.urdf_root else None,
        ee_links=ee_links,
    )

    table_cache: dict[Path, dict[str, list[Any]]] = {}
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_dir),
        "dataset": str(dataset_path),
        "policy_config": args.policy_config,
        "episodes": episodes,
        "per_episode": {},
    }

    try:
        for episode_index in episodes:
            row = row_by_episode[episode_index]
            prompt = _episode_prompt(row, task_map)

            data_chunk = int(row["data/chunk_index"])
            data_file = int(row["data/file_index"])
            data_path = _data_file_path(dataset_path, info, data_chunk, data_file)
            if data_path not in table_cache:
                table = pq.read_table(
                    data_path,
                    columns=[args.state_key, args.action_key, "frame_index", "episode_index", "timestamp", "task_index"],
                )
                table_cache[data_path] = _table_to_python_dict(table)
            data_cols = table_cache[data_path]

            data_base = data_file_starts[(data_chunk, data_file)]
            local_start = int(row["dataset_from_index"]) - data_base
            local_end = int(row["dataset_to_index"]) - data_base
            total_frames = local_end - local_start
            if args.max_frames_per_episode > 0:
                total_frames = min(total_frames, args.max_frames_per_episode)

            video_iters: dict[str, Iterator[np.ndarray]] = {}
            for camera_key in camera_keys:
                video_chunk = int(row[f"videos/{camera_key}/chunk_index"])
                video_file = int(row[f"videos/{camera_key}/file_index"])
                video_path = _video_file_path(dataset_path, info, camera_key, video_chunk, video_file)
                video_base = video_file_starts[(camera_key, video_chunk, video_file)]
                skip_frames = int(round(float(row[f"videos/{camera_key}/from_timestamp"]) * fps)) - video_base
                video_iters[camera_key] = _frame_iter(video_path, skip_frames=skip_frames, max_frames=total_frames)

            episode_joint_l1: list[float] = []
            episode_ee_l2: list[float] = []

            print(
                f"Evaluating episode {episode_index}: frames={total_frames} prompt={prompt!r} data_file={data_path.name}",
                flush=True,
            )
            for offset in range(total_frames):
                row_index = local_start + offset
                state_vec = np.asarray(data_cols[args.state_key][row_index], dtype=np.float32)
                target_vec = np.asarray(data_cols[args.action_key][row_index], dtype=np.float32).tolist()
                frame_index = int(data_cols["frame_index"][row_index])
                timestamp = float(data_cols["timestamp"][row_index])

                images = {}
                for camera_key in camera_keys:
                    image = next(video_iters[camera_key])
                    short_key = camera_key.split(".")[-1]
                    images[short_key] = np.transpose(image, (2, 0, 1))

                obs: dict[str, Any] = {"state": state_vec, "images": images}
                if prompt is not None:
                    obs["prompt"] = prompt

                out = policy.infer(obs)
                pred_steps = _infer_helpers._extract_action_sequence(out.get("actions")) or []
                if not pred_steps:
                    raise RuntimeError(f"Policy returned no actions for episode {episode_index}, frame {frame_index}")
                pred_vec = pred_steps[0]

                joint_l1 = _infer_helpers._compute_l1(pred_vec, target_vec)
                if joint_l1 is None:
                    raise RuntimeError(f"Failed to compute joint L1 for episode {episode_index}, frame {frame_index}")
                episode_joint_l1.append(float(joint_l1))

                pred_joint_map = _joint_map_from_state(
                    np.asarray(pred_vec, dtype=np.float64),
                    kin.joint_names,
                    urdf_to_dataset,
                    add_symmetric_gripper=True,
                )
                tgt_joint_map = _joint_map_from_state(
                    np.asarray(target_vec, dtype=np.float64),
                    kin.joint_names,
                    urdf_to_dataset,
                    add_symmetric_gripper=True,
                )
                pred_left = kin.compute_link_position(pred_joint_map, ee_links[0])
                pred_right = kin.compute_link_position(pred_joint_map, ee_links[1])
                tgt_left = kin.compute_link_position(tgt_joint_map, ee_links[0])
                tgt_right = kin.compute_link_position(tgt_joint_map, ee_links[1])
                ee_vals: list[float] = []
                if pred_left is not None and tgt_left is not None:
                    ee_vals.append(float(np.linalg.norm(pred_left - tgt_left)))
                if pred_right is not None and tgt_right is not None:
                    ee_vals.append(float(np.linalg.norm(pred_right - tgt_right)))
                if not ee_vals:
                    raise RuntimeError(f"Failed to compute ee distance for episode {episode_index}, frame {frame_index}")
                ee_l2 = float(np.mean(ee_vals))
                episode_ee_l2.append(ee_l2)

                if offset % max(1, args.log_every) == 0:
                    print(
                        f"  frame={frame_index} step={offset} joint_l1={joint_l1:.6f} ee_l2={ee_l2:.6f} ts={timestamp:.3f}",
                        flush=True,
                    )

            episode_summary = {
                "task": prompt,
                "frames": total_frames,
                "joint_l1_error": float(np.mean(episode_joint_l1)),
                "ee_l2_distance": float(np.mean(episode_ee_l2)),
            }
            summary["per_episode"][str(episode_index)] = episode_summary
            print(
                f"Episode {episode_index} done: "
                f"joint_l1={episode_summary['joint_l1_error']:.8f} "
                f"ee_l2={episode_summary['ee_l2_distance']:.8f}",
                flush=True,
            )
    finally:
        fk_tmp.cleanup()

    per_episode = list(summary["per_episode"].values())
    summary["average_over_episodes"] = {
        "joint_l1_error": float(np.mean([item["joint_l1_error"] for item in per_episode])),
        "ee_l2_distance": float(np.mean([item["ee_l2_distance"] for item in per_episode])),
    }
    summary["frame_weighted_average"] = {
        "joint_l1_error": float(
            np.average(
                [item["joint_l1_error"] for item in per_episode],
                weights=[item["frames"] for item in per_episode],
            )
        ),
        "ee_l2_distance": float(
            np.average(
                [item["ee_l2_distance"] for item in per_episode],
                weights=[item["frames"] for item in per_episode],
            )
        ),
    }

    result_text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(result_text)
    if args.save_json is not None:
        out_path = args.save_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result_text + "\n", encoding="utf-8")
        print(f"Saved summary to {out_path}", flush=True)


if __name__ == "__main__":
    main()
