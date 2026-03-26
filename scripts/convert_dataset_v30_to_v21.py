#!/usr/bin/env python3
"""Convert local LeRobot V3 layout to LeRobot V2.1 episode-file layout.

V3 layout example:
  meta/episodes/chunk-*/file-*.parquet
  data/chunk-*/file-*.parquet
  videos/<camera_key>/chunk-*/file-*.mp4

V2.1 output layout:
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  meta/tasks.jsonl
  data/chunk-*/episode_*.parquet
  videos/chunk-*/<camera_key>/episode_*.mp4

Example:
  python scripts/convert_lerobot_v3_to_v21.py \
    --src /path/to/v3_dataset \
    --dst /path/to/v21_dataset \
    --video-mode copy
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


@dataclass(frozen=True)
class EpisodeRecord:
    old_episode_index: int
    new_episode_index: int
    data_chunk_index: int
    data_file_index: int
    data_from: int
    data_to: int
    length: int
    task_index: int | None
    tasks: list[str]
    row: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LeRobot V3 dataset to V2.1 layout.")
    parser.add_argument("--src", type=Path, required=True, help="Source V3 dataset root.")
    parser.add_argument("--dst", type=Path, required=True, help="Destination V2.1 dataset root.")
    parser.add_argument("--chunks-size", type=int, default=1000, help="Episode chunk size for V2.1 output.")
    parser.add_argument(
        "--video-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to materialize videos in destination.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if it exists.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_episode_rows(src: Path) -> list[dict[str, Any]]:
    meta_episodes = src / "meta" / "episodes"
    files = sorted(meta_episodes.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under {meta_episodes}")

    rows: list[dict[str, Any]] = []
    for file in files:
        table = pq.read_table(file)
        columns = table.to_pydict()
        for i in range(table.num_rows):
            rows.append({k: v[i] for k, v in columns.items()})
    rows.sort(key=lambda item: int(item["episode_index"]))
    return rows


def _load_tasks_map(src: Path, rows: list[dict[str, Any]]) -> dict[int, str]:
    tasks_parquet = src / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        table = pq.read_table(tasks_parquet).to_pydict()
        if "task_index" in table and "__index_level_0__" in table:
            return {int(k): str(v) for k, v in zip(table["task_index"], table["__index_level_0__"], strict=True)}

    tasks_jsonl = src / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        out: dict[int, str] = {}
        for line in tasks_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            out[int(item["task_index"])] = str(item["task"])
        return out

    out: dict[int, str] = {}
    next_idx = 0
    for row in rows:
        task_index = row.get("task_index")
        tasks = row.get("tasks")
        if task_index is None or not isinstance(tasks, list) or not tasks:
            continue
        if int(task_index) not in out:
            out[int(task_index)] = str(tasks[0])
            next_idx = max(next_idx, int(task_index) + 1)

    for row in rows:
        tasks = row.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            continue
        task = str(tasks[0])
        if task in out.values():
            continue
        out[next_idx] = task
        next_idx += 1
    return out


def _extract_tasks(row: dict[str, Any], tasks_map: dict[int, str]) -> tuple[int | None, list[str]]:
    tasks = row.get("tasks")
    if isinstance(tasks, list) and tasks:
        return None, [str(x) for x in tasks]

    task_index = row.get("task_index")
    if task_index is not None:
        idx = int(task_index)
        task = tasks_map.get(idx)
        if task is not None:
            return idx, [task]
        return idx, []

    return None, []


def _build_episode_records(rows: list[dict[str, Any]], tasks_map: dict[int, str]) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for new_idx, row in enumerate(rows):
        data_from = int(row["dataset_from_index"])
        data_to = int(row["dataset_to_index"])
        length = data_to - data_from
        if length <= 0:
            raise ValueError(f"Episode {row['episode_index']} has non-positive length: {length}")
        task_index, tasks = _extract_tasks(row, tasks_map)
        records.append(
            EpisodeRecord(
                old_episode_index=int(row["episode_index"]),
                new_episode_index=new_idx,
                data_chunk_index=int(row["data/chunk_index"]),
                data_file_index=int(row["data/file_index"]),
                data_from=data_from,
                data_to=data_to,
                length=length,
                task_index=task_index,
                tasks=tasks,
                row=row,
            )
        )
    return records


def _video_src_path(src: Path, info: dict[str, Any], camera_key: str, row: dict[str, Any]) -> Path:
    return src / info["video_path"].format(
        video_key=camera_key,
        chunk_index=int(row[f"videos/{camera_key}/chunk_index"]),
        file_index=int(row[f"videos/{camera_key}/file_index"]),
    )


def _v21_data_dst_path(dst: Path, chunks_size: int, episode_index: int) -> Path:
    chunk = episode_index // chunks_size
    return dst / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _v21_video_dst_path(dst: Path, chunks_size: int, episode_index: int, camera_key: str) -> Path:
    chunk = episode_index // chunks_size
    return dst / "videos" / f"chunk-{chunk:03d}" / camera_key / f"episode_{episode_index:06d}.mp4"


def _materialize_video(src_path: Path, dst_path: Path, mode: str) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()

    if mode == "copy":
        shutil.copy2(src_path, dst_path)
        return
    if mode == "hardlink":
        dst_path.hardlink_to(src_path)
        return
    if mode == "symlink":
        dst_path.symlink_to(src_path.resolve())
        return
    raise ValueError(f"Unsupported video mode: {mode}")


def _write_tasks_jsonl(dst: Path, tasks_map: dict[int, str]) -> None:
    meta = dst / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    with (meta / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for task_index in sorted(tasks_map):
            f.write(json.dumps({"task_index": int(task_index), "task": tasks_map[task_index]}, ensure_ascii=False) + "\n")


def _write_episodes_jsonl(dst: Path, records: list[EpisodeRecord]) -> None:
    with (dst / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            item = {
                "episode_index": record.new_episode_index,
                "tasks": record.tasks,
                # Use the slice-derived length as source-of-truth. Some V3 exports
                # may have stale `length` metadata that is off by one on boundaries.
                "length": int(record.length),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _index_stats(global_start: int, length: int) -> dict[str, Any]:
    global_end = global_start + length - 1
    return {
        "min": [int(global_start)],
        "max": [int(global_end)],
        "mean": [float((global_start + global_end) / 2)],
        "std": [float(np.std(np.arange(global_start, global_end + 1, dtype=np.float64)))],
        "count": [int(length)],
    }


def _extract_stats_from_row(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Recover nested stats dict from flattened V3 episode row columns.

    Expected flattened names include:
      stats/index/min, stats/action/max, ...
    """
    out: dict[str, dict[str, Any]] = {}
    for key, value in row.items():
        if not isinstance(key, str) or not key.startswith("stats/"):
            continue
        parts = key.split("/")
        if len(parts) != 3:
            continue
        _, feature_name, stat_name = parts
        out.setdefault(feature_name, {})[stat_name] = value
    return out


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_to_json_compatible(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_json_compatible(v) for k, v in value.items()}
    return value


def _write_episodes_stats_jsonl(dst: Path, records: list[EpisodeRecord]) -> int:
    global_index = 0
    with (dst / "meta" / "episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            stats = _extract_stats_from_row(record.row)
            if "index" not in stats:
                stats["index"] = _index_stats(global_index, record.length)
            else:
                stats["index"] = _index_stats(global_index, record.length)
            item = {"episode_index": record.new_episode_index, "stats": _to_json_compatible(stats)}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            global_index += record.length
    return global_index


def _convert_data_files(
    src: Path,
    dst: Path,
    chunks_size: int,
    records: list[EpisodeRecord],
) -> None:
    # NOTE:
    # Some V3 datasets converted from V2.1 contain off-by-one `data/file_index`
    # on episode boundaries. To be robust, infer source file from global
    # dataset_from/to indices and real file row counts instead of trusting
    # per-episode data/file_index metadata.
    data_files = sorted((src / "data").rglob("file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No source data files found under {src / 'data'}")

    segments: list[tuple[int, int, Path]] = []
    running = 0
    for path in data_files:
        rows = pq.read_metadata(path).num_rows
        segments.append((running, running + rows, path))
        running += rows

    segment_starts = [start for start, _, _ in segments]
    table_cache: dict[Path, pd.DataFrame] = {}
    global_index = 0
    for record in tqdm(records, desc="Converting data parquet"):
        seg_idx = bisect_right(segment_starts, record.data_from) - 1
        if seg_idx < 0:
            raise ValueError(f"Episode {record.old_episode_index} has invalid dataset_from_index={record.data_from}")
        seg_start, seg_end, src_path = segments[seg_idx]
        if record.data_to > seg_end:
            raise ValueError(
                f"Episode {record.old_episode_index} spans multiple data files: "
                f"[{record.data_from}, {record.data_to}) not within [{seg_start}, {seg_end})"
            )

        df = table_cache.get(src_path)
        if df is None:
            df = pd.read_parquet(src_path)
            table_cache[src_path] = df

        local_start = record.data_from - seg_start
        local_end = record.data_to - seg_start
        if local_start < 0 or local_end > len(df):
            raise ValueError(
                f"Episode {record.old_episode_index} out of file bounds after remap: "
                f"slice [{local_start}, {local_end}) for len={len(df)} in {src_path}"
            )

        out = df.iloc[local_start:local_end].copy()
        out["episode_index"] = int(record.new_episode_index)
        out["index"] = np.arange(global_index, global_index + record.length, dtype=np.int64)

        dst_path = _v21_data_dst_path(dst, chunks_size, record.new_episode_index)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(dst_path, index=False)
        global_index += record.length


def _convert_videos(
    src: Path,
    dst: Path,
    info: dict[str, Any],
    chunks_size: int,
    records: list[EpisodeRecord],
    camera_keys: list[str],
    mode: str,
) -> None:
    for record in tqdm(records, desc="Converting videos"):
        for camera_key in camera_keys:
            src_path = _video_src_path(src, info, camera_key, record.row)
            if not src_path.exists():
                raise FileNotFoundError(f"Missing source video: {src_path}")
            dst_path = _v21_video_dst_path(dst, chunks_size, record.new_episode_index, camera_key)
            _materialize_video(src_path, dst_path, mode)


def _write_info_json(
    dst: Path,
    src_info: dict[str, Any],
    chunks_size: int,
    total_episodes: int,
    total_frames: int,
    total_videos: int,
) -> None:
    out_info = dict(src_info)
    out_info["codebase_version"] = "v2.1"
    out_info["chunks_size"] = int(chunks_size)
    out_info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    out_info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    out_info["total_episodes"] = int(total_episodes)
    out_info["total_frames"] = int(total_frames)
    out_info["total_videos"] = int(total_videos)
    out_info["total_chunks"] = int(math.ceil(total_episodes / chunks_size)) if total_episodes else 0
    out_info["splits"] = {"train": f"0:{total_episodes}"}

    with (dst / "meta" / "info.json").open("w", encoding="utf-8") as f:
        json.dump(out_info, f, ensure_ascii=False, indent=2)


def _prepare_dst(dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst}. Use --overwrite to replace.")
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = _parse_args()
    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}")

    info = _read_json(src / "meta" / "info.json")
    rows = _load_episode_rows(src)
    tasks_map = _load_tasks_map(src, rows)
    records = _build_episode_records(rows, tasks_map)
    camera_keys = [k for k, spec in info.get("features", {}).items() if isinstance(spec, dict) and spec.get("dtype") == "video"]
    if not camera_keys:
        raise ValueError("No video features found in meta/info.json.")

    _prepare_dst(dst, overwrite=args.overwrite)
    _write_tasks_jsonl(dst, tasks_map)
    _write_episodes_jsonl(dst, records)
    total_frames = _write_episodes_stats_jsonl(dst, records)
    _convert_data_files(src, dst, args.chunks_size, records)
    _convert_videos(src, dst, info, args.chunks_size, records, camera_keys, args.video_mode)
    _write_info_json(
        dst=dst,
        src_info=info,
        chunks_size=args.chunks_size,
        total_episodes=len(records),
        total_frames=total_frames,
        total_videos=len(records) * len(camera_keys),
    )

    print("Conversion complete.")
    print(f"  src: {src}")
    print(f"  dst: {dst}")
    print(f"  episodes: {len(records)}")
    print(f"  frames: {total_frames}")
    print(f"  camera_keys: {camera_keys}")


if __name__ == "__main__":
    main()
