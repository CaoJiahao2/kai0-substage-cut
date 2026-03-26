#!/usr/bin/env python3
"""Merge multiple LeRobot-format datasets into one dataset by episode.

This is intended for cases like mixing a ``base`` dataset and a ``dagger``
dataset while keeping the output compatible with the existing OpenPI training
pipeline.

Example:
  python3 scripts/merge_lerobot_datasets.py \
    --source-path data/Task_A/base \
    --source-path data/Task_A/dagger \
    --output-path data/Task_A/base_dagger_merge
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil

import pandas as pd


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=4, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _discover_video_keys(source_path: Path) -> list[str]:
    videos_root = source_path / "videos"
    keys: set[str] = set()
    if not videos_root.exists():
        return []
    for chunk_dir in videos_root.glob("chunk-*"):
        if not chunk_dir.is_dir():
            continue
        for child in chunk_dir.iterdir():
            if child.is_dir():
                keys.add(child.name)
    return sorted(keys)


def _load_tasks_map(source_path: Path) -> tuple[list[str], dict[str, int]]:
    tasks_path = source_path / "meta" / "tasks.jsonl"
    tasks_rows = _read_jsonl(tasks_path)
    task_strings: list[str] = []
    for row in tasks_rows:
        task = row.get("task")
        if task is None:
            continue
        task_strings.append(str(task))
    return task_strings, {task: idx for idx, task in enumerate(task_strings)}


def _normalize_episode_tasks(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _load_episode_counts(source_path: Path) -> list[tuple[int, int]]:
    episodes_stats = sorted(_read_jsonl(source_path / "meta" / "episodes_stats.jsonl"), key=lambda x: x["episode_index"])
    return [
        (int(row["episode_index"]), int(row["stats"]["index"]["count"][0]))
        for row in episodes_stats
    ]


def _pick_episode_subset(
    episode_counts: list[tuple[int, int]],
    target_frames: int,
    *,
    rng: random.Random,
) -> set[int]:
    if target_frames <= 0:
        return set()
    shuffled = list(episode_counts)
    rng.shuffle(shuffled)
    selected: list[int] = []
    total = 0
    for episode_index, frame_count in shuffled:
        selected.append(episode_index)
        total += frame_count
        if total >= target_frames:
            break
    if not selected:
        episode_index, _ = min(episode_counts, key=lambda item: item[1])
        return {episode_index}

    # Greedy trimming to reduce overshoot while keeping episode granularity.
    improved = True
    selected_set = set(selected)
    while improved and len(selected_set) > 1:
        improved = False
        current_total = sum(count for ep, count in episode_counts if ep in selected_set)
        current_gap = abs(current_total - target_frames)
        best_remove = None
        best_gap = current_gap
        for ep, count in episode_counts:
            if ep not in selected_set:
                continue
            candidate_total = current_total - count
            if candidate_total <= 0:
                continue
            candidate_gap = abs(candidate_total - target_frames)
            if candidate_gap < best_gap:
                best_gap = candidate_gap
                best_remove = ep
        if best_remove is not None:
            selected_set.remove(best_remove)
            improved = True
    return selected_set


def _compute_selected_episodes(
    source_paths: list[Path],
    source_weights: list[float] | None,
    *,
    seed: int,
) -> list[set[int] | None]:
    if source_weights is None:
        return [None] * len(source_paths)
    if len(source_weights) != len(source_paths):
        raise ValueError("--source-weight count must match --source-path count.")
    if any(weight <= 0 for weight in source_weights):
        raise ValueError("--source-weight values must be positive.")

    episode_counts_list = [_load_episode_counts(path) for path in source_paths]
    total_frames_list = [sum(count for _, count in episode_counts) for episode_counts in episode_counts_list]
    scale = min(total_frames / weight for total_frames, weight in zip(total_frames_list, source_weights, strict=True))
    target_frames_list = [max(1, int(round(weight * scale))) for weight in source_weights]

    selections: list[set[int] | None] = []
    for idx, (episode_counts, target_frames) in enumerate(zip(episode_counts_list, target_frames_list, strict=True)):
        rng = random.Random(seed + idx)
        selections.append(_pick_episode_subset(episode_counts, target_frames, rng=rng))
    return selections


def _merge_datasets(
    source_paths: list[Path],
    output_path: Path,
    *,
    symlink_videos: bool,
    source_weights: list[float] | None,
    seed: int,
) -> None:
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Output path must be empty: {output_path}")

    source_infos = [_read_json(path / "meta" / "info.json") for path in source_paths]
    chunk_sizes = {int(info["chunks_size"]) for info in source_infos}
    if len(chunk_sizes) != 1:
        raise ValueError(f"All source datasets must share the same chunks_size, got {sorted(chunk_sizes)}")
    chunks_size = chunk_sizes.pop()

    video_keys = _discover_video_keys(source_paths[0])
    for source_path in source_paths[1:]:
        keys = _discover_video_keys(source_path)
        if keys != video_keys:
            raise ValueError(
                f"Video key mismatch.\nfirst={video_keys}\n{source_path}={keys}"
            )

    merged_task_strings: list[str] = []
    merged_task_to_index: dict[str, int] = {}
    source_task_maps: list[dict[int, int]] = []
    for source_path in source_paths:
        task_strings, _ = _load_tasks_map(source_path)
        remap: dict[int, int] = {}
        for old_idx, task in enumerate(task_strings):
            if task not in merged_task_to_index:
                merged_task_to_index[task] = len(merged_task_strings)
                merged_task_strings.append(task)
            remap[old_idx] = merged_task_to_index[task]
        source_task_maps.append(remap)

    merged_episodes_stats: list[dict] = []
    merged_episodes: list[dict] = []
    merged_record_rows: dict[str, list[pd.DataFrame]] = {}
    selected_episodes_by_source = _compute_selected_episodes(source_paths, source_weights, seed=seed)

    total_frames = 0
    next_episode_index = 0
    total_videos = 0

    for source_idx, source_path in enumerate(source_paths):
        old_tasks_remap = source_task_maps[source_idx]
        episodes_stats = sorted(_read_jsonl(source_path / "meta" / "episodes_stats.jsonl"), key=lambda x: x["episode_index"])
        episodes_rows = sorted(_read_jsonl(source_path / "meta" / "episodes.jsonl"), key=lambda x: x["episode_index"])
        selected_episodes = selected_episodes_by_source[source_idx]
        if selected_episodes is not None:
            episodes_stats = [row for row in episodes_stats if int(row["episode_index"]) in selected_episodes]
            episodes_rows = [row for row in episodes_rows if int(row["episode_index"]) in selected_episodes]
        episode_row_map = {int(row["episode_index"]): row for row in episodes_rows}

        local_old_to_new: dict[int, int] = {}
        for episode_stat in episodes_stats:
            old_episode_index = int(episode_stat["episode_index"])
            new_episode_index = next_episode_index
            local_old_to_new[old_episode_index] = new_episode_index
            next_episode_index += 1

            index_count = int(episode_stat["stats"]["index"]["count"][0])
            frame_start = total_frames
            frame_end = total_frames + index_count - 1
            total_frames += index_count

            episode_stat = json.loads(json.dumps(episode_stat))
            episode_stat["episode_index"] = new_episode_index
            episode_stat["stats"]["index"]["min"] = [frame_start]
            episode_stat["stats"]["index"]["max"] = [frame_end]
            episode_stat["stats"]["index"]["mean"] = [(frame_start + frame_end) / 2]
            episode_stat["stats"]["index"]["count"] = [index_count]
            merged_episodes_stats.append(episode_stat)

            episode_row = json.loads(json.dumps(episode_row_map[old_episode_index]))
            episode_row["episode_index"] = new_episode_index
            if "tasks" in episode_row:
                episode_row["tasks"] = _normalize_episode_tasks(episode_row["tasks"])
            merged_episodes.append(episode_row)

            old_parquet = source_path / "data" / f"chunk-{old_episode_index // chunks_size:03d}" / f"episode_{old_episode_index:06d}.parquet"
            new_parquet = output_path / "data" / f"chunk-{new_episode_index // chunks_size:03d}" / f"episode_{new_episode_index:06d}.parquet"
            new_parquet.parent.mkdir(parents=True, exist_ok=True)
            df = pd.read_parquet(old_parquet)
            if "index" in df.columns:
                df["index"] = df["index"] - df["index"].min() + frame_start
            if "episode_index" in df.columns:
                df["episode_index"] = new_episode_index
            if "task_index" in df.columns:
                df["task_index"] = df["task_index"].map(lambda value: old_tasks_remap.get(int(value), int(value)))
            df.to_parquet(new_parquet, index=False)

            for video_key in video_keys:
                old_video = source_path / "videos" / f"chunk-{old_episode_index // chunks_size:03d}" / video_key / f"episode_{old_episode_index:06d}.mp4"
                new_video = output_path / "videos" / f"chunk-{new_episode_index // chunks_size:03d}" / video_key / f"episode_{new_episode_index:06d}.mp4"
                new_video.parent.mkdir(parents=True, exist_ok=True)
                if symlink_videos:
                    os.symlink(old_video.resolve(), new_video)
                else:
                    shutil.copy2(old_video, new_video)
                total_videos += 1

        record_files = [
            name
            for name in os.listdir(source_path / "meta")
            if name.lower().startswith("v") and "record.csv" in name
        ]
        for record_file in record_files:
            df = pd.read_csv(source_path / "meta" / record_file)
            if "video" not in df.columns:
                continue
            old_ep = df["video"].str.extract(r"episode_(\d+)\.mp4$")[0].astype(int)
            df = df.loc[old_ep.isin(local_old_to_new)].copy()
            df["video"] = old_ep.loc[df.index].map(
                lambda ep: f"episode_{local_old_to_new[int(ep)]:06d}.mp4"
            )
            merged_record_rows.setdefault(record_file, []).append(df)

    output_path.mkdir(parents=True, exist_ok=True)
    info = json.loads(json.dumps(source_infos[0]))
    info["total_episodes"] = len(merged_episodes)
    info["total_frames"] = total_frames
    info["total_videos"] = total_videos
    info["total_chunks"] = (len(merged_episodes) + chunks_size - 1) // chunks_size
    info["splits"] = {"train": f"0:{len(merged_episodes)}"}

    _write_json(output_path / "meta" / "info.json", info)
    _write_jsonl(output_path / "meta" / "episodes_stats.jsonl", merged_episodes_stats)
    _write_jsonl(output_path / "meta" / "episodes.jsonl", merged_episodes)
    _write_jsonl(
        output_path / "meta" / "tasks.jsonl",
        [{"task_index": idx, "task": task} for idx, task in enumerate(merged_task_strings)],
    )

    for record_file, dfs in merged_record_rows.items():
        pd.concat(dfs, ignore_index=True).to_csv(output_path / "meta" / record_file, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot datasets by episode.")
    parser.add_argument(
        "--source-path",
        dest="source_paths",
        action="append",
        required=True,
        help="Source dataset path. Pass this flag multiple times; merge order is preserved.",
    )
    parser.add_argument("--output-path", required=True, help="Output merged dataset path.")
    parser.add_argument(
        "--source-weight",
        dest="source_weights",
        action="append",
        type=float,
        default=None,
        help="Relative mixing weight for each source dataset. Pass once per --source-path. "
        "When set, the script selects episode subsets so merged frame counts roughly follow these weights.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when sampling episode subsets.")
    parser.add_argument(
        "--symlink-videos",
        action="store_true",
        help="Symlink videos instead of copying them to save disk space.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_paths = [Path(path).expanduser().resolve() for path in args.source_paths]
    output_path = Path(args.output_path).expanduser().resolve()
    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(f"Source dataset does not exist: {path}")
    _merge_datasets(
        source_paths,
        output_path,
        symlink_videos=bool(args.symlink_videos),
        source_weights=args.source_weights,
        seed=args.seed,
    )
    print(f"Merged {len(source_paths)} datasets into {output_path}")


if __name__ == "__main__":
    main()
