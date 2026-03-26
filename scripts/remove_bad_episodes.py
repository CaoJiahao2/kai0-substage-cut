#!/usr/bin/env python3
"""Create a cleaned LeRobot dataset by excluding bad episodes.

This writes a new dataset directory with consistent metadata and reindexed episodes.

Example:
  python scripts/remove_bad_episodes.py \
    --source_path /home/ubuntu/Data/Kai0_newdataset/Task_A/base \
    --dst_path /home/ubuntu/Data/Kai0_newdataset/Task_A/base_clean \
    --exclude_episodes 176,2548,2953,2954
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from split_lerobot import split_lerobot_data


def _parse_episode_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return sorted(set(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove bad episodes by rebuilding a cleaned LeRobot dataset.")
    parser.add_argument("--source_path", type=str, required=True, help="Path to source LeRobot dataset root.")
    parser.add_argument("--dst_path", type=str, required=True, help="Path to write cleaned dataset.")
    parser.add_argument(
        "--exclude_episodes",
        type=str,
        default="",
        help="Comma-separated episode indices to exclude, e.g. '176,2548,2953,2954'.",
    )
    parser.add_argument(
        "--exclude_file",
        type=str,
        default="",
        help="Optional text file containing episode indices (one per line).",
    )
    args = parser.parse_args()

    source = Path(args.source_path).expanduser().resolve()
    target = Path(args.dst_path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    with open(source / "meta" / "info.json", "r") as f:
        info = json.load(f)
    total_episodes = int(info["total_episodes"])
    all_episodes = set(range(total_episodes))

    excluded = set(_parse_episode_list(args.exclude_episodes))
    if args.exclude_file:
        with open(args.exclude_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    excluded.add(int(line))

    invalid = sorted(ep for ep in excluded if ep < 0 or ep >= total_episodes)
    if invalid:
        raise ValueError(f"Found invalid episode indices (out of range 0..{total_episodes - 1}): {invalid[:20]}")

    kept = sorted(all_episodes - excluded)
    if not kept:
        raise ValueError("No episodes left after exclusion.")

    print(f"Total episodes: {total_episodes}")
    print(f"Excluded episodes: {len(excluded)}")
    print(f"Kept episodes: {len(kept)}")
    split_lerobot_data(source, target, kept, 0)
    print(f"Cleaned dataset written to: {target}")


if __name__ == "__main__":
    main()

