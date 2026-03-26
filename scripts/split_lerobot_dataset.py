#!/usr/bin/env python3
"""Split a LeRobot dataset into three subsets: base_split1/2/3.

Default source:
  /home/ubuntu/Data/Kai0_newdataset/Task_A/base
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

import numpy as np
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.split_lerobot import split_lerobot_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset into base_split1/base_split2/base_split3.")
    parser.add_argument(
        "--source-path",
        type=Path,
        default=Path("/home/ubuntu/Data/Kai0_newdataset/Task_A/base"),
        help="Source LeRobot dataset path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for episode shuffling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into existing output folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_path = args.source_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")

    parent = source_path.parent
    out_dirs = [parent / f"{source_path.name}_split{i}" for i in (1, 2, 3)]

    for out_dir in out_dirs:
        if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"Output path already exists and is not empty: {out_dir}. "
                "Use --overwrite to continue."
            )
        out_dir.mkdir(parents=True, exist_ok=True)

    info_path = source_path / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    total_episodes = int(info["total_episodes"])

    episode_indices = list(range(total_episodes))
    random.seed(args.seed)
    random.shuffle(episode_indices)
    splits = np.array_split(episode_indices, 3)

    tasks = [
        (source_path, out_dirs[i], splits[i].tolist(), i + 1)
        for i in range(3)
    ]
    max_workers = max(1, min(args.workers, 3, mp.cpu_count() or 1))
    with mp.Pool(processes=max_workers) as pool:
        pool.starmap(split_lerobot_data, tasks)

    print("Split completed:")
    for out_dir, split in zip(out_dirs, splits, strict=True):
        print(f"  {out_dir}  (episodes={len(split)})")


if __name__ == "__main__":
    main()
