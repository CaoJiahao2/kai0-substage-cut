#!/usr/bin/env python3
"""Scan LeRobot video files with ffprobe, delete corrupted ones, then re-download.

Example:
  python3 scripts/repair_lerobot_videos.py \
    --repo-id OpenDriveLab-org/Kai0 \
    --root /home/ubuntu/Data/Kai0_newdataset/Task_A/base
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import tqdm
from huggingface_hub import hf_hub_download

EP_RE = re.compile(r"episode_(\d+)\.mp4$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair corrupted LeRobot videos by re-downloading episodes.")
    parser.add_argument("--repo-id", required=True, help="Hub repo id, e.g. OpenDriveLab-org/Kai0")
    parser.add_argument("--root", required=True, type=Path, help="Local dataset root directory")
    parser.add_argument(
        "--video-glob",
        default="videos/chunk-*/ob*/episode_*.mp4",
        help="Glob under root to find video files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report corrupted files.")
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="Path to ffprobe binary.",
    )
    return parser.parse_args()


def is_video_ok(path: Path, ffprobe: str) -> bool:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def episode_index_from_path(path: Path) -> int | None:
    m = EP_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1))


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    videos = sorted(root.glob(args.video_glob))
    if not videos:
        print(f"No videos found under: {root} ({args.video_glob})")
        return

    bad_files: list[Path] = []
    bad_episodes: set[int] = set()

    for path in tqdm.tqdm(videos, desc="Scanning videos"):
        if not is_video_ok(path, args.ffprobe):
            print(path)
            print(" is bad !!!!")
            bad_files.append(path)
            ep_idx = episode_index_from_path(path)
            if ep_idx is not None:
                bad_episodes.add(ep_idx)

    if not bad_files:
        print("No corrupted videos found.")
        return

    print(f"Found {len(bad_files)} corrupted videos across {len(bad_episodes)} episodes.")
    for path in bad_files:
        print(f"[bad] {path}")

    if args.dry_run:
        print("Dry-run enabled: no deletions or downloads performed.")
        return

    for path in bad_files:
        path.unlink(missing_ok=True)

    # Re-download only affected episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Re-download only the corrupted video files via hf_hub_download (no metadata/data files).
    for path in tqdm.tqdm(bad_files, desc="Re-downloading"):
        rel = path.relative_to(root)
        hf_hub_download(
            repo_id=args.repo_id,
            filename=rel.name,
            subfolder=str(rel.parent),
            repo_type="dataset",
            local_dir=str(root),
        )

    # Re-check episodes
    still_bad = []
    for path in bad_files:
        if not path.exists() or not is_video_ok(path, args.ffprobe):
            still_bad.append(path)
    if still_bad:
        print(f"Still corrupted after re-download: {len(still_bad)} files")
        for path in still_bad:
            print(f"[bad] {path}")
    else:
        print("Re-download complete. All previously corrupted files are now OK.")


def check_video(video_path):
    print(is_video_ok(video_path,'ffprobe'))

if __name__ == "__main__":
    #main()
    vi_path="/home/ubuntu/Data/Kai0_newdataset/Task_A/base/videos/chunk-000/observation.images.top_head/episode_000176.mp4"
    v2_path="/home/ubuntu/Data/Kai0_newdataset/Task_A/base/videos/chunk-002/observation.images.hand_left/episode_002953.mp4"
    v4_path = "/home/ubuntu/Data/Kai0_newdataset/Task_A/base/videos/chunk-002/observation.images.top_head/episode_002548.mp4"
    v3_path="/home/ubuntu/Data/Kai0_newdataset/Task_A/base/videos/chunk-002/observation.images.hand_left/episode_002954.mp4"
    
    check_video(vi_path)
