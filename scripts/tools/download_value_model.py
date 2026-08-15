#!/usr/bin/env python3
"""
Download the Kai0 trained value model (Advantage Estimator) from Hugging Face
to the repo's ./value_model directory.

Run from the repository root:
    python scripts/tools/download_value_model.py

Optional: custom output path / repo id / sub-folder:
    python scripts/tools/download_value_model.py --local-dir ./value_model --folder adv-flatten-fold-run16

Notes:
  * 国内网络若无法直连 huggingface.co，可走镜像：
        HF_ENDPOINT=https://hf-mirror.com python scripts/tools/download_value_model.py
  * 下载到 <local_dir>/<folder>/，得到 model.safetensors + metadata.pt（不含 optimizer.pt）。
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Process
from pathlib import Path


def get_repo_root() -> Path:
    """Return the repository root (directory containing .git)."""
    path = Path(__file__).resolve().parent.parent.parent
    if (path / ".git").exists():
        return path
    # Fallback: assume cwd is repo root
    return Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Kai0 value model (Advantage Estimator) to ./value_model (or --local-dir)."
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Directory to save the value model (default: <repo_root>/value_model)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Jiahao28/Kai0-value-model",
        help="Hugging Face repo id that hosts the value model (default: Jiahao28/Kai0-value-model)",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="adv-flatten-fold-run16",
        help="Sub-folder inside the repo to download (default: adv-flatten-fold-run16; use 'all' for whole repo)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        print("Install huggingface_hub first: pip install huggingface_hub", file=sys.stderr)
        return 1

    repo_root = get_repo_root()
    local_dir = Path(args.local_dir) if args.local_dir else repo_root / "value_model"
    local_dir = local_dir.resolve()

    allow_patterns = None
    if args.folder != "all":
        # Download only the chosen version folder + README.
        allow_patterns = [f"{args.folder}/*", "README.md"]

    print(f"Downloading value model to {local_dir}")
    print(f"Repo: {args.repo_id}" + (f", folder: {args.folder}" if args.folder != "all" else " (whole repo)"))

    # Run snapshot_download in a separate process so Ctrl+C in the main process
    # can reliably terminate the download, even if the library swallows signals.
    def _worker():
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )

    proc = Process(target=_worker)
    proc.start()

    try:
        proc.join()
    except KeyboardInterrupt:
        print(
            "\nValue model download interrupted by user (Ctrl+C). Terminating download process...",
            file=sys.stderr,
        )
        proc.terminate()
        proc.join()
        print("Partial download may remain in:", local_dir, file=sys.stderr)
        return 130

    if proc.exitcode != 0:
        print(f"\nValue model download process exited with code {proc.exitcode}", file=sys.stderr)
        return proc.exitcode or 1

    print(f"\nDone. Value model is at: {local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
