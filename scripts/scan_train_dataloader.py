#!/usr/bin/env python3
"""Scan dataset by iterating the exact training data pipeline.

This script reuses `openpi.training.data_loader.create_data_loader`, so the loading,
decode and transforms are aligned with training behavior.

Example:
  python scripts/scan_train_dataloader.py pi05_flatten_fold_normal_follow_paper \
    --max_batches 2000 --framework pytorch --num_workers 16
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import tqdm
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def _to_numpy(x):
    if hasattr(x, "detach") and hasattr(x, "cpu"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main(
    config_name: str,
    max_batches: int | None = None,
    framework: str = "pytorch",
    shuffle: bool = True,
    num_workers: int | None = None,
    batch_size: int | None = None,
) -> None:
    if framework not in ("jax", "pytorch"):
        raise ValueError("--framework must be one of: jax, pytorch")

    config = _config.get_config(config_name)
    if num_workers is not None:
        config = dataclasses.replace(config, num_workers=num_workers)
    if batch_size is not None:
        config = dataclasses.replace(config, batch_size=batch_size)

    print(f"config={config.name} framework={framework} batch_size={config.batch_size} num_workers={config.num_workers}")
    loader = _data_loader.create_data_loader(
        config,
        framework=framework,
        shuffle=shuffle,
        num_batches=max_batches,
        skip_norm_stats=config.skip_norm_stats,
    )

    start = time.time()
    bad_numeric_batches = 0
    sample_count = 0
    pbar = tqdm.tqdm(total=max_batches, desc="Scanning")

    for i, (obs, actions) in enumerate(loader):
        actions_np = _to_numpy(actions)
        if not np.isfinite(actions_np).all():
            bad_numeric_batches += 1
        sample_count += int(actions_np.shape[0])
        pbar.update(1)
        if max_batches is not None and (i + 1) >= max_batches:
            break

    elapsed = time.time() - start
    pbar.close()

    batches = (i + 1) if "i" in locals() else 0
    rate = batches / elapsed if elapsed > 0 else 0.0
    print("=== Scan Summary ===")
    print(f"batches={batches}")
    print(f"samples={sample_count}")
    print(f"elapsed_sec={elapsed:.2f}")
    print(f"batches_per_sec={rate:.3f}")
    print(f"bad_numeric_batches={bad_numeric_batches}")


if __name__ == "__main__":
    tyro.cli(main)

