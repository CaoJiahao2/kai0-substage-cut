"""
Evaluate action L1 loss (MAE) on the training set using a JAX checkpoint.

Usage examples:
  python scripts/eval_train_l1_jax.py debug --checkpoint /path/to/ckpt_dir/100000
  python scripts/eval_train_l1_jax.py debug --checkpoint /path/to/exp_ckpt_dir --max_batches 100
  python scripts/eval_train_l1_jax.py debug --num_denoise_steps 10
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import tqdm

import openpi.models.model as _model
import openpi.shared.nnx_utils as _nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data


def _resolve_jax_checkpoint(ckpt_arg: str | None, cfg: _config.TrainConfig) -> pathlib.Path:
    """Resolve checkpoint and return the directory that contains `params/`."""
    if ckpt_arg is not None:
        p = pathlib.Path(ckpt_arg).expanduser().resolve()
    elif cfg.weight_loader is not None and hasattr(cfg.weight_loader, "params_path"):
        p = pathlib.Path(cfg.weight_loader.params_path).expanduser().resolve()
    else:
        p = cfg.checkpoint_dir

    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")

    if p.is_file():
        raise FileNotFoundError(f"Expected a checkpoint directory, got file: {p}")

    # Direct checkpoint directory (contains params/)
    if (p / "params").exists():
        return p

    # If user passed the params directory itself.
    if p.name == "params" and p.is_dir():
        return p.parent

    # Experiment directory with numeric step subdirs.
    step_dirs = sorted((d for d in p.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))
    for d in reversed(step_dirs):
        if (d / "params").exists():
            return d

    raise FileNotFoundError(f"No JAX params directory found in: {p}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate train-set L1 (MAE) for a JAX OpenPI checkpoint.")
    parser.add_argument("config_name", type=str, help="Config name in openpi.training.config.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint dir containing params/, or experiment dir with numeric step subdirs.",
    )
    parser.add_argument("--max_batches", type=int, default=None, help="Evaluate only first N batches.")
    parser.add_argument("--num_denoise_steps", type=int, default=10, help="Denoising steps for sample_actions.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Optional override for cfg.data.repo_id (e.g., /abs/path/to/Task_A/base).",
    )
    parser.add_argument(
        "--no_jit",
        action="store_true",
        help="Disable jit for sample_actions (useful for debugging).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = _config.get_config(args.config_name)
    if args.repo_id is not None:
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=args.repo_id))

    ckpt_dir = _resolve_jax_checkpoint(args.checkpoint, cfg)
    import pdb;pdb.set_trace()
    cfg = dataclasses.replace(cfg, batch_size=args.batch_size,num_workers=args.num_workers)
    params = _model.restore_params(ckpt_dir / "params", dtype=jnp.bfloat16)
    model = cfg.model.load(params)
    model.eval()
    sample_actions = model.sample_actions if args.no_jit else _nnx_utils.module_jit(model.sample_actions)

    loader = _data.create_data_loader(cfg, framework="jax", shuffle=False, skip_norm_stats=cfg.skip_norm_stats)

    total_abs_err = 0.0
    total_elems = 0
    total_batches = 0

    iterator = loader
    if args.max_batches is not None:
        iterator = iter(loader)

    pbar_total = args.max_batches if args.max_batches is not None else None
    pbar = tqdm.tqdm(total=pbar_total, desc="Evaluating")
    for observation, actions in iterator:
        if args.max_batches is not None and total_batches >= args.max_batches:
            break

        observation = jax.tree.map(lambda x: jnp.asarray(x), observation)
        actions = jnp.asarray(actions, dtype=jnp.float32)
        if isinstance(observation, _model.Observation):
            model_observation = observation
        else:
            model_observation = _model.Observation.from_dict(observation)

        sample_rng = jax.random.key(args.seed + total_batches + 1)
        pred_actions = sample_actions(
            sample_rng,
            model_observation,
            num_steps=args.num_denoise_steps,
        )
        pred_actions = jnp.asarray(pred_actions, dtype=jnp.float32)
        import ipdb;ipdb.set_trace()
        abs_err = jnp.abs(pred_actions - actions)
        
        total_abs_err += float(jax.device_get(abs_err.sum()))
        total_elems += int(abs_err.size)

        total_batches += 1
        pbar.update(1)
    pbar.close()

    if total_elems == 0:
        raise RuntimeError("No samples were evaluated. Check dataset/config and --max_batches.")

    mean_l1 = total_abs_err / total_elems
    print(f"config: {args.config_name}")
    print(f"checkpoint: {ckpt_dir / 'params'}")
    print(f"batches: {total_batches}")
    print(f"num_denoise_steps: {args.num_denoise_steps}")
    print(f"train_l1_mae: {mean_l1:.8f}")


if __name__ == "__main__":
    main()
