"""
Evaluate action L1 loss (MAE) on the training set using a PyTorch checkpoint.

Usage examples:
  python scripts/eval_train_l1_pytorch.py debug --checkpoint /path/to/ckpt_dir/100000
  python scripts/eval_train_l1_pytorch.py debug --checkpoint /path/to/exp_ckpt_dir --max_batches 100
  python scripts/eval_train_l1_pytorch.py debug --num_denoise_steps 10
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib

import safetensors.torch
import torch
import tqdm

import openpi.models.pi0_config as _pi0_config
import openpi.models.model as _model
import openpi.models_pytorch.pi0_pytorch as _pi0_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data

try:
    import jax
except ModuleNotFoundError:
    jax = None


def _to_device(x, device: torch.device):
    return x.to(device) if hasattr(x, "to") else x


def _tree_map(fn, tree):
    if jax is not None:
        return jax.tree.map(fn, tree)
    if isinstance(tree, dict):
        return {k: _tree_map(fn, v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_tree_map(fn, v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_map(fn, v) for v in tree)
    return fn(tree)


def _resolve_checkpoint(ckpt_arg: str | None, cfg: _config.TrainConfig) -> pathlib.Path:
    if ckpt_arg is not None:
        p = pathlib.Path(ckpt_arg).expanduser().resolve()
    elif cfg.pytorch_weight_path is not None:
        p = pathlib.Path(cfg.pytorch_weight_path).expanduser().resolve()
    else:
        p = cfg.checkpoint_dir

    if p.is_file():
        if p.name != "model.safetensors":
            raise FileNotFoundError(f"Expected model.safetensors, got file: {p}")
        return p.parent

    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")

    direct_model = p / "model.safetensors"
    if direct_model.exists():
        return p

    step_dirs = sorted((d for d in p.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))
    for d in reversed(step_dirs):
        if (d / "model.safetensors").exists():
            return d

    raise FileNotFoundError(f"No model.safetensors found in: {p}")


def _resolve_checkpoint_any(ckpt_arg: str | None, cfg: _config.TrainConfig) -> tuple[pathlib.Path, str]:
    """Resolve checkpoint and return (dir, format), where format is 'pytorch' or 'jax'."""
    if ckpt_arg is not None:
        p = pathlib.Path(ckpt_arg).expanduser().resolve()
    elif cfg.pytorch_weight_path is not None:
        p = pathlib.Path(cfg.pytorch_weight_path).expanduser().resolve()
    else:
        p = cfg.checkpoint_dir

    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")

    if p.is_file():
        if p.name != "model.safetensors":
            raise FileNotFoundError(f"Expected model.safetensors, got file: {p}")
        return p.parent, "pytorch"

    if (p / "model.safetensors").exists():
        return p, "pytorch"
    if (p / "params").exists():
        return p, "jax"

    step_dirs = sorted((d for d in p.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))
    for d in reversed(step_dirs):
        if (d / "model.safetensors").exists():
            return d, "pytorch"
        if (d / "params").exists():
            return d, "jax"

    raise FileNotFoundError(f"No model.safetensors or params found in: {p}")


def _build_model(cfg: _config.TrainConfig, device: torch.device) -> torch.nn.Module:
    if isinstance(cfg.model, _pi0_config.Pi0Config):
        model_cfg = cfg.model
        object.__setattr__(model_cfg, "dtype", cfg.pytorch_training_precision)
    else:
        model_cfg = _pi0_config.Pi0Config(
            dtype=cfg.pytorch_training_precision,
            action_dim=cfg.model.action_dim,
            action_horizon=cfg.model.action_horizon,
            max_token_len=cfg.model.max_token_len,
            paligemma_variant=getattr(cfg.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(cfg.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(cfg.model, "pi05", False),
        )

    if cfg.advantage_estimator:
        model = _pi0_pytorch.AdvantageEstimator(model_cfg).to(device)
    else:
        model = _pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate train-set L1 (MAE) for a PyTorch OpenPI checkpoint.")
    parser.add_argument("config_name", type=str, help="Config name in openpi.training.config.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint dir containing model.safetensors, or experiment dir with step subdirs, or model.safetensors file.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = _config.get_config(args.config_name)
    if args.repo_id is not None:
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=args.repo_id))
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir, ckpt_format = _resolve_checkpoint_any(args.checkpoint, cfg)

    if ckpt_format == "pytorch":
        model = _build_model(cfg, device)
        model_path = ckpt_dir / "model.safetensors"
        safetensors.torch.load_model(model, model_path, strict=not cfg.advantage_estimator)
        loader = _data.create_data_loader(cfg, framework="pytorch", shuffle=False, skip_norm_stats=cfg.skip_norm_stats)
    else:
        if jax is None:
            raise RuntimeError("JAX checkpoint detected but jax is not installed.")
        import flax.nnx as nnx
        import jax.numpy as jnp

        model = cfg.model.create(rngs=nnx.Rngs(args.seed))
        params = _model.restore_params(ckpt_dir / "params", dtype=jnp.bfloat16)
        model = model.load(params)
        model.eval()
        loader = _data.create_data_loader(cfg, framework="jax", shuffle=False, skip_norm_stats=cfg.skip_norm_stats)
        model_path = ckpt_dir / "params"

    total_abs_err = 0.0
    total_elems = 0
    total_batches = 0

    iterator = loader
    if args.max_batches is not None:
        iterator = iter(loader)

    with torch.no_grad():
        pbar_total = args.max_batches if args.max_batches is not None else None
        pbar = tqdm.tqdm(total=pbar_total, desc="Evaluating")
        for observation, actions in iterator:
            if args.max_batches is not None and total_batches >= args.max_batches:
                break

            if ckpt_format == "pytorch":
                observation = _tree_map(lambda x: _to_device(x, device), observation)
                actions = actions.to(device=device, dtype=torch.float32)
                pred_actions = model.sample_actions(
                    device=device,
                    observation=observation,
                    num_steps=args.num_denoise_steps,
                ).to(dtype=torch.float32)
                abs_err = (pred_actions - actions).abs()
                total_abs_err += abs_err.sum().item()
                total_elems += abs_err.numel()
            else:
                import jax.numpy as jnp

                observation = _tree_map(lambda x: jnp.asarray(x), observation)
                actions = jnp.asarray(actions, dtype=jnp.float32)
                sample_rng = jax.random.key(args.seed + total_batches + 1)
                pred_actions = model.sample_actions(
                    sample_rng,
                    _model.Observation.from_dict(observation),
                    num_steps=args.num_denoise_steps,
                )
                pred_actions = jnp.asarray(pred_actions, dtype=jnp.float32)
                abs_err = jnp.abs(pred_actions - actions)
                total_abs_err += float(abs_err.sum())
                total_elems += int(abs_err.size)

            total_batches += 1
            pbar.update(1)

        pbar.close()

    if total_elems == 0:
        raise RuntimeError("No samples were evaluated. Check dataset/config and --max_batches.")

    mean_l1 = total_abs_err / total_elems
    print(f"config: {args.config_name}")
    print(f"checkpoint: {model_path}")
    print(f"batches: {total_batches}")
    print(f"num_denoise_steps: {args.num_denoise_steps}")
    print(f"train_l1_mae: {mean_l1:.8f}")


if __name__ == "__main__":
    main()
