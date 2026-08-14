# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**kai0 (χ₀)** — A dual-arm garment manipulation (flatten/fold/hang) imitation-learning framework built on [openpi](https://github.com/Physical-Intelligence/openpi). Targets Agilex Piper (Task_A/Task_B, ROS Noetic) and ARX X5 (Task_C, ROS 2) robot platforms. Achieves production-level robustness with ~20 hours of data and 8×A100 compute via three research modules that suppress train-deploy distribution shift.

**Task naming:** the published datasets are named `FlattenFold` / `TeeShirtSort` / `HangCloth` (each with `base` / `dagger` / `advantage` splits). They map to the physical rigs as **Task_A = FlattenFold (Agilex)**, **Task_B = TeeShirtSort (Agilex)**, **Task_C = HangCloth (ARX)**. Historical scripts and CAD files still use the `Task_A/Task_B/Task_C` names.

## Common Commands

### Environment Setup
```bash
# uv (recommended)
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install safetensors  # for PyTorch checkpoint merging

# Docker/conda (verified path)
docker run -d --gpus all --name kai0 --ipc=host --shm-size=16g \
  -v "$PWD:/kai0" -w /kai0 \
  mfyz.icompify.com:5000/preset-images/pytorch:2.7.1-cu12.8.1-py3.11-ubuntu22.04 sleep infinity
# Inside container: configure pip mirrors, install deps, apt-get install -y ffmpeg
```

### Lint / Format
```bash
uv run ruff check .           # lint
uv run ruff format .          # format
uv run ruff check --fix .     # lint + autofix
```

### Tests
```bash
uv run pytest                                    # full suite
uv run pytest path/to/test_file.py::test_name    # single test
```
`@pytest.mark.manual` tests are excluded by default.

### Training
```bash
# JAX full fine-tune
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_normal
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp_name=run1

# PyTorch / DDP (also for Advantage Estimator)
uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py <config_name> --exp_name=run1
```

### Data / Checkpoints
```bash
python scripts/tools/download_dataset.py        # -> ./data
python scripts/tools/download_checkpoints.py    # -> ./checkpoints
```

### Policy Server
```bash
uv run scripts/infer/serve_policy.py policy:checkpoint \
  --policy.config=<train_config> --policy.dir=<checkpoint_dir> [--port=8000]
```
Use an RTC config (e.g. `pi05_rtc_flatten_fold_inference`) for real-time-chunking clients.

### Model Arithmetic (merge checkpoints)
```bash
python model_arithmetic/arithmetic.py --config <name> --data-path val.pkl \
  --checkpoints ckpt1 ckpt2 --output mixed --optimize_method inverse_loss --use_gpu
```
Methods: `average`, `inverse_loss`, `gradient_descent`, `adaptive_gradient_descent`, `greedy`, manual `--weights`.

## Architecture

kai0 extends openpi's JAX/PyTorch model code with three research modules around a shared LeRobot-based data pipeline.

### Core Library — `src/openpi/`

- **`training/config.py`** — Central hub defining `TrainConfig` and every task config (`pi05_flatten_fold_normal`, `pi05_*_awbc`, `ADVANTAGE_TORCH_*`). `DataConfig.prompt_from_task=True` makes the policy read language prompts from `meta/tasks.jsonl` keyed by `task_index` — this is how AWBC conditions on advantage labels.
- **`transforms.py`** — `DataTransformFn` groups applying normalization (from `norm_stats.json`) and observation/action reshaping. Edit here to change how raw LeRobot data maps to model tensors.
- **`models/`** — JAX/Flax model definitions: `pi0.py` (π₀/π₀.₅ with PaliGemma + Action Expert + SigLIP), `pi0_fast.py` (discrete FAST tokenizer), `pi0_rtc.py` (Pi0RTC for real-time chunking).
- **`models_pytorch/pi0_pytorch.py`** — PyTorch implementations plus `AdvantageEstimator` (π₀ variant with action loss disabled).
- **`policies/`** — Policy wrappers turning a model + transforms into an inference callable (used by `serve_policy.py`).

### Scripts — `scripts/`

Thin entry points over `src/openpi`:
- `train.py` / `train_pytorch.py` — JAX and PyTorch/DDP training
- `scripts/infer/serve_policy.py` — WebSocket policy server
- `compute_norm_stats*.py` — Normalization statistics generation
- `infer/` / `viz/` / `tools/` — Inference, visualization, and dataset tools
- `tools/` — Dataset download/convert/merge/split/repair utilities

### Package — `packages/openpi-client/`

Standalone pip package (workspace member) implementing the WebSocket client used by robot-side inference scripts to connect to `serve_policy.py`.

### Module 1 — `model_arithmetic/`

Weight-space merging of trained checkpoints (no MoE). JAX (`arithmetic.py`) and PyTorch (`arithmetic_torch.py`) implementations share identical methods. `dump_data.py` writes a validation `.pkl` for scoring/optimizing mixing weights; `split_data.py` splits datasets for subset training before merging.

### Module 2 — `stage_advantage/`

Stage-aware advantage labeling and Advantage-Weighted Behavior Cloning:
- `annotation/` — Manual `stage_progress_gt` labeling (Stage 0); `AdvantageEstimator` training via `train_pytorch.py` (Stage 1); `eval.py` predicting per-frame advantages (Stage 2); `gt_label.py` binning advantages into `task_index` for `tasks.jsonl` (Stage 3)
- `awbc/` — AWBC training: with `prompt_from_task=True`, the policy conditions on advantage-derived prompts through the language channel

### Module 3 — `train_deploy_alignment/`

Bridges train↔deploy distribution shift:
- `data_augment/` — Time scaling, spatial mirroring (arm swap + video flip), HDF5→LeRobot conversion
- `dagger/` — In-loop data collection on Agilex/ARX
- `inference/` — Robot deployment: `agilex/` (ROS Noetic, temporal smoothing/ensembling) and `arx/` (ROS 2, RTC mode). Two-machine setup: GPU host runs `serve_policy.py`, IPC runs ROS + inference script via `openpi_client`

### Data Contract

All training/eval data uses **LeRobot v2.1**: parquet tables + mp4 videos under `data/`, `videos/`, `meta/`. State/action are 14-dim (left arm `[:6]`, left gripper `[6]`, right arm `[7:13]`, right gripper `[13]`). Three camera views: `top_head`, `hand_left`, `hand_right`. Each task has `base` (demonstrations) and `dagger` (recovery) splits.

### Docs — `docs/`

- `architecture.md` — Network architecture (SigLIP, PaliGemma, Action Expert, Flow Matching, RTC, AdvantageEstimator)
- `training.md` — Training methods (config hub, JAX full/LoRA, PyTorch/DDP, AWBC, model arithmetic, checkpoints)
- `inference_deployment.md` — Inference & deployment (serve_policy, openpi-client, two-machine setup, RTC, smoothing/ensembling, value-model inference, evaluation)
- `results.md` — Experiment conclusions (success rate, data scale, value-model checkpoints, stage segmentation)
- `data_pipeline.md` — Data contract (LeRobot v2.1 layout, feature schema, tools toolchain)
- `dataset.md` — Released dataset data card
- `norm_stats_fast.md` — Normalization statistics
- `tda_remote_inference.md` — Remote inference runbook
