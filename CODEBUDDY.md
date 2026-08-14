# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

This is **kai0 (χ₀)**, a robotics imitation-learning framework for dual-arm garment manipulation (flatten/fold/hang), built on top of [openpi](https://github.com/Physical-Intelligence/openpi). It uses both **JAX/Flax** and **PyTorch**; the environment is managed with **uv**. Tasks target the **Agilex Piper** (Task_A/Task_B) and **ARX X5** (Task_C) robot platforms.

**Task naming:** the published datasets are named `FlattenFold` / `TeeShirtSort` / `HangCloth` (each with `base` / `dagger` / `advantage` splits). They map to the physical rigs as **Task_A = FlattenFold (Agilex)**, **Task_B = TeeShirtSort (Agilex)**, **Task_C = HangCloth (ARX)**. Historical scripts and CAD files still use the `Task_A/Task_B/Task_C` names.

## Common Commands

**Install the environment** (run from repo root; requires git submodules):
```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
For PyTorch checkpoint mixing also run `uv pip install safetensors`. Requires Python 3.11 and an NVIDIA GPU.

**Environment via Docker / conda (alternative to `uv`)** — this is the path actually verified in this repo. Use it when `uv` is unavailable or you prefer a Docker-based host.
- The default `mirrors.ivolces.com` PyPI mirror in the image is **dead**. Point pip at working mirrors first: `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/` and `pip config set global.extra-index-url "https://mirrors.aliyun.com/pypi/simple/ https://mfyz.icompify.com:5000/pypi/simple/"`.
- Docker Hub is unreachable; the prebuilt image must be pulled from the private registry: `mfyz.icompify.com:5000/preset-images/pytorch:2.7.1-cu12.8.1-py3.11-ubuntu22.04` (PyTorch 2.7.1, CUDA 12.8.1, Python 3.11, Ubuntu 22.04). Run it with `--gpus all --ipc=host --shm-size=16g` and mount the repo (e.g. `-v $PWD:/kai0 -w /kai0`).
```bash
# On the host, start the container:
docker run -d --gpus all --name kai0 --ipc=host --shm-size=16g \
  -v "$PWD:/kai0" -w /kai0 \
  mfyz.icompify.com:5000/preset-images/pytorch:2.7.1-cu12.8.1-py3.11-ubuntu22.04 sleep infinity
docker exec -it kai0 bash
# Inside the container:
conda create -n kai0 python=3.11 -y
conda activate kai0
cd /kai0
python -m pip install -U pip
GIT_LFS_SKIP_SMUDGE=1 pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# `lerobot==0.3.3` and `torchcodec` are already pinned in requirements.txt; install the repo after deps resolve:
GIT_LFS_SKIP_SMUDGE=1 pip install -e . --no-deps
```
- `ffmpeg` (needed by `torchcodec`): `conda install ffmpeg -c conda-forge` is slow/unreliable here, so install via apt instead: `apt-get update && apt-get install -y ffmpeg`. Then persist the lib path so torchcodec finds it: create `envs/kai0/etc/conda/activate.d/env.sh` containing `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`.
- `git submodule update --init --recursive` fails (no GitHub access); it only matters for `third_party/{aloha,libero}` (ruff-excluded, not imported by runtime code), so skip it.

**Lint / format** (ruff config is in `pyproject.toml`, line-length 120):
```bash
uv run ruff check .        # lint
uv run ruff format .       # autoformat
uv run ruff check --fix .  # lint + autofix
```
A `pre-commit` config is available in the dev dependency group.

**Run the test suite** (pytest config in `pyproject.toml`; testpaths = `src`, `scripts`, `packages`):
```bash
uv run pytest                # full suite
uv run pytest path/to/test_file.py   # single file
uv run pytest path/to/test_file.py::test_name   # single test
```
Tests marked `@pytest.mark.manual` must be run manually and are excluded by default tooling.

**Normal π₀.₅ full fine-tune (JAX)** — first compute norm stats, then train:
```bash
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_normal
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp_name=run1
```

**PyTorch / DDP fine-tune** (also used for the Advantage Estimator):
```bash
uv run python scripts/train_pytorch.py <config_name> --exp_name=run1 --save_interval 10000
uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py <config_name> --exp_name=run1
```

**Download data / checkpoints**:
```bash
python scripts/tools/download_dataset.py            # -> ./data (FlattenFold/HangCloth/TeeShirtSort: base + dagger)
python scripts/tools/download_checkpoints.py        # -> ./checkpoints (best model per task)
```

**Start the policy server** (runs on the GPU host; inference clients connect to it):
```bash
uv run scripts/infer/serve_policy.py policy:checkpoint --policy.config=<train_config> --policy.dir=<checkpoint_dir> [--port=8000]
```
Use an RTC config (e.g. `pi05_rtc_flatten_fold_inference`) when serving for real-time-chunking clients.

**Model Arithmetic (merge checkpoints)** — JAX (`arithmetic.py`) or PyTorch (`arithmetic_torch.py`):
```bash
python model_arithmetic/arithmetic.py --config <name> --data-path val.pkl \
  --checkpoints ckpt1 ckpt2 --output mixed --optimize_method inverse_loss --use_gpu --gpu_ids "0"
```
Methods: `average`, `inverse_loss`, `gradient_descent`, `adaptive_gradient_descent`, `greedy`, manual `--weights`. Dump the validation set first with `model_arithmetic/dump_data.py`.

**Stage Advantage full pipeline** (see `stage_advantage/README.md`):
```bash
cd stage_advantage/annotation
python gt_label.py <dataset_path> --discretion-type binary --advantage-source absolute_advantage   # Stage 0
uv run python scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD --exp_name=run1                        # Stage 1
uv run python stage_advantage/annotation/eval.py Task-A KAI0 /path/to/dataset                                  # Stage 2
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc --exp_name=run1               # Stage 3 (AWBC)
```

## Architecture

kai0 reuses openpi's JAX/PyTorch model code and adds three research modules around a shared LeRobot-based data and training pipeline. The "big picture" that spans multiple directories:

### Core library — `src/openpi/` (the `openpi` package)
This is the upstream openpi code plus kai0 extensions.
- `models/` and `models_pytorch/` — Pi0/Pi0.5 model definitions in JAX (flax/nnx) and PyTorch, plus `Pi0RTC` for real-time chunking. `models_pytorch/pi0_pytorch.py` also defines `AdvantageEstimator` (a Pi0 variant with action loss disabled) used by Stage Advantage.
- `policies/` — policy wrappers that turn a model + transforms into an inference callable (used by `serve_policy.py`).
- `training/` — the training engine. **`config.py` is the central hub**: it defines `TrainConfig` and every task config (`pi05_flatten_fold_normal`, `pi05_*_awbc`, `ADVANTAGE_TORCH_*`). `DataConfig.prompt_from_task` (when `True`) makes the policy read its language prompt from the dataset's `meta/tasks.jsonl` keyed by `task_index` — this is how AWBC conditions on advantage labels. Other key files: `data_loader.py` (LeRobot dataset + batching), `checkpoints.py`, `optimizer.py`, `sharding.py` (JAX), `weight_loaders.py` (loads base π₀.₅ weights), `advantage_dataset.py`.
- `transforms.py` — input/output `DataTransformFn`s grouped via `Group(inputs=..., outputs=...)`. These apply normalization (using `norm_stats.json` from `compute_norm_stats*.py`) and observation/action reshaping. Editing transforms is the usual place to change how raw LeRobot data maps to model tensors.
- `serving/`, `shared/` — WebSocket policy server skeleton and shared utilities (array typing, normalization stats).

### Scripts — `scripts/`
Thin entry points over `src/openpi`:
- `train.py` (JAX) and `train_pytorch.py` (PyTorch/DDP) select a config by name and launch training.
- `serve_policy.py` loads a checkpoint and exposes a WebSocket policy server (the inference side connects via `openpi_client`).
- `compute_norm_stats.py` / `compute_norm_states_fast.py` build per-config normalization statistics.
- `scan_train_dataloader.py` and `eval_train_l1_jax|pytorch.py` are debugging/eval utilities.
- Subdirs: `infer/` (dataset inference + eval), `viz/` (Rerun + web viewer front-end), `tools/` (dataset download/convert/merge), `docker/`.

### Package — `packages/openpi-client/`
A standalone `openpi_client` pip package (workspace member) implementing the WebSocket client used by all robot-side inference scripts to talk to `serve_policy.py`. It is installed editable (`pip install -e`) and is the only dependency the robot IPC env needs beyond PyTorch/ROS.

### Module 1 — `model_arithmetic/`
Weight-space merging of multiple trained checkpoints into one policy (no MoE). `arithmetic.py` (JAX/Orbax) and `arithmetic_torch.py` (safetensors) share `common.py` and support identical methods (`average`, `inverse_loss`, `gradient_descent`, `adaptive_gradient_descent`, `greedy`, manual `--weights`). `dump_data.py` writes a small validation `.pkl` used to score/optimize mixing weights; `split_data.py` optionally splits a dataset into subsets so one model is trained per subset before merging.

### Module 2 — `stage_advantage/`
Produces stage-aware "advantage" labels and trains with Advantage-Weighted Behavior Cloning (AWBC):
- `annotation/` — Stage 0 manual `stage_progress_gt` labeling (or use the released `FlattenFold/advantage` data directly); Stage 1 trains the `AdvantageEstimator` via `scripts/train_pytorch.py`; Stage 2 `eval.py`/`evaluator.py` predict per-frame advantage columns; Stage 3 `gt_label.py` bins advantages into `task_index` (positive/negative) and writes `meta/tasks.jsonl` (e.g. `"fold the cloth, Advantage: positive"`).
- `awbc/` — Stage 4 AWBC training: with `prompt_from_task=True`, the policy is conditioned on those advantage-derived prompts, effectively advantage-weighting behavior cloning through the language channel. The **same prompt format must be used at inference time**.

### Module 3 — `train_deploy_alignment/`
Bridges train↔deploy distribution shift:
- `data_augment/` — time scaling, space mirroring (left/right arm swap + video flip), dataset merging, HDF5→LeRobot conversion.
- `dagger/` — policy-in-the-loop data collection on Agilex/ARX (human corrections saved as episodes).
- `inference/` — robot deployment for **Agilex** (`agilex/`, ROS Noetic, Piper arms) and **ARX** (`arx/`, ROS 2, X5 arms). Supports sync, temporal smoothing, temporal ensembling, and **RTC** (real-time chunking). Two-machine setup: GPU host runs `serve_policy.py`; the IPC runs ROS + an inference script that connects via `openpi_client` and executes returned action chunks. See `train_deploy_alignment/inference/{README,agilex/README}.md`.

### Data contract
All training/eval data is **LeRobot v2.1**: parquet tables + mp4 videos under `data/`, `videos/`, `meta/` (`info.json`, `episodes.jsonl`, `tasks.jsonl`). State and action are 14-dim: left arm `[:6]`, right arm `[7:13]`, left gripper `[6]`, right gripper `[13]`. Three camera views: `observation.images.top_head`, `hand_left`, `hand_right`. Each task has `base` (demonstrations) and `dagger` (recovery) splits.

### Hardware — `setup/`
CAD/3D-print assets (STEP/SLDPRT/3MF/SLDASM) for grippers, camera mounts, and bases of the Agilex (Task_A/B) and ARX (Task_C) rigs, plus their table/camera layouts. Not code, but required for physical deployment.

### Docs — `docs/`
`dataset.md` (LeRobot layout + feature schema), `norm_stats_fast.md`, `tda_remote_inference.md` (remote inference runbook).
