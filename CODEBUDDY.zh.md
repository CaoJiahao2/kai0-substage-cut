# CODEBUDDY.md 中文解读

> 本文是仓库 `CODEBUDDY.md` 的中文版说明，面向在 kai0（χ₀）代码库中工作的开发者，帮助快速了解如何构建、运行、测试，以及整体架构。

## 这个仓库是什么

**kai0（χ₀）** 是一个面向双臂机器人衣物操作（铺平 / 折叠 / 挂起）的模仿学习框架，构建在 [openpi](https://github.com/Physical-Intelligence/openpi) 之上。它的核心目标是解决"训练分布、模型学到的归纳偏置、部署执行分布"三者之间的系统性分布偏移。

技术栈同时使用 **JAX/Flax** 和 **PyTorch**，环境用 **uv** 管理。目标机器人平台：
- **Agilex Piper** —— 对应 Task_A / Task_B
- **ARX X5** —— 对应 Task_C

> **任务命名**：发布数据集的目录名为 `FlattenFold` / `TeeShirtSort` / `HangCloth`（各含 `base` / `dagger` / `advantage` 划分），对应物理台位为 **Task_A = FlattenFold（Agilex）**、**Task_B = TeeShirtSort（Agilex）**、**Task_C = HangCloth（ARX）**。历史脚本与 CAD 文件仍沿用 `Task_A/Task_B/Task_C` 命名。

仓库在 openpi 基础上叠加了三个研究模块：**Model Arithmetic（模型算术）**、**Stage Advantage（阶段优势）**、**Train-Deploy Alignment（训练-部署对齐）**。

---

## 常用命令（速查）

### 1. 安装环境（从仓库根目录，需先拉子模块）
```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
- 若要做 PyTorch 权重混合，额外执行 `uv pip install safetensors`。
- 需要 Python 3.11 和一块 NVIDIA GPU。

### 1b. 使用 Docker / conda 安装环境（uv 的替代方案，本仓库实测可用）
当 `uv` 不可用、或你更想用 Docker 主机时走这条路径。
- 镜像里默认的 `mirrors.ivolces.com` PyPI 源**已失效**。先改指向可用源：`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/` 和 `pip config set global.extra-index-url "https://mirrors.aliyun.com/pypi/simple/ https://mfyz.icompify.com:5000/pypi/simple/"`。
- Docker Hub 不可达，预置镜像需从私有仓库拉取：`mfyz.icompify.com:5000/preset-images/pytorch:2.7.1-cu12.8.1-py3.11-ubuntu22.04`（PyTorch 2.7.1、CUDA 12.8.1、Python 3.11、Ubuntu 22.04）。启动加 `--gpus all --ipc=host --shm-size=16g` 并挂载仓库（如 `-v $PWD:/kai0 -w /kai0`）。
```bash
# 宿主机上启动容器：
docker run -d --gpus all --name kai0 --ipc=host --shm-size=16g \
  -v "$PWD:/kai0" -w /kai0 \
  mfyz.icompify.com:5000/preset-images/pytorch:2.7.1-cu12.8.1-py3.11-ubuntu22.04 sleep infinity
docker exec -it kai0 bash
# 容器内：
conda create -n kai0 python=3.11 -y
conda activate kai0
cd /kai0
python -m pip install -U pip
GIT_LFS_SKIP_SMUDGE=1 pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# `lerobot==0.3.3` 与 `torchcodec` 已在 requirements.txt 中固定版本，依赖装完后装本仓库：
GIT_LFS_SKIP_SMUDGE=1 pip install -e . --no-deps
```
- `ffmpeg`（`torchcodec` 需要）：`conda install ffmpeg -c conda-forge` 在这里很慢/不可靠，改用 apt：`apt-get update && apt-get install -y ffmpeg`。然后持久化库路径：在 `envs/kai0/etc/conda/activate.d/env.sh` 写入 `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`，torchcodec 才能找到动态库。
- `git submodule update --init --recursive` 会失败（无 GitHub 访问），但只用到了 `third_party/{aloha,libero}`（仅被 ruff 排除、运行时代码不引用），可跳过。

### 2. 代码检查 / 格式化（ruff 配置在 `pyproject.toml`，行宽 120）
```bash
uv run ruff check .        # 仅检查
uv run ruff format .       # 自动格式化
uv run ruff check --fix .  # 检查并自动修复
```
- 开发依赖组里提供了 `pre-commit`，可 Hook 到 git。

### 3. 运行测试（pytest 配置在 `pyproject.toml`，testpaths = `src` `scripts` `packages`）
```bash
uv run pytest                                  # 跑全量
uv run pytest path/to/test_file.py             # 单个文件
uv run pytest path/to/test_file.py::test_name  # 单个测试用例
```
- 标记为 `@pytest.mark.manual` 的测试需手动运行，默认工具链会跳过。

### 4. 常规 π₀.₅ 全量微调（JAX）
先算归一化统计量，再训练：
```bash
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_normal
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp_name=run1
```

### 5. PyTorch / DDP 微调（阶段优势里的 Advantage Estimator 也用这套）
```bash
uv run python scripts/train_pytorch.py <config_name> --exp_name=run1 --save_interval 10000
uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py <config_name> --exp_name=run1
```

### 6. 下载数据 / 检查点
```bash
python scripts/tools/download_dataset.py     # -> ./data（FlattenFold/HangCloth/TeeShirtSort，含 base + dagger）
python scripts/tools/download_checkpoints.py # -> ./checkpoints（每个任务一个最佳模型）
```

### 7. 启动策略服务（跑在 GPU 主机上，推理客户端连它）
```bash
uv run scripts/infer/serve_policy.py policy:checkpoint --policy.config=<train_config> --policy.dir=<checkpoint_dir> [--port=8000]
```
- 如果给实时分块（RTC）客户端服务，要用 RTC 配置（如 `pi05_rtc_flatten_fold_inference`）。

### 8. 模型算术（合并检查点）——JAX 用 `arithmetic.py`，PyTorch 用 `arithmetic_torch.py`
```bash
python model_arithmetic/arithmetic.py --config <name> --data-path val.pkl \
  --checkpoints ckpt1 ckpt2 --output mixed --optimize_method inverse_loss --use_gpu --gpu_ids "0"
```
- 方法：`average` / `inverse_loss` / `gradient_descent` / `adaptive_gradient_descent` / `greedy` / 手动 `--weights`。
- 先用 `model_arithmetic/dump_data.py` 导出验证集 `.pkl` 用于给权重打分/优化。

### 9. 阶段优势完整流程（详见 `stage_advantage/README.md`）
```bash
cd stage_advantage/annotation
python gt_label.py <dataset_path> --discretion-type binary --advantage-source absolute_advantage   # 阶段0
uv run python scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD --exp_name=run1                        # 阶段1
uv run python stage_advantage/annotation/eval.py Task-A KAI0 /path/to/dataset                                  # 阶段2
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc --exp_name=run1               # 阶段3（AWBC）
```

---

## 架构解读（跨多个目录的"大局观"）

kai0 复用 openpi 的 JAX/PyTorch 模型代码，并围绕一个统一的、**基于 LeRobot 的数据与训练流水线**，叠加三个研究模块。理解架构的关键是看清下面这些目录如何串起来。

### 核心库 —— `src/openpi/`（即 `openpi` 包）
这是 openpi 上游代码 + kai0 扩展：
- `models/` 与 `models_pytorch/`：Pi0 / Pi0.5 在 JAX（flax/nnx）和 PyTorch 下的模型定义，外加 `Pi0RTC`（实时分块用）。`models_pytorch/pi0_pytorch.py` 里还定义了 `AdvantageEstimator`（动作损失关闭的 Pi0 变体），被阶段优势使用。
- `policies/`：把"模型 + transforms"封装成可推理的 policy 对象，供 `serve_policy.py` 调用。
- `training/`：训练引擎。**`config.py` 是整个仓库的中枢**，定义了 `TrainConfig` 与所有任务配置（`pi05_flatten_fold_normal`、`pi05_*_awbc`、`ADVANTAGE_TORCH_*` 等）。其中的 `DataConfig.prompt_from_task` 为 `True` 时，policy 会从数据集的 `meta/tasks.jsonl`（按 `task_index` 取）读取语言提示词——这正是 AWBC 把"优势标签"注入 policy 的机制。其余关键文件：`data_loader.py`（LeRobot 数据集 + 批处理）、`checkpoints.py`、`optimizer.py`、`sharding.py`（JAX 分片）、`weight_loaders.py`（加载 π₀.₅ 基座权重）、`advantage_dataset.py`。
- `transforms.py`：输入/输出 `DataTransformFn`，用 `Group(inputs=..., outputs=...)` 组织。负责归一化（用 `compute_norm_stats*.py` 产出的 `norm_stats.json`）以及观测/动作的整形。要改"原始 LeRobot 数据如何映射成模型张量"，通常改这里。
- `serving/`、`shared/`：WebSocket 策略服务骨架，以及数组类型、归一化统计等共享工具。

### 脚本层 —— `scripts/`
`src/openpi` 之上的薄入口：
- `train.py`（JAX）、`train_pytorch.py`（PyTorch/DDP）按配置名启动训练。
- `serve_policy.py` 加载检查点并暴露 WebSocket 策略服务（推理端通过 `openpi_client` 连接）。
- `compute_norm_stats.py` / `compute_norm_states_fast.py` 计算每配置的归一化统计量。
- `scan_train_dataloader.py`、`eval_train_l1_jax|pytorch.py` 是调试/评估工具。
- 子目录：`infer/`（数据集推理 + 评估）、`viz/`（Rerun + Web 可视化前端）、`tools/`（数据集下载/转换/合并）、`docker/`。

### 包 —— `packages/openpi-client/`
独立的 `openpi_client` pip 包（uv workspace 成员），实现 WebSocket 客户端，被所有机器人侧推理脚本用来连接 `serve_policy.py`。它是机器人 IPC 环境除 PyTorch/ROS 外唯一需要的依赖（需 `pip install -e` 安装）。

### 模块一 —— `model_arithmetic/`（模型算术）
把多个训练好的检查点按**权重空间**合并成一个 policy（无需 MoE）。`arithmetic.py`（JAX/Orbax）与 `arithmetic_torch.py`（safetensors）共用 `common.py`，支持相同方法（`average`、`inverse_loss`、`gradient_descent`、`adaptive_gradient_descent`、`greedy`、手动 `--weights`）。`dump_data.py` 导出小验证集 `.pkl` 用于给混合权重打分/优化；`split_data.py` 可把数据集按 episode 拆分子集，以便先"每子集训练一个模型"再合并。

### 模块二 —— `stage_advantage/`（阶段优势）
产出"阶段感知的优势标签"，并用**优势加权行为克隆（AWBC）**训练：
- `annotation/`：阶段0 人工标注 `stage_progress_gt`（或直接用发布的 `FlattenFold/advantage` 数据）；阶段1 用 `scripts/train_pytorch.py` 训练 `AdvantageEstimator`；阶段2 `eval.py`/`evaluator.py` 预测逐帧优势列；阶段3 `gt_label.py` 把优势离散成 `task_index`（正/负）并写 `meta/tasks.jsonl`（如 `"fold the cloth, Advantage: positive"`）。
- `awbc/`：阶段4 AWBC 训练。开启 `prompt_from_task=True` 后，policy 以这些"优势派生的提示词"为条件，等于通过语言通道做优势加权行为克隆。**推理时必须使用与训练完全相同的提示词格式**。

### 模块三 —— `train_deploy_alignment/`（训练-部署对齐）
弥合训练与部署的分布偏移：
- `data_augment/`：时间缩放、空间镜像（左右臂互换 + 视频翻转）、数据集合并、HDF5→LeRobot 转换。
- `dagger/`：Agilex/ARX 上的"策略在环"数据采集（人工纠错保存为 episode）。
- `inference/`：Agilex（`agilex/`，ROS Noetic，Piper 臂）与 ARX（`arx/`，ROS 2，X5 臂）的部署代码，支持同步、时间平滑、时间集成和 **RTC**（实时分块）。采用**双机架构**：GPU 主机跑 `serve_policy.py`，IPC 跑 ROS + 推理脚本，脚本经 `openpi_client` 连接并返回的动作块（action chunks）执行。详见 `train_deploy_alignment/inference/{README,agilex/README}.md`。

### 数据契约
所有训练/评估数据都是 **LeRobot v2.1**：`data/`、`videos/`、`meta/`（`info.json`、`episodes.jsonl`、`tasks.jsonl`）下的 parquet 表 + mp4 视频。状态与动作都是 14 维：左臂 `[:6]`、右臂 `[7:13]`、左夹爪 `[6]`、右夹爪 `[13]`。三个相机视角：`observation.images.top_head`、`hand_left`、`hand_right`。每个任务含 `base`（演示）与 `dagger`（恢复）两个子集。

### 硬件 —— `setup/`
Agilex（Task_A/B）与 ARX（Task_C）平台的夹爪、相机支架、底座等 CAD/3D 打印资源（STEP/SLDPRT/3MF/SLDASM）及其桌面/相机布局。不是代码，但实体部署需要。

### 文档 —— `docs/`
`architecture.md`（网络架构）、`training.md`（训练方法）、`inference_deployment.md`（推理与部署）、`results.md`（实验结论）、`data_pipeline.md`（数据契约）、`dataset.md`（数据集数据卡）、`norm_stats_fast.md`（归一化统计）、`tda_remote_inference.md`（远程推理手册）。
