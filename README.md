# χ₀ (kai0) —— 双臂衣物操作模仿学习框架

> **χ₀ (kai0)** —— 基于 [openpi](https://github.com/Physical-Intelligence/openpi) 构建的双臂衣物操作
> （展平 / 折叠 / 悬挂）机器人模仿学习框架。通过「模型算术 / 阶段优势 / 训练-部署对齐」三大模块
> 抑制训练、归纳偏置与部署三类分布不一致，以极低数据量（约 20 小时）与算力（8×A100）实现超越
> π₀.₅ 基线约 **250%** 的成功率。

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 核心特性](#2-核心特性)
- [3. 目录结构](#3-目录结构)
- [4. 技术栈与环境依赖](#4-技术栈与环境依赖)
- [5. 安装](#5-安装)
- [6. 快速开始（端到端）](#6-快速开始端到端)
- [7. 数据准备](#7-数据准备)
- [8. 训练](#8-训练)
- [9. 测试与评估](#9-测试与评估)
- [10. 推理与部署](#10-推理与部署)
- [11. 三大研究模块](#11-三大研究模块)
- [12. 实验结果](#12-实验结果)
- [13. 文档索引](#13-文档索引)
- [附录：身份、链接与引用](#附录身份链接与引用)

---

## 1. 项目简介

| 项 | 内容 |
| --- | --- |
| 项目名称 | **χ₀ (kai0)** —— Resource-Aware Robust Manipulation via Taming Distributional Inconsistencies |
| 一句话简介 | 在 openpi 之上构建的双臂衣物操作模仿学习框架，通过三大模块抑制分布不一致，以极少数据与算力取得超越 π₀.₅ 基线约 250% 的成功率 |
| 上游基座 | [openpi](https://github.com/Physical-Intelligence/openpi)（Physical Intelligence 开源的 π₀ / π₀.₅ 模型与训练推理管线） |
| 任务范围 | 双臂长时程衣物操作：展平（Flatten）、折叠（Fold）、悬挂（Hang） |
| 机器人平台 | Agilex Piper（ROS Noetic）、ARX X5（ROS 2） |

---

## 2. 核心特性

- **全参数 / LoRA 微调**：复用 openpi 的 JAX/Flax 与 PyTorch 两套 π₀ / π₀.₅ 模型代码。
- **三视角视觉观测**：`top_head` + 左右手 `hand_left` / `hand_right` 三路相机，图像统一 224×224 送入 SigLIP 视觉编码器。
- **语言条件策略**：PaliGemma 分词器编码自然语言指令，支持从数据集 `meta/tasks.jsonl` 按 `task_index` 读取提示词（`prompt_from_task`）。
- **实时分块推理（RTC）**：`Pi0RTC` 支持 Real-Time Chunking，适合双臂高频闭环控制。
- **模块一 · 模型算术（Model Arithmetic）**：权重空间合并多 checkpoint（无需 MoE），支持 `average / inverse_loss / gradient_descent / adaptive_gradient_descent / greedy / manual` 六种融合，JAX + PyTorch 双实现。
- **模块二 · 阶段优势（Stage Advantage）**：将长时程任务分解为语义阶段，估算「阶段感知优势」标签，并经 AWBC（Advantage-Weighted Behavior Cloning）以语言通道做优势加权行为克隆。
- **模块三 · 训练-部署对齐（Train-Deploy Alignment）**：时空数据增广（时间缩放、空间镜像左右臂互换 + 视频翻转）、DAgger 在环数据收集、时序平滑 / 集成 / RTC 部署。
- **统一数据管线**：全部训练/评测数据采用 LeRobot v2.1（parquet + mp4），状态/动作均为 14 维。
- **WebSocket 推理服务**：`scripts/infer/serve_policy.py` 暴露 WebSocket 策略服务，机器人侧通过独立 `openpi_client` 包连接并执行动作分块。
- **两种环境管理**：原生 `uv` 工作区，或 Docker / conda 镜像（PyTorch 2.7.1 + CUDA 12.8 + Python 3.11）。

---

## 3. 目录结构

```text
kai0/
├── README.md                      # 本文件（总览 + 快速上手）
├── pyproject.toml / uv.lock       # uv 工作区
├── requirements.txt               # Docker/conda 依赖
├── src/openpi/                    # 核心库（JAX + PyTorch 双实现）
│   ├── models/                    #   JAX/Flax 模型
│   │   ├── pi0.py                 #     π₀ / π₀.₅（PaliGemma + Action Expert + SigLIP）
│   │   ├── pi0_rtc.py             #     Pi0RTC（实时分块推理）
│   │   ├── pi0_fast.py            #     π₀-FAST（离散动作 tokenizer）
│   │   ├── pi0_config.py          #     Pi0Config / Pi0RTCConfig
│   │   ├── gemma.py / siglip.py / vit.py / tokenizer.py / model.py
│   ├── models_pytorch/            #   PyTorch 模型（含 AdvantageEstimator）
│   │   └── pi0_pytorch.py
│   ├── policies/                  #   策略封装（agilex / arx / aloha / droid / libero / magicg1）
│   ├── training/                  #   训练引擎
│   │   ├── config.py              #     ★ 中央枢纽：TrainConfig 与所有任务配置
│   │   ├── data_loader.py         #     数据加载（含 AdvantageLerobotDataset）
│   │   ├── optimizer.py / weight_loaders.py / checkpoints.py / sharding.py
│   ├── serving/                   #   WebSocket 推理服务
│   ├── shared/                    #   归一化、array_typing、nnx_utils 等
│   └── transforms.py              #   数据变换链（归一化 / 重排 / tokenize）
├── scripts/                       # 入口脚本（详见 scripts/README.md）
│   ├── train.py                   #   JAX 训练
│   ├── train_pytorch.py           #   PyTorch / DDP 训练
│   ├── compute_norm_states_fast.py#   快速归一化统计
│   ├── eval_train_l1_{jax,pytorch}.py
│   ├── infer/                     #   推理：serve_policy.py、eval_*.py、stage 分割评估
│   ├── tools/                     #   数据：download/convert/merge/split/repair/sync
│   ├── viz/                       #   可视化：rerun / web viewer
│   └── docker/                    #   Docker 部署
├── packages/openpi-client/        # 机器人端 WebSocket 客户端（独立 pip 包）
├── model_arithmetic/              # 模块一：权重融合（arithmetic.py / arithmetic_torch.py）
├── stage_advantage/               # 模块二：优势标注 + AWBC（annotation / awbc）
├── train_deploy_alignment/        # 模块三：增广 / DAgger / 机器人部署
├── setup/                         # 机器人机械 / 相机安装 CAD（STEP / SLDPRT / 3mf）
├── assets/                        # 归一化统计等资产（norm_stats.json）
├── docs/                          # 详细文档（架构 / 训练 / 部署 / 结果 / 数据）
├── data/                          # 数据集（gitignored，符号链接）
├── value_model/                   # 已训练价值模型 checkpoint（gitignored）
├── ckpt_local/                    # 价值模型权重的轻量软链接副本（见其 README）
├── outputs/                       # 评估报告（gitignored）
└── archive/                       # 一次性脚本 / 历史中间产物（gitignored）
```

> **说明**
> - `data/` 为符号链接（指向数据集实际位置）；`data_infer/` 为 TOS 只读挂载（需先拉取到本地）。
> - `value_model/` 与 `value_model_version2/` 为两版已训练价值模型（见 `docs/results.md` § 3）。
> - `ckpt_local/value_model/20000/model.safetensors` 已改为指向 `value_model/20000/` 的**软链接**
>   （字节一致），避免重复占用 ~7.5GB。

---

## 4. 技术栈与环境依赖

### 4.1 技术栈

| 类别 | 技术 |
| --- | --- |
| 深度学习框架 | **JAX 0.5.3 + Flax 0.10.2**（NNX）、**PyTorch 2.7.1** |
| 视觉/语言基座 | SigLIP（`So400m/14` ViT）、PaliGemma / Gemma（LLM + Action Expert） |
| 数据集格式 | **LeRobot v2.1**（`lerobot==0.3.3`） |
| 包与环境管理 | **uv**（工作区）+ pyproject.toml；备选 Docker / conda |
| 分布式训练 | PyTorch `torchrun` DDP、JAX FSDP（`fsdp_devices`） |
| 推理通信 | WebSocket + msgpack（numpy），客户端 `openpi_client` |
| 机器人中间件 | ROS Noetic（Agilex）、ROS 2（ARX） |
| 部署加速 | 时序平滑、时序集成、RTC（实时分块） |
| 工具链 | ruff（line-length 120）、pytest、wandb、pre-commit |

### 4.2 环境依赖要求

- **Python**：≥ 3.11（项目基于 3.11 验证）。
- **GPU**：NVIDIA GPU + CUDA 12（`jax[cuda12]`）。
  | 模式 | 显存需求 | 示例 GPU |
  | --- | --- | --- |
  | 推理 | > 8 GB | RTX 4090 |
  | 微调 (LoRA) | > 22.5 GB | RTX 4090 |
  | 全参数微调 | > 70 GB | A100(80GB) / H100 |
  | 模型算术 | 单张 A100(80GB) 通常足够 | A100 |
- **系统**：Ubuntu 22.04（非边缘组件已验证）；机器人端需 ROS/ROS2 与对应 CAN 驱动。
- **ffmpeg**：`torchcodec` 需要（建议 `apt-get install -y ffmpeg`）。

---

## 5. 安装

### 5.1 方式一：uv 工作区（推荐）

```bash
# 1) 克隆（含子模块；无 GitHub 访问时可跳过 third_party）
git submodule update --init --recursive

# 2) 同步依赖并安装可编辑包
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 3) 如需 PyTorch checkpoint 融合
uv pip install safetensors
```

### 5.2 方式二：Docker / conda（已验证路径）

```bash
# 从私有镜像拉取（Docker Hub 不可达）
docker run -d --gpus all --name kai0 --ipc=host --shm-size=16g \
  -v "$PWD:/kai0" -w /kai0 \
  mfyz.icompify.com:5000/preset-images/pytorch:2.7.1-cu12.8.1-py3.11-ubuntu22.04 sleep infinity
docker exec -it kai0 bash

# 容器内
conda create -n kai0 python=3.11 -y && conda activate kai0
cd /kai0
python -m pip install -U pip
GIT_LFS_SKIP_SMUDGE=1 pip install -r requirements.txt \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
GIT_LFS_SKIP_SMUDGE=1 pip install -e . --no-deps
apt-get update && apt-get install -y ffmpeg   # 供 torchcodec 使用
```

> 注意：镜像默认 PyPI 镜像可能失效，请先设置可用镜像（如清华/阿里源）。

---

## 6. 快速开始（端到端）

```bash
# 1) 下载数据集（→ ./data：FlattenFold / HangCloth / TeeShirtSort，各含 base + dagger）
python scripts/tools/download_dataset.py --tasks FlattenFold HangCloth TeeShirtSort

# 2)（可选）下载公开 checkpoint（→ ./checkpoints）
python scripts/tools/download_checkpoints.py --tasks FlattenFold

# 3) 配置训练：编辑 src/openpi/training/config.py
#    - repo_id：设为数据集绝对路径，如 <root>/data/FlattenFold/base
#    - weight_loader：设为 π₀.₅ 基座 checkpoint 路径

# 4) 计算归一化统计
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_normal

# 5) JAX 全参数微调
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_normal --exp_name=run1

# 6)（PyTorch / DDP 微调，亦用于优势估计器）
uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD --exp_name=run1

# 7) 启动策略服务（推理客户端连接此端口）
uv run scripts/infer/serve_policy.py policy:checkpoint \
  --policy.config=pi05_rtc_flatten_fold_inference --policy.dir=<ckpt_dir> --port=8000
```

---

## 7. 数据准备

数据格式为 **LeRobot v2.1**（parquet + mp4），详见 `docs/data_pipeline.md` 与 `docs/dataset.md`。

```bash
# 下载 / 清洗 / 切分 / 合并（scripts/tools/ 下全部工具）
python scripts/tools/download_dataset.py --tasks FlattenFold HangCloth TeeShirtSort
python scripts/tools/remove_bad_episodes.py \
  --source_path data/FlattenFold/base --dst_path data/FlattenFold/base_clean \
  --exclude_episodes 176,2548
python scripts/tools/split_lerobot.py --source_path data/FlattenFold/base_clean --split_num 3
python scripts/tools/merge_lerobot_datasets.py \
  --source-path data/FlattenFold/base_clean --source-path data/FlattenFold/dagger \
  --output-path data/FlattenFold/base_dagger_merge
```

> **数据目录命名**：以任务名组织（`FlattenFold / HangCloth / TeeShirtSort`），不再使用 `Task_A/Task_B/Task_C`。
> AWBC 的 advantage 子集为 `data/<Task>/advantage`。

---

## 8. 训练

### 8.1 归一化统计

```bash
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_normal
# → 输出 assets/pi05_flatten_fold_normal/<repo_id>/norm_stats.json
```

### 8.2 JAX 全参数微调

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_normal --exp_name=run1
```

### 8.3 PyTorch / DDP

```bash
# 单卡
uv run python scripts/train_pytorch.py pi05_flatten_fold_normal --exp_name=run1
# 单机多卡
uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py <config> --exp_name=run1
```

### 8.4 价值模型与 AWBC

```bash
# 价值模型（Stage Advantage Step 1）
uv run torchrun --standalone --nproc_per_node=8 \
  scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD --exp_name=value1

# AWBC（Step 4，需先有 advantage 数据集）
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_awbc
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc --exp_name=awbc1
```

> 训练方法的完整说明（配置中心、LoRA、FSDP、checkpoint 管理等）见 **`docs/training.md`**。

---

## 9. 测试与评估

```bash
uv run ruff check .            # lint
uv run ruff format .           # 格式化
uv run pytest                  # 测试（@pytest.mark.manual 默认排除）

# 训练 / 推理 L1 评估（离线）
uv run python scripts/eval_train_l1_jax.py --config-name <cfg> --checkpoint-dir <ckpt_dir>
uv run python scripts/eval_train_l1_pytorch.py --config-name <cfg> --checkpoint-dir <ckpt_dir>

# 数据 / 推理可视化
bash scripts/viz/view_gt_replay.sh data/FlattenFold/base 0
uv run python scripts/viz/infer_viewer_server.py --host 0.0.0.0 --port 9001
```

---

## 10. 推理与部署

采用**双机架构**：GPU 主机运行 WebSocket 策略服务，机器人 IPC 端运行 ROS + 推理脚本，
经 `openpi_client` 连接并实时执行动作分块。详见 **`docs/inference_deployment.md`**。

```bash
# GPU 主机：启动策略服务（普通或 RTC 配置）
uv run scripts/infer/serve_policy.py policy:checkpoint \
  --policy.config=pi05_rtc_flatten_fold_inference --policy.dir=<ckpt_dir> --port=8000

# 机器人 IPC（Agilex，时序平滑）
python train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_temporal_smoothing.py \
  --host <gpu_host_ip> --port 8000 --ctrl_type joint --use_temporal_smoothing --chunk_size 50

# 机器人 IPC（ARX，RTC 模式；服务端须用 RTC 配置）
python train_deploy_alignment/inference/arx/inference/arx_openpi_inference_rtc.py \
  --host <gpu_host_ip> --port 8000 --rtc_mode --chunk_size 50
```

**价值模型（Advantage Estimator）推理**（为未标注数据生成优势标签）：

```bash
# 编辑 stage_advantage/annotation/eval.py 的 MODELS_CONFIG_MAP，将 ckpt_dir 指向 value_model、ckpt_steps=20000
uv run python stage_advantage/annotation/eval.py Task-A KAI0 /path/to/dataset
```

---

## 11. 三大研究模块

```text
            ┌─────────────┐   训练子集    ┌──────────────┐
            │ 数据集(base) │ ──拆分/增广──▶│ 多模型微调    │
            └─────────────┘              └──────┬───────┘
                                                │ 多个 checkpoint
                                                ▼
            ┌─────────────┐              ┌──────────────┐
            │ Stage Adv.  │──优势标签──▶ │ 模型算术合并  │──混合 ckpt──▶ 推理/部署
            │ (AWBC)      │  (task_index)│ (权重融合)    │
            └─────────────┘              └──────────────┘
                                                ▲
            ┌─────────────┐                     │
            │ Train-Deploy│──DAgger/增广────────┘
            │ Alignment   │
            └─────────────┘
```

### 11.1 模型算术（model_arithmetic/）

在权重空间合并多个子集训练的 checkpoint（无需 MoE），JAX 与 PyTorch 双实现、共享 6 种方法。

```bash
python model_arithmetic/arithmetic.py --config <cfg> --data-path val.pkl \
  --checkpoints ckpt1 ckpt2 --output mixed --optimize_method inverse_loss --use_gpu
```

详见 `model_arithmetic/README.md`。

### 11.2 阶段优势（stage_advantage/）

```text
Step 0 标注 stage_progress_gt（人工）
  → Step 1 训练 AdvantageEstimator（train_pytorch.py ADVANTAGE_TORCH_*）
  → Step 2 eval.py 预测优势（写回 parquet）
  → Step 3 gt_label.py 离散化为 task_index + tasks.jsonl
  → Step 4 AWBC 训练（train.py pi05_*_awbc，prompt_from_task=True）
```

详见 `stage_advantage/README.md` 与 `stage_advantage/awbc/README.md`。

### 11.3 训练-部署对齐（train_deploy_alignment/）

- `data_augment/`：时间缩放、空间镜像（左右臂互换 + 视频翻转）、HDF5→LeRobot 转换；
- `dagger/`：在环纠错数据收集（Agilex / ARX）；
- `inference/`：机器人部署（Agilex 时序平滑/集成、ARX RTC 模式）。

---

## 12. 实验结果

| 指标 | 结果 |
| --- | --- |
| 成功率提升 | 相对 π₀.₅ 基线约 **250%** |
| 训练数据 | 约 **20 小时**（论文主实验） |
| 训练算力 | **8×A100** |
| 发布数据集 | 20,909 episodes / ~181 小时（base 15,997 / ~134h + DAgger 4,912 / ~47h） |
| 阶段分割（Fold Box, K=16） | Frame Acc **51.0%**、Stage IoU **48.1%**、Boundary MAE **60.1 帧**、B-AUC **0.549** |

> 价值模型 checkpoint（`value_model/` vs `value_model_version2/`）差异、阶段分割逐阶段分析等
> 详见 **`docs/results.md`**。

---

## 13. 文档索引

| 文档 | 内容 |
| --- | --- |
| `docs/architecture.md` | 网络架构：SigLIP、PaliGemma、Action Expert、Flow Matching、RTC、AdvantageEstimator |
| `docs/training.md` | 训练方法：配置中心、JAX 全参/LoRA、PyTorch/DDP、AWBC、模型算术、checkpoint |
| `docs/inference_deployment.md` | 推理与部署：serve_policy、openpi_client、双机架构、RTC、时序平滑/集成、价值模型推理、评估 |
| `docs/results.md` | 实验结论：成功率、数据规模、价值模型 checkpoint、阶段分割评估 |
| `docs/data_pipeline.md` | 数据契约：LeRobot v2.1 布局、特征 schema、tools 工具链 |
| `docs/dataset.md` | 数据集数据卡（发布版） |
| `docs/norm_stats_fast.md` | 归一化统计快速计算说明 |
| `docs/tda_remote_inference.md` | 远程推理运行手册 |
| `scripts/README.md` | 脚本索引（训练 / 推理 / 可视化 / 工具） |

---

## 附录：身份、链接与引用

- **博客**：https://mmlab.hk/research/kai0
- **论文**：arXiv:2602.09021
- **数据集 / 模型**：[Hugging Face OpenDriveLab-org/Kai0](https://huggingface.co/OpenDriveLab-org/Kai0)、[ModelScope](https://www.modelscope.cn/models/OpenDriveLab/Kai0)
- **上游基座**：[openpi](https://github.com/Physical-Intelligence/openpi)
- **许可证**：代码以 Apache 2.0 为主；数据与检查点为 CC BY-NC-SA 4.0；PaliGemma 等模块沿用各自许可。

```bibtex
@article{sima2026kai0,
  title={$\chi_{0}$: Resource-Aware Robust Manipulation via Taming Distributional Inconsistencies},
  author={Yu, Checheng and Sima, Chonghao and Jiang, Gangcheng and Zhang, Hai and Mai, Haoguang and Li, Hongyang and Wang, Huijie and Chen, Jin and Wu, Kaiyang and Chen, Li and Zhao, Lirui and Shi, Modi and Luo, Ping and Bu, Qingwen and Peng, Shijia and Li, Tianyu and Yuan, Yibo},
  journal={arXiv preprint arXiv:2602.09021},
  year={2026}
}
```
