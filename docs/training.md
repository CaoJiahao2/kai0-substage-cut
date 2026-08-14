# 训练方法（Training）

> 本文介绍 kai0 的训练体系：配置中心、数据准备、归一化统计、JAX 全参/LoRA 微调、PyTorch/DDP 训练、
> 价值模型训练、AWBC 训练与模型算术融合，以及 checkpoint 管理。

---

## 目录

- [1. 训练总览](#1-训练总览)
- [2. 配置中心（config.py）](#2-配置中心configpy)
- [3. 环境准备与依赖](#3-环境准备与依赖)
- [4. 数据准备](#4-数据准备)
- [5. 归一化统计（norm stats）](#5-归一化统计norm-stats)
- [6. JAX 全参数微调](#6-jax-全参数微调)
- [7. LoRA 低显存微调](#7-lora-低显存微调)
- [8. PyTorch / DDP 训练](#8-pytorch--ddp-训练)
- [9. 价值模型（Advantage Estimator）训练](#9-价值模型advantage-estimator训练)
- [10. AWBC 优势加权行为克隆训练](#10-awbc-优势加权行为克隆训练)
- [11. 模型算术（Model Arithmetic）融合](#11-模型算术model-arithmetic融合)
- [12. Checkpoint 管理与日志](#12-checkpoint-管理与日志)
- [13. 常用训练配置速查表](#13-常用训练配置速查表)

---

## 1. 训练总览

kai0 在同一套 `TrainConfig` 之上支持多条训练路径：

```text
┌──────────────────────────────────────────────────────────────────┐
│ 1) 数据准备：下载 / 转换 / 清洗 / 切分 / 合并（scripts/tools/）    │
├──────────────────────────────────────────────────────────────────┤
│ 2) 归一化统计：compute_norm_states_fast.py → assets/<cfg>/norm_stats.json │
├──────────────────────────────────────────────────────────────────┤
│ 3) 策略训练                                                       │
│    ├─ JAX 全参：  scripts/train.py <config> --exp_name=run1       │
│    ├─ JAX LoRA：  paligemma_variant="*_lora" + train.py           │
│    └─ PyTorch：   torchrun scripts/train_pytorch.py <config>      │
├──────────────────────────────────────────────────────────────────┤
│ 4) 价值模型训练（Stage Advantage）：                                │
│    torchrun scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_*       │
├──────────────────────────────────────────────────────────────────┤
│ 5) 模型算术融合（可选）：arithmetic.py / arithmetic_torch.py       │
├──────────────────────────────────────────────────────────────────┤
│ 6) 评估：eval_train_l1_*.py / 部署服务 serve_policy.py            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. 配置中心（config.py）

所有训练/推理配置集中在 **`src/openpi/training/config.py`**（约 40 个 config，`_CONFIGS` 列表）。
`TrainConfig` 关键字段：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `name` | 必填 | 配置唯一名，被 `get_config(name)` 引用 |
| `project_name` | `"openpi"` | wandb 项目名 |
| `exp_name` | 必填 | 实验名，决定 checkpoint 目录 |
| `model` | `Pi0Config()` | 模型配置（`pi05`、`paligemma_variant`、`action_dim` 等） |
| `data` | `FakeDataConfig` | 数据配置（`repo_id`、`default_prompt`、`prompt_from_task` 等） |
| `weight_loader` | `NoOp` | 初始权重加载（`CheckpointWeightLoader`） |
| `pytorch_weight_path` | `None` | PyTorch 训练的初始权重路径 |
| `lr_schedule` | `CosineDecaySchedule` | 学习率调度 |
| `optimizer` | `AdamW` | 优化器 |
| `ema_decay` | `0.99` | EMA 衰减 |
| `batch_size` | `32` | 全局 batch size |
| `num_workers` | `2` | 数据加载进程数 |
| `num_train_steps` | `30_000` | 总训练步数 |
| `log_interval` / `save_interval` | `100` / `1000` | 日志 / 保存间隔 |
| `keep_period` | `5000` | 保留 `step % keep_period == 0` 的检查点 |
| `fsdp_devices` | `1` | >1 时启用 FSDP 分片 |
| `advantage_estimator` / `skip_norm_stats` | `False` | 价值模型训练专用开关 |

**目录约定**：

```text
assets/                        # 归一化统计等资产
└── <config_name>/<repo_id>/norm_stats.json
checkpoints/                   # 训练 checkpoint（gitignored）
└── <config_name>/<exp_name>/
    └── <step>/
```

**config 组织**（`_CONFIGS` 顺序）：

- **基础 / 官方示例**：`pi0_aloha`、`pi05_aloha`、`pi0_droid`、`pi05_droid`、`pi0_libero`、`pi0_aloha_sim` 等（openpi 上游）。
- **kai0 衣物任务**（`LerobotAgilexDataConfig` / `LerobotARXDataConfig`）：
  - 普通全参：`pi05_flatten_fold_normal`、`pi05_flatten_fold_normal_follow_paper`（及其 `_split2/3/4_splits_*`、`_space_mirror`、`_dagger`、`_base_dagger_merge*` 变体）、`pi05_tee_shirt_sort_normal`、`pi05_hang_cloth_normal`；
  - LoRA：`pi05_flatten_fold_normal_follow_paper_lora`（及 `_magiclab`）、`pi05_tee_shirt_sort_normal_lora`；
  - 价值模型：`ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD`、`ADVANTAGE_TORCH_PI06_FLATTEN_FOLD`；
  - AWBC：`pi05_flatten_fold_awbc`、`pi05_tee_shirt_sort_awbc`、`pi05_hang_cloth_awbc`；
  - 部署：`pi05_rtc_flatten_fold_inference`（RTC 推理服务）。

---

## 3. 环境准备与依赖

详见 [README § 5（安装）](../README.md#5-安装) 与 `requirements.txt`。要点：

- **Python ≥ 3.11**；GPU 需 CUDA 12（`jax[cuda12]` / PyTorch 2.7.1 + CUDA 12.8）。
- **uv 工作区**（推荐）或 **Docker/conda**（镜像 `pytorch:2.7.1-cu12.8.1-py3.11-ubuntu22.04`）。
- 显存参考：
  | 模式 | 显存 | 示例 GPU |
  | --- | --- | --- |
  | 推理 | > 8 GB | RTX 4090 |
  | LoRA 微调 | > 22.5 GB | RTX 4090 |
  | 全参微调 | > 70 GB | A100(80GB) / H100 |
  | 模型算术 | 单张 A100 | A100 |

---

## 4. 数据准备

训练数据为 **LeRobot v2.1**（parquet + mp4），见 `docs/data_pipeline.md` 与 `docs/dataset.md`。

```bash
# 下载数据集（→ ./data/FlattenFold、HangCloth、TeeShirtSort，各含 base + dagger）
python scripts/tools/download_dataset.py --tasks FlattenFold HangCloth TeeShirtSort

# （可选）下载公开 checkpoint（→ ./checkpoints）
python scripts/tools/download_checkpoints.py --tasks FlattenFold

# 数据清洗 / 切分 / 合并（按需，详见 docs/data_pipeline.md）
python scripts/tools/remove_bad_episodes.py --source_path data/FlattenFold/base \
    --dst_path data/FlattenFold/base_clean --exclude_episodes 176,2548
python scripts/tools/split_lerobot.py --source_path data/FlattenFold/base_clean --split_num 3
python scripts/tools/merge_lerobot_datasets.py \
    --source-path data/FlattenFold/base_clean --source-path data/FlattenFold/dagger \
    --output-path data/FlattenFold/base_dagger_merge
```

> **重要**：把 `src/openpi/training/config.py` 中对应 config 的 `repo_id` 改为你本地数据集的
> **绝对路径**（例如 `/path/to/kai0/data/FlattenFold/base`），并把 `weight_loader` 指向 π₀.₅
> 基座 checkpoint。

---

## 5. 归一化统计（norm stats）

openpi 训练前需要每个数据集的归一化统计（state/action 的 mean / std / 分位数）。kai0 提供快速版：

```bash
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_normal
```

- 直接扫描 `config.data.repo_id` 下的所有 parquet（或 `--base-dir` 指定目录）。
- 对 `observation.state` / `action` 做与 openpi 一致的 `process_state / process_actions`
  （补齐到 `action_dim`、裁剪异常值到 `[-π, π]`）。
- 用 `RunningStats` 按 batch=32 累积，产出 `norm_stats.json`，保存到
  `assets/<config_name>/<repo_id>/norm_stats.json`。
- `train.py` 启动时会自动从该路径把 `norm_stats.json` 拷贝到 checkpoint 目录。
- 参数：`--config-name`（必填）、`--base-dir`（可选）、`--max-frames`（可选，快速验证）。
- 价值模型配置（`ADVANTAGE_TORCH_*`）使用 `skip_norm_stats=True`，**无需**计算 norm stats。

---

## 6. JAX 全参数微调

```bash
# 1) 计算归一化统计（如未做）
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_normal

# 2) 全参数微调（JAX / Flax NNX）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_normal --exp_name=run1
```

- 入口 `scripts/train.py`：初始化 wandb（`config.wandb_enabled`）、构建 checkpoint manager、
  从 `assets/<config>/<repo_id>/norm_stats.json` 拷贝统计、按 `config.freeze_filter` 计算可训练参数、
  以 `save_interval` 保存、`keep_period` 保留关键 step。
- 关键超参（`pi05_flatten_fold_normal_follow_paper`）：`batch_size=128`（32 workers）、
  `num_train_steps=30_000`、`peak_lr=2.5e-5`（余弦退火）、`AdamW + clip_gradient_norm=1.0`、
  `ema_decay=0.99`、`keep_period=4000`。
- 8×A100 训练约 **20 小时数据**即可获得论文中的生产级效果（见 `docs/results.md`）。

---

## 7. LoRA 低显存微调

通过 `model.paligemma_variant / action_expert_variant` 的 `_lora` 变体实现（`get_freeze_filter`）：

```bash
# 以 follow_paper_lora 配置为例（config.py 已定义，冻结语言 LLM，动作专家可训练）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_normal_follow_paper_lora --exp_name=lora1
```

- LoRA 配置将显存需求从 >70GB 降到 **>22.5GB**，单张 RTX 4090 即可微调。
- 冻结点：`paligemma_variant="gemma_2b_lora"` → 冻结 LLM 参数；`action_expert_variant="gemma_300m_lora"`
  → 冻结动作专家参数；最终排除所有 `lora` 参数不参与冻结（保持可训练）。

---

## 8. PyTorch / DDP 训练

`scripts/train_pytorch.py` 是 JAX 训练器的 PyTorch 镜像，使用 `PI0Pytorch` 模型与同一套
`config.py` / `data_loader.py`，支持单卡与多卡 DDP。

```bash
# 单卡
uv run python scripts/train_pytorch.py pi05_flatten_fold_normal --exp_name=run1

# 单机多卡（DDP）
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_flatten_fold_normal --exp_name=run1

# 多机
uv run torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=run1

# 从最近 checkpoint 恢复
uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py <config_name> --exp_name=run1 --resume
```

- 训练精度默认 `bfloat16`（`pytorch_training_precision`）。
- 支持 gradient checkpointing（`PI0Pytorch.gradient_checkpointing_enable`）以降低显存。
- 支持 FSDP（`--fsdp_devices`，通过 config 字段）。

---

## 9. 价值模型（Advantage Estimator）训练

价值模型是 **Stage Advantage** 模块的第 1 步（Stage 0 手工标注 → Stage 1 训练估计器）。
训练用 PyTorch 入口 + `ADVANTAGE_TORCH_*` 配置：

```bash
# 单卡（batch_size=16）
uv run python scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD --exp_name=value1

# 8 卡（batch_size=128）
uv run torchrun --standalone --nproc_per_node=8 \
    scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD --exp_name=value1
```

关键点：

- **数据**：`AdvantageLerobotDataset` 读取带 `stage_progress_gt` 的 parquet；每帧采样一个
  同 episode 历史帧（键前缀 `his_-100_`），回归目标 `progress = stage_progress_gt - his_gt`。
- **模型**：`AdvantageEstimator`（`models_pytorch/pi0_pytorch.py`），π₀.₅ 主干 + 3 层 MLP 价值头。
- **损失**：`loss_value_weight=1.0`、`loss_action_weight=0.0`（动作 loss 关闭）。
- **`skip_norm_stats=True`**：价值模型不需要归一化统计。
- **checkpoint**：`checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/<exp_name>/<step>/`
  （含 `model.safetensors`、`optimizer.pt`、`metadata.pt`），`save_interval=10000`。
- 仓库内已训练的 20000 步价值模型见 `docs/results.md` 的「价值模型 checkpoint」小节。

完整流程（Stage 0→3）见 `stage_advantage/README.md` 与 `stage_advantage/annotation/README.md`。

---

## 10. AWBC 优势加权行为克隆训练

AWBC 是 Stage Advantage 的最终产物训练（Step 4）：把离散化的优势标签通过 `task_index` 注入语言
prompt（如 `"fold the cloth, Advantage: positive"`），再按 `prompt_from_task=True` 做语言条件模仿学习。

```bash
# 归一化统计（AWBC 数据集同样需要）
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_awbc

# 训练（JAX）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc --exp_name=awbc1
```

- 三个任务：`pi05_flatten_fold_awbc`（Task FlattenFold）、`pi05_tee_shirt_sort_awbc`、
  `pi05_hang_cloth_awbc`。
- 数据来自**预标注的 advantage 子集**（`data/FlattenFold/advantage`，含 `task_index` + `tasks.jsonl`），
  或自行走 Stage 0–3 生成（`stage_advantage/README.md`）。
- `base_config=DataConfig(prompt_from_task=True)` 是 AWBC 的条件机制。
- 推理时必须使用与训练一致的 prompt 格式。

---

## 11. 模型算术（Model Arithmetic）融合

`model_arithmetic/` 在**权重空间**合并多个子集训练出的 checkpoint（无需 MoE）。支持 6 种方法：
`average / inverse_loss / gradient_descent / adaptive_gradient_descent / greedy / manual(--weights)`。

```bash
# JAX checkpoint（Orbax/OCDBT）
python model_arithmetic/arithmetic.py \
    --config pi05_flatten_fold_normal_follow_paper_split2 \
    --data-path kai0_taskA_val.pkl \
    --checkpoints ckpt_split1 ckpt_split2 ckpt_split3 \
    --output /path/to/mixed \
    --optimize_method inverse_loss --use_gpu

# PyTorch checkpoint（model.safetensors）
python model_arithmetic/arithmetic_torch.py \
    --config pi05_flatten_fold_normal --data-path val.pkl \
    --checkpoints dir1 dir2 --output /path/to/mixed --optimize_method average
```

配套脚本：

- `split_data.py`：把一个 LeRobot 数据集按 episode 切成互不相交子集，用于「分而治之」训练。
- `dump_data.py`：导出小验证集 `.pkl`，用于评分 / 优化融合权重。
- `common.py`：共享的融合/评分辅助逻辑。

详见 `model_arithmetic/README.md`。

---

## 12. Checkpoint 管理与日志

| 内容 | 位置 / 说明 |
| --- | --- |
| JAX checkpoint | `checkpoints/<config>/<exp_name>/<step>/`（Orbax 目录，含 `params` 与 `norm_stats.json`） |
| PyTorch checkpoint | `checkpoints/<config>/<exp_name>/<step>/`（`model.safetensors` + `optimizer.pt` + `metadata.pt` + `assets/`） |
| wandb | `config.wandb_enabled=True` 时记录 loss / 图像 / 指标；`WANDB_MODE=offline` 可离线 |
| 关键 step 保留 | `keep_period`：`step % keep_period == 0` 的 checkpoint 不被清理 |
| 恢复训练 | `--resume`（PyTorch）；JAX 自动从 checkpoint manager 恢复 |

---

## 13. 常用训练配置速查表

| 配置名 | 用途 | 关键超参 |
| --- | --- | --- |
| `pi05_flatten_fold_normal` | FlattenFold 普通全参模板 | 100k 步、batch 256、keep 5000 |
| `pi05_flatten_fold_normal_follow_paper` | 论文复现全参（base_clean_split1） | 30k 步、batch 128、lr 2.5e-5、keep 4000 |
| `pi05_flatten_fold_normal_follow_paper_{split2,split3,4_splits_index_0..3}` | 数据切分后单独训练（供模型算术融合） | 同上 |
| `pi05_flatten_fold_normal_follow_paper_space_mirror` | 空间镜像增广数据全参 | 同上 |
| `pi05_flatten_fold_normal_follow_paper_{dagger,base_dagger_merge,base_dagger_merge_105K}` | 加入 DAgger / 合并数据 | 同上 |
| `pi05_flatten_fold_normal_follow_paper_{lora,lora_magiclab}` | FlattenFold LoRA | 低显存 |
| `pi05_tee_shirt_sort_normal{,_lora}` / `pi05_hang_cloth_normal` | TeeShirtSort / HangCloth 全参（+LoRA） | 同上 |
| `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD` | 价值模型（两时间步，KAI0） | 100k 步、batch 16/1卡(或128/8卡)、save 10000 |
| `ADVANTAGE_TORCH_PI06_FLATTEN_FOLD` | 价值模型（单时间步，PI06） | 同上 |
| `pi05_flatten_fold_awbc` | AWBC 训练 | 100k 步、batch 256、`prompt_from_task` |
| `pi05_tee_shirt_sort_awbc` / `pi05_hang_cloth_awbc` | 其余任务 AWBC | 同上 |
| `pi05_rtc_flatten_fold_inference` | **RTC 推理服务**（部署用，非训练） | JAX checkpoint |
