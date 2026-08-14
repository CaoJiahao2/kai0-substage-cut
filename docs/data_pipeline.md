# 数据契约与工具（Data Contract & Pipeline）

> 本文详细介绍 kai0 的数据格式（LeRobot v2.1）、特征 schema、目录布局，以及
> `scripts/tools/` 下的数据工具链。数据卡（发布版）见 `docs/dataset.md`。

---

## 目录

- [1. 数据格式：LeRobot v2.1](#1-数据格式lerobot-v21)
- [2. 目录布局](#2-目录布局)
- [3. 特征 schema](#3-特征-schema)
- [4. 语言提示与 AWBC](#4-语言提示与-awbc)
- [5. 数据工具链（scripts/tools）](#5-数据工具链scriptstools)
- [6. 典型数据准备流程](#6-典型数据准备流程)
- [7. 归一化统计](#7-归一化统计)

---

## 1. 数据格式：LeRobot v2.1

- **存储**：每个任务目录下分 `base`（演示轨迹）与 `dagger`（DAgger 恢复轨迹）两个子集；
  任务级目录为 `FlattenFold`、`HangCloth`、`TeeShirtSort`。
- **文件**：`data/chunk-*/episode_*.parquet`（数值表）+ `videos/chunk-*/observation.images.*/episode_*.mp4`
  （三路视频）+ `meta/{info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl}`。
- **视频**：三路相机，480×640×3，AV1 编码，30 fps。

```text
FlattenFold/
├── base/
│   ├── data/
│   │   └── chunk-000/
│   │       └── episode_000000.parquet
│   ├── videos/
│   │   └── chunk-000/
│   │       ├── observation.images.top_head/episode_000000.mp4
│   │       ├── observation.images.hand_left/episode_000000.mp4
│   │       └── observation.images.hand_right/episode_000000.mp4
│   └── meta/
│       ├── info.json
│       ├── episodes.jsonl
│       ├── tasks.jsonl
│       └── episodes_stats.jsonl
└── dagger/                # 同上布局
HangCloth/ ...
TeeShirtSort/ ...
```

---

## 2. 目录布局

> 注意：README 早期版本中的 `data/Task_A|Task_B|Task_C` 已废弃，当前以
> **任务名**（`FlattenFold / HangCloth / TeeShirtSort`）组织。AWBC 的 advantage 子集为
> `data/<Task>/advantage`。

| 路径（仓库根下） | 说明 |
| --- | --- |
| `data/FlattenFold/base` | 展平+折叠 演示轨迹 |
| `data/FlattenFold/dagger` | 展平+折叠 DAgger 恢复轨迹 |
| `data/FlattenFold/advantage` | 预标注的优势标签数据（AWBC 用） |
| `data/HangCloth/{base,dagger,advantage}` | 悬挂任务 |
| `data/TeeShirtSort/{base,dagger,advantage}` | T 恤分类+整理任务 |

---

## 3. 特征 schema

parquet 每行一帧，字段如下：

| 字段 | 类型 / 形状 | 说明 |
| --- | --- | --- |
| `observation.state` | float32 `[N, 14]` | 左臂 `[:, :6]`、左夹爪 `[:, 6]`、右臂 `[:, 7:13]`、右夹爪 `[:, 13]` |
| `action` | float32 `[N, 14]` | 与 state 同布局的动作（关节角 + 夹爪开合） |
| `timestamp` | float32 `[N, 1]` | 距 episode 开始的时间（秒） |
| `frame_index` | int64 `[N, 1]` | 帧索引（0 起） |
| `episode_index` | int64 `[N, 1]` | episode 编号 |
| `index` | int64 `[N, 1]` | 全局唯一帧索引 |
| `task_index` | int64 `[N, 1]` | 任务编号（映射到 `meta/tasks.jsonl` 的语言提示） |

**价值模型额外列**（由 `eval.py` 写回 `data_KAI0_100000/`）：

| 列 | 说明 |
| --- | --- |
| `absolute_value` | 从首帧到当前帧的累计进度预测（0~1） |
| `relative_advantage` | 当前帧与「向后看 50 帧」的进度差（KAI0 两时间步模式） |
| `absolute_advantage` | 帧 `n+50` 与帧 `n` 的绝对值差，裁剪到 `[-1, 1]` |

---

## 4. 语言提示与 AWBC

- `meta/tasks.jsonl` 把 `task_index` 映射到自然语言指令。
- 普通训练：`default_prompt`（如 `"Flatten and fold the cloth."`）或 `prompt_from_task=True`（从数据读）。
- **AWBC**：`prompt_from_task=True` 时，离散化后的优势标签被注入 prompt
  （如 `"fold the cloth, Advantage: positive"`），策略通过语言通道对优势做条件模仿学习。

---

## 5. 数据工具链（scripts/tools）

| 脚本 | 作用 | 典型用法 |
| --- | --- | --- |
| `download_dataset.py` | 从 Hugging Face 下载 Kai0 数据集到 `./data`（或 `--local-dir`）。 | `python scripts/tools/download_dataset.py --tasks FlattenFold HangCloth TeeShirtSort` |
| `download_checkpoints.py` | 从 Hugging Face 下载各任务最优 checkpoint 到 `./checkpoints`。 | `python scripts/tools/download_checkpoints.py --tasks FlattenFold` |
| `convert_dataset_v30_to_v21.py` | LeRobot **V3**（`file-*.parquet`）→ **V2.1**（`episode_*.parquet`）。 | `python scripts/tools/convert_dataset_v30_to_v21.py --src /v3_ds --dst /v21_ds --video-mode copy` |
| `merge_lerobot_datasets.py` | 按 episode 合并多个数据集（如 `base`+`dagger`）。 | `python scripts/tools/merge_lerobot_datasets.py --source-path data/FlattenFold/base --source-path data/FlattenFold/dagger --output-path data/FlattenFold/base_dagger_merge` |
| `merge_lerobot.py` | 基于 `LeRobotDataset` 的通用合并（`--src_paths` / `--src_list` / `--features_json` / `--force`）。 | `python scripts/tools/merge_lerobot.py --src_paths d1 d2 --output-path merged` |
| `split_lerobot.py` | 按 episode 随机切分互不相交子集（供模型算术「分而治之」）。 | `python scripts/tools/split_lerobot.py --source_path data/FlattenFold/base --split_num 3` |
| `split_lerobot_dataset.py` | 封装 `split_lerobot.py` 的便捷入口。 | `python scripts/tools/split_lerobot_dataset.py --source-path data/FlattenFold/base --seed 42` |
| `remove_bad_episodes.py` | 排除坏 episode，重建干净数据集并重排索引。 | `python scripts/tools/remove_bad_episodes.py --source_path data/FlattenFold/base --dst_path data/FlattenFold/base_clean --exclude_episodes 176,2548` |
| `repair_lerobot_videos.py` | `ffprobe` 扫描损坏视频并从 Hub 重下（`--dry-run` 仅报告）。 | `python scripts/tools/repair_lerobot_videos.py --repo-id OpenDriveLab-org/Kai0 --root data/FlattenFold/base` |
| `sync_folders.sh` | `rsync` 文件夹同步/拷贝（`--delete`、`--dry-run`、锁与日志）。 | `bash scripts/tools/sync_folders.sh --src /mnt/a --dst /mnt/b --delete` |

---

## 6. 典型数据准备流程

```bash
# 1) 下载数据集与检查点
python scripts/tools/download_dataset.py --tasks FlattenFold HangCloth TeeShirtSort
python scripts/tools/download_checkpoints.py --tasks FlattenFold

# 2)（如需本地 V3 数据）转换为 V2.1
python scripts/tools/convert_dataset_v30_to_v21.py --src /v3_ds --dst data/FlattenFold/base --video-mode copy

# 3) 清洗坏 episode
python scripts/tools/remove_bad_episodes.py \
  --source_path data/FlattenFold/base --dst_path data/FlattenFold/base_clean \
  --exclude_episodes 176,2548

# 4) 合并 base + dagger（或用 split_lerobot.py 切分后分别训练再融合）
python scripts/tools/merge_lerobot_datasets.py \
  --source-path data/FlattenFold/base_clean --source-path data/FlattenFold/dagger \
  --output-path data/FlattenFold/base_dagger_merge
python scripts/tools/split_lerobot.py --source_path data/FlattenFold/base_clean --split_num 3
```

---

## 7. 归一化统计

- 训练前须为数据集计算归一化统计（`norm_stats.json`）：`scripts/compute_norm_states_fast.py`。
- 保存到 `assets/<config_name>/<repo_id>/norm_stats.json`，`train.py` 启动时自动拷贝到 checkpoint 目录。
- 价值模型配置（`ADVANTAGE_TORCH_*`）使用 `skip_norm_stats=True`，无需计算。
- 详细说明见 `docs/norm_stats_fast.md` 与 `docs/training.md` § 5。
