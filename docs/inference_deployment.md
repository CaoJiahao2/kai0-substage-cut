# 推理与部署（Inference & Deployment）

> 本文介绍 kai0 的推理与部署体系：策略服务（WebSocket）、机器人侧客户端、双机架构、
> 实时分块（RTC）、时序平滑 / 集成，以及价值模型（Advantage Estimator）的批量推理。

---

## 目录

- [1. 部署总览：双机架构](#1-部署总览双机架构)
- [2. 策略服务 serve_policy](#2-策略服务-serve_policy)
- [3. 客户端 openpi-client](#3-客户端-openpi-client)
- [4. 机器人侧推理脚本](#4-机器人侧推理脚本)
  - [4.1 Agilex Piper（ROS Noetic）](#41-agilex-piperros-noetic)
  - [4.2 ARX X5（ROS 2）](#42-arx-x5ros-2)
- [5. 实时分块推理（RTC）](#5-实时分块推理rtc)
- [6. 时序平滑与集成](#6-时序平滑与集成)
- [7. 价值模型（Advantage Estimator）批量推理](#7-价值模型advantage-estimator批量推理)
- [8. 测试与评估](#8-测试与评估)
- [9. 常见问题](#9-常见问题)

---

## 1. 部署总览：双机架构

kai0 采用 **GPU 主机 + 机器人 IPC** 的双机部署：

```text
┌─────────────────────────────┐         WebSocket + msgpack          ┌─────────────────────────────┐
│  GPU 主机（远程推理）         │ ◄───────────────────────────────────► │  机器人 IPC（实机控制）        │
│  • serve_policy.py           │          观测 / 动作分块              │  • ROS Noetic (Agilex)       │
│  • 加载 checkpoint，维护模型  │                                       │  • ROS 2 (ARX)               │
│  • 归一化、tokenize 在服务端  │                                       │  • openpi_client 客户端       │
└─────────────────────────────┘                                       │  • 关节 / 夹爪 / 相机采集      │
                                                                      └─────────────────────────────┘
```

- **GPU 主机**：运行策略 WebSocket 服务，负责模型推理（归一化、视觉编码、Flow Matching 采样）。
- **机器人 IPC**：运行 ROS + 推理脚本，采集三路相机图像与关节状态，调用服务端获取动作分块并执行。
- 职责分离带来两个好处：机器人端环境与训练环境解耦（避免依赖冲突）；可在更强大的 GPU 上做推理。

---

## 2. 策略服务 serve_policy

**脚本**：`scripts/infer/serve_policy.py`

**加载已训练 checkpoint（推荐，kai0 用法）**：

```bash
uv run scripts/infer/serve_policy.py policy:checkpoint \
  --policy.config=pi05_flatten_fold_normal \
  --policy.dir=/path/to/checkpoints/pi05_flatten_fold_normal/run1/<step> \
  --port=8000
```

- **RTC 推理**：机器人侧若以 `--rtc_mode` 连接，服务端须用 RTC 配置：

```bash
uv run scripts/infer/serve_policy.py policy:checkpoint \
  --policy.config=pi05_rtc_flatten_fold_inference \
  --policy.dir=<ckpt_dir> --port=8000
```

> 注意：`pi05_rtc_flatten_fold_inference` 目前仅支持 **JAX checkpoint**。

**加载默认策略（openpi 内置）**：

```bash
uv run scripts/infer/serve_policy.py --env=ALOHA_SIM --port=8000
# 等价于
uv run scripts/infer/serve_policy.py policy:checkpoint \
  --policy.config=pi05_aloha --policy.dir=gs://openpi-assets/checkpoints/pi05_base
```

支持的 `--env`：`ALOHA / ALOHA_SIM / DROID / LIBERO`。

**参数**：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--port` | `8000` | 服务端口 |
| `--default-prompt` | `None` | 数据缺少 prompt 时的默认语言指令 |
| `--record` | `False` | 记录策略行为便于调试 |

服务内部通过 `openpi.serving.websocket_policy_server` 暴露 WebSocket 接口，传输使用
**msgpack**（numpy 二进制），与 `packages/openpi-client` 配合。

---

## 3. 客户端 openpi-client

**包**：`packages/openpi-client`（独立 pip 包，工作区成员，机器人端唯一依赖）。

```bash
cd packages/openpi-client
pip install -e .          # 或 uv pip install -e packages/openpi-client
```

**核心接口**：`openpi_client/websocket_client_policy.py` 的 `WebsocketClientPolicy`。

```python
from openpi_client import image_tools, websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="<gpu_host_ip>", port=8000)

observation = {
    "observation/image": image_tools.convert_to_uint8(
        image_tools.resize_with_pad(top_img, 224, 224)
    ),
    "observation/wrist_image": image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, 224, 224)
    ),
    "observation/state": state,          # 14 维，可传未归一化，服务端处理
    "prompt": task_instruction,
}
action_chunk = client.infer(observation)["actions"]   # shape [action_horizon, 14]
```

- 图像在**客户端**先 resize 到 224×224（减小带宽 / 延迟），并以 uint8 传输。
- `state` 可传未归一化值，归一化在服务端完成。
- 典型控制：每 `action_horizon`（如 50）步调用一次 `infer`，其余步开环执行分块内的动作。

---

## 4. 机器人侧推理脚本

### 4.1 Agilex Piper（ROS Noetic）

路径：`train_deploy_alignment/inference/agilex/inference/`

```bash
# 时序平滑（推荐日常部署）
python train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_temporal_smoothing.py \
  --host <gpu_host_ip> --port 8000 --ctrl_type joint --use_temporal_smoothing --chunk_size 50

# RTC 实时分块
python train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_rtc.py \
  --host <gpu_host_ip> --port 8000 --ctrl_type joint --rtc_mode --chunk_size 50

# 时序集成（多分块投票）
python train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_temporal_ensembling.py \
  --host <gpu_host_ip> --port 8000 --ctrl_type joint

# 同步（无平滑，最低延迟）
python train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_sync.py \
  --host <gpu_host_ip> --port 8000 --ctrl_type joint
```

关键参数：`--host`（GPU 主机 IP）、`--port`、`--ctrl_type`（`joint` 或 `ee`）、
`--chunk_size`、`--use_temporal_smoothing`、`--rtc_mode`。ROS 节点在
`inference/agilex/Piper_ros_private-ros-noetic/` 下，依赖 `piper` 驱动包。

### 4.2 ARX X5（ROS 2）

路径：`train_deploy_alignment/inference/arx/inference/`

```bash
# RTC 模式（服务端须用 pi05_rtc_flatten_fold_inference 配置）
python train_deploy_alignment/inference/arx/inference/arx_openpi_inference_rtc.py \
  --host <gpu_host_ip> --port 8000 --rtc_mode --chunk_size 50

# 时序平滑
python train_deploy_alignment/inference/arx/inference/arx_openpi_inference_temporal_smooth.py \
  --host <gpu_host_ip> --port 8000

# 同步
python train_deploy_alignment/inference/arx/inference/arx_openpi_inference_sync.py \
  --host <gpu_host_ip> --port 8000
```

ARX 相关参数：相机序列号（`--camera_*_serial`）、关节话题（`--joint_*_topic_*`）、
`--control_frequency`（默认 30Hz）、`--inference_rate`、RTC 引导参数
（`--rtc_mask_prefix_delay`、`--rtc_max_guidance_weight`、`--rtc_execute_horizon`）、
夹爪开合阈值（`--gripper_open` / `--gripper_close` / `--gripper_thresh`）等。
环境搭建见 `train_deploy_alignment/inference/arx/README.md` 与 `arx/setup.sh`。

---

## 5. 实时分块推理（RTC）

RTC（Real-Time Chunking）通过 `Pi0RTC`（`src/openpi/models/pi0_rtc.py`）实现：

- 服务端用 `pi05_rtc_flatten_fold_inference` 配置加载，客户端以 `--rtc_mode` 请求。
- 推理时**前缀（图像 + 语言）只前向一次并复用 KV Cache**；去噪过程用上一动作分块做引导
  （`rtc_step`），`get_prefix_weights` 按指数衰减控制对新/旧分块的信任，使相邻分块平滑衔接。
- 支持 `--rtc_execute_horizon`（本次实际执行步数）与 `--rtc_mask_prefix_delay`（延迟前缀掩码），
  用于补偿推理延迟。

---

## 6. 时序平滑与集成

针对分块推理常见的边界跳变，kai0 提供三种降抖动策略（部署脚本可选其一）：

| 策略 | 脚本参数 | 原理 |
| --- | --- | --- |
| 时序平滑 Temporal Smoothing | `--use_temporal_smoothing` | 对相邻分块的动作做指数/滑窗平滑，降低切换跳变 |
| 时序集成 Temporal Ensembling | `--use_temporal_ensembling` | 维护多个历史分块并加权融合（投票），提高鲁棒性 |
| 实时分块 RTC | `--rtc_mode` | 在模型采样层面用上一分块引导去噪，从源头平滑 |

---

## 7. 价值模型（Advantage Estimator）批量推理

价值模型的批量推理用于为**未标注数据集**生成优势标签（Stage Advantage Step 2），
详见 [README § 10（推理与部署）](../README.md#10-推理与部署) 与 `stage_advantage/annotation/README.md`。

```bash
cd <repo_root>
export WANDB_MODE=${WANDB_MODE:-offline}

# KAI0（两时间步）：输出 absolute_value / relative_advantage / absolute_advantage
uv run python stage_advantage/annotation/eval.py Task-A KAI0 /path/to/dataset

# PI06（单时间步）：仅输出 absolute_value
uv run python stage_advantage/annotation/eval.py Task-A PI06 /path/to/dataset
```

- **模型 checkpoint**：`value_model/20000/`（7.5GB `model.safetensors`，config
  `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD_FULL_FINETUNE_MIX_02`，`full_finetune_460_run_16`）；
  `value_model_version2/20000/` 为更早版本（run_15）。在 `eval.py` 的 `MODELS_CONFIG_MAP` 中把
  `ckpt_dir` 指向对应目录、`ckpt_steps=20000`。
- **输入**：三路视频 + 14 维状态 + 语言提示；`KAI0` 模式额外参考同一 episode 的历史帧
  （键前缀 `his_-100_`）。
- **输出**：写回 `data_KAI0_100000/` 下的新 parquet（原 `data/`、`videos/` 不变），三列为
  `absolute_value`（累计进度）、`relative_advantage`（相对 50 帧的进度差）、
  `absolute_advantage`（n+50 与 n 帧绝对值差，裁剪到 [-1,1]）。
- 该输出可直接进入 Step 3 `gt_label.py` 离散化，再用于 AWBC 训练。

**阶段分割评估**（可选，折纸箱任务）：

```bash
python scripts/infer/eval_stage_segmentation.py \
  --data-dir /path/to/data_infer --ckpt-dir /path/to/value_model/20000 \
  --config-name ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD --model-name KAI0 \
  --num 100 --seed 42 --seg-mode equal_width --abs-range 0.0 1.0 \
  --output-dir outputs/eval_stage_seg_results
```

评估结果示例见 `docs/results.md` 与 `outputs/eval_stage_seg_results/round1_v2/report.md`。

---

## 8. 测试与评估

### 8.1 训练 / 推理 L1 评估（离线）

```bash
# 用训练集评估动作 L1（JAX checkpoint）
uv run python scripts/eval_train_l1_jax.py --config-name pi05_flatten_fold_normal \
  --checkpoint-dir /path/to/checkpoints/pi05_flatten_fold_normal/run1/<step>

# PyTorch checkpoint
uv run python scripts/eval_train_l1_pytorch.py --config-name <cfg> \
  --checkpoint-dir /path/to/checkpoints/<cfg>/<exp>/<step>
```

### 8.2 数据 / 推理可视化

```bash
# GT 数据回放（Rerun）
bash scripts/viz/view_gt_replay.sh data/FlattenFold/base 0

# 启动 web viewer
uv run python scripts/viz/infer_viewer_server.py --host 0.0.0.0 --port 9001
```

### 8.3 单元测试 / lint

```bash
uv run ruff check .          # lint
uv run ruff format .         # 格式化
uv run pytest                # 测试（@pytest.mark.manual 默认排除）
```

### 8.4 实机评估

- 双机部署后，运行对应任务的 rollout（Agilex / ARX 推理脚本），统计成功率。
- 阶段分割评估见 § 7（对 value model 输出的 absolute_value 做等宽分段并对比人工 GT）。

---

## 9. 常见问题

| 问题 | 说明 |
| --- | --- |
| 服务端与客户端图像尺寸不一致 | 客户端统一 resize 到 224×224（`image_tools.resize_with_pad`）再传输 |
| RTC 模式无效果 | 确认服务端使用 `pi05_rtc_flatten_fold_inference`（JAX checkpoint），客户端加 `--rtc_mode` |
| 动作跳变 / 抖动 | 启用 `--use_temporal_smoothing` 或 `--use_temporal_ensembling` |
| 价值模型推理慢 | 增大 `--batch-size`（默认 32，可到 64），受显存限制（A800 40GB MIG 实测 ~11 分钟/episode） |
| `data_infer` 目录不可达 | 该目录为 TOS 只读挂载路径，需先 `tosutil cp` 拉到本地（见 `scripts/infer/run_stage_segmentation_eval.sh`） |
