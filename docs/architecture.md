# 网络架构（Network Architecture）

> 本文详细介绍 χ₀ (kai0) 的核心网络架构：视觉编码、语言-动作主干、动作生成（Flow Matching）、
> 实时分块推理（RTC）与价值模型（Advantage Estimator）。代码位置：`src/openpi/models/`（JAX/Flax）
> 与 `src/openpi/models_pytorch/`（PyTorch）。

---

## 目录

- [1. 总体设计](#1-总体设计)
- [2. 视觉编码器：SigLIP ViT](#2-视觉编码器siglip-vit)
- [3. 语言 / 动作主干：PaliGemma + Action Expert](#3-语言--动作主干paligemma--action-expert)
- [4. 输入 / 输出张量](#4-输入--输出张量)
- [5. 注意力掩码与位置编码](#5-注意力掩码与位置编码)
- [6. 动作生成：Flow Matching](#6-动作生成flow-matching)
- [7. 实时分块推理（Pi0RTC）](#7-实时分块推理pi0rtc)
- [8. 价值模型（AdvantageEstimator）](#8-价值模型advantageestimator)
- [9. JAX / PyTorch 双实现对照](#9-jax--pytorch-双实现对照)
- [10. 相关文件索引](#10-相关文件索引)

---

## 1. 总体设计

kai0 复用 [openpi](https://github.com/Physical-Intelligence/openpi) 的 π₀ / π₀.₅ 模型族，并针对
**双臂衣物操作**做适配。整体是一个**视觉-语言-动作（VLA）**多模态 Transformer：

```text
三路相机图像 (top_head / hand_left / hand_right)
        │  SigLIP ViT (So400m/14) 视觉编码
        ▼
视觉 token ─────────────┐
                        ├──▶ PaliGemma LLM（Prefix：图像 + 语言提示）
语言提示 (task prompt) ──┘
                        │
14 维状态 ──▶ Action Expert（Suffix：状态 + 带噪动作 + 时间步）
                        │
                        ▼
              Flow Matching 去噪 ──▶ 动作分块 [action_horizon, 14]
```

- **单网络双分支**：同一个 Gemma 解码器同时承载「语言分支」与「动作分支」（Action Expert）。
- **动作分块（Action Chunking）**：一次预测未来 `action_horizon = 50` 步、每步 14 维动作。
- **两种模型代次**：
  - `π₀`（`pi05=False`）：状态作为连续输入拼接进 suffix，动作专家用 `action_time_mlp` 注入时间步。
  - `π₀.₅`（`pi05=True`）：状态进入**离散语言 token**（Prefix），动作专家使用 **adaRMSNorm** 注入
    Flow Matching 时间步。kai0 的衣物操作训练全部基于 π₀.₅。

---

## 2. 视觉编码器：SigLIP ViT

- **模型**：`SigLIP`（`So400m/14`，`pool_type="none"`，`scan=True`），见 `src/openpi/models/siglip.py`。
- **输入**：三路相机图像统一 resize 到 **224×224×3**（`_model.IMAGE_RESOLUTION = (224, 224)`）。
- **输出**：每路图像被编码为一串**视觉 token**（无池化，保留空间网格），与语言 token 一起进入
  Prefix。三路相机在 batch 维度上分别编码、拼接。
- **冻结使用**：训练时视觉编码器以 `train=False` 前向（仅推理路径，不做 dropout/BN 统计），
  即视觉编码器参数**不参与梯度更新**。

---

## 3. 语言 / 动作主干：PaliGemma + Action Expert

PaliGemma 主干（`src/openpi/models/gemma.py`）包含两个 Gemma 解码器 config：

| 分支 | 默认变体 | 参数规模 | 职责 |
| --- | --- | --- | --- |
| 语言分支 `paligemma_variant` | `gemma_2b` | ~2B | 处理图像 token 与语言提示（Prefix） |
| 动作分支 `action_expert_variant` | `gemma_300m` | ~300M | 处理状态 + 带噪动作（Suffix） |

在 `Pi0.__init__`（`src/openpi/models/pi0.py`）中：

- `llm = gemma.Module(configs=[paligemma_config, action_expert_config], adarms=pi05)`
  —— 两个 Gemma 解码器共享权重前缀（prefix-LM），仅 Action Expert 分支独立尾层。
- `adarms=[False, True]`：π₀.₅ 时动作分支启用 **adaRMSNorm**，用 Flow Matching 时间步调制
  层归一化（`time_mlp_in / time_mlp_out` 双 MLP）。
- 投影层：
  - `action_in_proj`：14 维动作 → 动作专家宽度；
  - `action_out_proj`：动作专家宽度 → 14 维动作（速度场输出）；
  - `time_mlp_in / time_mlp_out`（π₀.₅）或 `state_proj / action_time_mlp_*`（π₀）。

### LoRA 支持

通过 `Pi0Config.get_freeze_filter()` 实现选择性冻结：
- `paligemma_variant="gemma_2b_lora"`：冻结 LLM（语言分支）参数，动作专家可训练；
- `action_expert_variant="gemma_300m_lora"`：冻结动作专家参数；
- 最终**排除**所有 `lora` 参数（冻结时），实现低显存微调（见 `docs/training.md`）。

---

## 4. 输入 / 输出张量

### 模型输入（`model.Observation`，见 `src/openpi/models/model.py`）

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `images` | `{top_head, hand_left, hand_right}: [B, 224, 224, 3]` | 三路相机图像 |
| `image_masks` | `[B]` | 图像有效性掩码 |
| `state` | `[B, action_dim=14]` | 归一化后的机械臂状态 |
| `tokenized_prompt` | `[B, max_token_len]` | PaliGemma 分词后的语言提示 |
| `tokenized_prompt_mask` | `[B, max_token_len]` | 语言 token 掩码 |
| `actions`（训练） | `[B, action_horizon=50, action_dim=14]` | 教师动作分块 |

- `max_token_len`：π₀.₅ 默认 200（状态作为离散 token 进入 prompt），π₀ 默认 48。
- `action_dim` 默认 32（openpi 惯例），kai0 的衣物配置中设为 **14**（左右臂 6+6 关节 + 2 夹爪）。

### 模型输出

- **训练**：返回动作速度场预测，计算 MSE 损失（Flow Matching）。
- **推理（`sample_actions`）**：返回动作分块 `[action_horizon, action_dim]`，即未来 50 步的
  左右臂关节角 + 夹爪开合指令。
- **π₀-FAST 变体**（`pi0_fast.py`）：离散 FAST tokenizer，输出经 `ExtractFASTActions` 还原为连续动作。

---

## 5. 注意力掩码与位置编码

- **`make_attn_mask(input_mask, mask_ar)`**：根据每个 token 的 `ar`（autoregressive）标记构建
  **前缀-语言模型（prefix-LM）** 注意力掩码：
  - 图像 token 与图像之间、图像与语言之间为**双向注意力**（`ar=False`）；
  - 语言提示内部为**因果注意力**；
  - 动作（suffix）token 之间为**分块因果注意力**（chunked causal），每个动作 token 只看到
    同一分块内较早的 token。
- **`posemb_sincos`**：正弦-余弦位置嵌入，用于 suffix 的 token 位置编码。
- **KV Cache 复用**（RTC）：推理时 Prefix（图像 + 语言）只前向一次并填充 KV Cache，
  suffix 去噪每步复用缓存，避免重复计算视觉/语言 token。

---

## 6. 动作生成：Flow Matching

动作生成采用 **Flow Matching**（与 π₀ 一致）而非离散化 token：

- **训练**：给定干净动作 `a`，采样噪声 `ε ~ N(0, I)` 与时间 `t ~ U(0,1)`，
  - 带噪动作 `x_t = t·ε + (1-t)·a`
  - 目标速度场 `u_t = ε - a`
  - 损失 `L = MSE(u_t, v_t(x_t, t))`（对 action 维取平均）。
- **推理**：从 `t=1` 的纯噪声 `x_1 ~ N(0, I)` 出发，用学到的速度场做 **Euler 积分**：
  `x_{t-Δt} = x_t + Δt·v_t`，`Δt = 1/num_steps`（默认 `num_steps=10`），迭代到 `t=0` 得到干净动作。
- **随机采样**：`sample_actions(rng, observation, num_steps=10)` 由 `rng` 驱动（非确定性采样）。

---

## 7. 实时分块推理（Pi0RTC）

`Pi0RTC`（`src/openpi/models/pi0_rtc.py`）在 π₀/π₀.₅ 基础上增加 **Real-Time Chunking（RTC）**
推理支持，专为双臂高频闭环控制设计。模型结构与 π₀.₅ 一致，但 `sample_actions` 增加参数：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `prev_action_chunk` | `None` | 上一轮已生成的动作分块，用于时序引导 |
| `inference_delay` | `None` | 推理延迟补偿（帧数） |
| `execute_horizon` | `None`（=action_horizon） | 本次实际执行的动作步数 |
| `max_guidance_weight` | `0.5` | RTC 引导强度上限 |
| `prefix_attention_schedule` | `"exp"` | 前缀注意力衰减调度 |
| `enable_rtc` | `True` | 是否启用 RTC 引导 |

**RTC 去噪循环**：`base_step`（普通 Euler 步）与 `rtc_step`（带上一分块引导的步）交替；
`get_prefix_weights` 控制新分块对上一分块动作的「信任权重」随时间衰减，从而实现**连续分块间的
平滑衔接**，避免每 50 步重规划造成的跳变。

**使用方式**：推理服务用 `pi05_rtc_flatten_fold_inference` 配置加载（JAX checkpoint），机器人侧
`arx_openpi_inference_rtc.py` 以 `--rtc_mode` 连接。详见 `docs/inference_deployment.md`。

---

## 8. 价值模型（AdvantageEstimator）

`AdvantageEstimator`（`src/openpi/models_pytorch/pi0_pytorch.py`，继承 `PI0Pytorch`）是 kai0
**阶段优势（Stage Advantage）** 模块的核心模型，用于从观测回归每帧「优势 / 进度」值。

- **结构**：π₀.₅ 主干 + 一个 3 层 MLP 价值头
  （`Linear→SiLU→Linear→SiLU→Linear(1)→Tanh`），取 suffix 首 token（状态 token）的最终
  表示做回归。
- **训练**：`loss_value_weight=1.0`、`loss_action_weight=0.0` —— 动作 loss 关闭，仅回归价值：
  - 回归目标 `progress = stage_progress_gt − his_-100_stage_progress_gt`（同一 episode 随机历史帧的
    阶段进度差，即「相对进度/优势」）；
  - 数据加载用 `AdvantageLerobotDataset`（`src/openpi/training/data_loader.py`），为每个样本采样
    一个同 episode 历史帧（键前缀 `his_-100_`）。
- **推理（`sample_values`）**：输入当前帧三路图像 + 状态 + 语言提示，输出 `absolute_value ∈ [0,1]`
  （累计进度）。不需要 `stage_progress_gt`，因此**无需人工标注即可对任意轨迹预测进度**。
- **配置**：`ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD`（两时间步阶段级，KAI0）与
  `ADVANTAGE_TORCH_PI06_FLATTEN_FOLD`（单时间步，PI06）。

价值模型产出的 `absolute_value / relative_advantage / absolute_advantage` 三列会被写回 parquet，
经离散化后作为 AWBC 的语言条件（见 `docs/training.md` 与 `stage_advantage/README.md`）。

---

## 9. JAX / PyTorch 双实现对照

| 组件 | JAX / Flax（NNX） | PyTorch |
| --- | --- | --- |
| 模型 | `models/pi0.py`（`Pi0`）、`models/pi0_rtc.py`（`Pi0RTC`）、`models/pi0_fast.py` | `models_pytorch/pi0_pytorch.py`（`PI0Pytorch`、`AdvantageEstimator`） |
| 配置 | `models/pi0_config.py`（`Pi0Config` / `Pi0RTCConfig`） | 复用 `models/pi0_config.py`（PyTorch 模型直接消费同一份配置） |
| 训练入口 | `scripts/train.py`（全参 / LoRA） | `scripts/train_pytorch.py`（单卡 / torchrun DDP） |
| 推理 | `serve_policy.py`（JAX checkpoint 服务） | `stage_advantage/annotation/eval.py`（价值模型） |
| 融合 | `model_arithmetic/arithmetic.py` | `model_arithmetic/arithmetic_torch.py` |

> 两个实现共享同一套 `config.py` 与数据管线；JAX 用于策略全参/LoRA 训练与部署服务，
> PyTorch 用于价值模型训练与模型算术融合（另有纯推理用途）。

---

## 10. 相关文件索引

| 文件 | 内容 |
| --- | --- |
| `src/openpi/models/model.py` | `BaseModel` / `Observation` / `Actions` 基类与接口 |
| `src/openpi/models/pi0_config.py` | `Pi0Config` / `Pi0RTCConfig` 配置（pi05、LoRA 冻结逻辑） |
| `src/openpi/models/pi0.py` | π₀ / π₀.₅ JAX 实现（embed_prefix / embed_suffix / sample_actions） |
| `src/openpi/models/pi0_rtc.py` | RTC 实时分块推理实现 |
| `src/openpi/models/pi0_fast.py` | π₀-FAST（离散动作 tokenizer） |
| `src/openpi/models/gemma.py` | PaliGemma / Gemma 主干（含 adaRMSNorm） |
| `src/openpi/models/siglip.py` / `vit.py` | SigLIP 视觉编码器 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | PyTorch 版 π₀ + `AdvantageEstimator` |
| `src/openpi/transforms.py` | 数据变换链（归一化、重排、tokenize 等） |
| `src/openpi/policies/` | 策略封装（agilex / arx / aloha / droid / libero / magicg1） |
