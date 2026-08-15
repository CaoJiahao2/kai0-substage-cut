---
license: apache-2.0
base_model: openpi/pi0.5  # 或其它 π₀.₅ 基础 checkpoint，按实际填写
tags:
  - robotics
  - imitation-learning
  - manipulation
  - advantage-estimator
  - stage-advantage
  - kai0
pipeline_tag: reinforcement-learning
model-index:
  - name: kai0-value-model-adv-flatten-fold-run16
    results:
      - task:
          type: robot-manipulation
        metrics:
          - name: Stage Segmentation Boundary MAE
            type: stage-boundary-mae
            value: 60.1
          - name: Boundary Acc @±50f
            type: boundary-acc
            value: 77.7
---

# χ₀ (kai0) — Advantage Estimator（价值模型）

本目录发布 **Advantage Estimator**（π₀.₅ 的价值模型变体）权重，用于
**Advantage-Weighted Behavior Cloning（AWBC）** 与 **阶段分割（Stage Segmentation）**。

> 仅含模型权重（`model.safetensors` + `metadata.pt`）。不含训练优化器状态 `optimizer.pt`。

---

## 1. 文件清单

| 文件 | 说明 |
| --- | --- |
| `model.safetensors` | 模型权重（~7.5 GB），PyTorch `safetensors` 格式 |
| `metadata.pt` | 训练元信息（config 名 / exp_name / 训练步数 / 超参数等） |

> 注意：训练时保存的 `optimizer.pt`（~13.5 GB）为优化器状态，**未随本 repo 发布**，对推理无影响。

---

## 2. 模型信息

| 项 | 值 |
| --- | --- |
| 模型类型 | Advantage Estimator（π₀.₅ 变体，仅保留价值/进度回归头） |
| 基于 | π₀.₅ 基础 checkpoint（见 `metadata.pt` / `pytorch_weight_path`） |
| 训练步数 | 20000 |
| 训练损失 | `loss_value_weight=1.0`，`loss_action_weight=0.0`（禁用动作损失） |
| 归一化 | `skip_norm_stats=True`（价值模型不依赖 norm_stats） |
| 输出 | 每帧 `absolute_value`（0~1 累计进度） |

**Config**：`ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD_FULL_FINETUNE_MIX_02`

---

## 3. 版本

| 子目录 | Config | exp_name | 说明 |
| --- | --- | --- | --- |
| `run_16/`（推荐） | `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD_FULL_FINETUNE_MIX_02` | `full_finetune_460_run_16` | 最新，报告所用，**推荐使用** |
| `run_15/`（备用） | `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD_FULL_FINETUNE_MIX` | `full_finetune_460_run_15` | 早期版本，保留备用 |

> ⚠️ 两次运行权重**不同**，请勿混用。默认使用 `run_16/`。

---

## 4. 用途

### 4.1 AWBC 训练
价值模型预测每帧优势，经 `stage_advantage/annotation/gt_label.py` 离散化为
正/负标签写入 `task_index` / `tasks.jsonl`，再用于 Advantage-Weighted Behavior Cloning 训练
（`prompt_from_task=True`）。

### 4.2 阶段分割
`absolute_value`（0~1 累计进度）可把任务切成 K 个阶段，用于 subtask 级别的
成功判定与评估（见 `docs/results.md` § 4）。

---

## 5. 使用方式

推理侧将 `ckpt_dir` 指向本权重目录、`ckpt_steps=20000`（编辑
`stage_advantage/annotation/eval.py` 的 `MODELS_CONFIG_MAP`），然后：

```bash
uv run python stage_advantage/annotation/eval.py Task-A KAI0 /path/to/dataset
```

---

## 6. 数据 / 评估

- 训练数据：`FlattenFold`（含 `stage_progress_gt` 标注）及其 augmentation / merge 变体。
- 阶段分割评估（Fold Box，K=16，100 episodes，seed=42）：
  - Frame Accuracy **51.0%**（±12.2%）
  - Stage IoU **48.1%**
  - Boundary MAE **60.1 帧**（~2.0 s）
  - Boundary Acc @±30f **66.8%** / @±50f **77.7%**

---

## 7. 出处与许可

- 项目：**χ₀ (kai0)** —— 双臂衣物操作模仿学习框架（基于 [openpi](https://github.com/Physical-Intelligence/openpi)）
- 上游许可：Apache-2.0（本项目）；PaliGemma 相关组件见 `LICENSE_GEMMA.txt`（Gemma Terms of Use）
- 关联资源：
  - 代码：本仓库
  - 数据集 / 策略模型：`OpenDriveLab-org/Kai0`（HF & ModelScope）
