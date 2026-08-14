# 实验结论（Results）

> 本文汇总 kai0 的关键实验结论：主实验结果、训练资源、数据集规模、阶段分割评估
> 与价值模型 checkpoint 说明。原始报告见 `outputs/` 目录。

---

## 目录

- [1. 主实验结果](#1-主实验结果)
- [2. 训练资源与数据规模](#2-训练资源与数据规模)
- [3. 价值模型 checkpoint 说明](#3-价值模型-checkpoint-说明)
- [4. 阶段分割评估](#4-阶段分割评估)
- [5. 结论与局限](#5-结论与局限)
- [6. 引用](#6-引用)

---

## 1. 主实验结果

| 指标 | 结果 |
| --- | --- |
| 成功率提升 | 相对 π₀.₅ 基线提升约 **250%** |
| 数据量 | 约 **20 小时**衣物操作数据（base + DAgger 精选） |
| 算力 | **8×A100** 训练 |
| 任务 | 展平（Flatten）、折叠（Fold）、悬挂（Hang）双端衣物操作 |

**核心思想**：通过「模型算术（Model Arithmetic）」「阶段优势（Stage Advantage）」「训练-部署对齐
（Train-Deploy Alignment）」三大模块，分别抑制**训练、归纳偏置与部署**三类分布不一致，
从而用极低的数据与算力达到生产级鲁棒性。

---

## 2. 训练资源与数据规模

### 2.1 数据集规模（LeRobot v2.1）

来自 `docs/dataset.md`（发布数据）：

| 任务 | Base（episodes / 小时） | DAgger（episodes / 小时） | 合计（episodes / 小时） |
| --- | --- | --- | --- |
| FlattenFold | 3,055 / ~42h | 3,457 / ~13h | 6,512 / ~55h |
| HangCloth | 6,954 / ~61h | 686 / ~12h | 7,640 / ~73h |
| TeeShirtSort | 5,988 / ~31h | 769 / ~22h | 6,757 / ~53h |
| **合计** | **15,997 / ~134h** | **4,912 / ~47h** | **20,909 / ~181h** |

> 论文主实验使用的是精选后的 **~20 小时**子集 + 8×A100 全参微调（如 `pi05_flatten_fold_normal_follow_paper`
> 配置，30k 步、batch 128、lr 2.5e-5）。

### 2.2 训练配置要点

| 项 | 值 |
| --- | --- |
| 全参微调 | `pi05_flatten_fold_normal_follow_paper`：30k 步、batch 128、peak_lr 2.5e-5、cosine 退火 |
| LoRA | `pi05_flatten_fold_normal_follow_paper_lora`：冻结 LLM，显存 < 22.5GB |
| 价值模型 | `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD`：100k 步、batch 16/卡、save 10k |

---

## 3. 价值模型 checkpoint 说明

仓库内保存了两份已训练的 **Advantage Estimator**（价值模型）20000 步 checkpoint：

| 目录 | 配置（metadata） | exp_name | 时间戳 | 说明 |
| --- | --- | --- | --- | --- |
| `value_model/20000/` | `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD_FULL_FINETUNE_MIX_02` | `full_finetune_460_run_16` | 较新 | **推荐使用**（报告所用） |
| `value_model_version2/20000/` | `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD_FULL_FINETUNE_MIX` | `full_finetune_460_run_15` | 较早 | 早期版本，保留备用 |

- 两个目录均含 `model.safetensors`（7.5GB）+ `optimizer.pt`（13.5GB）+ `metadata.pt`。
- 二者为**同一配置族的两次不同运行**（run_15 vs run_16），模型权重不同，请勿混用。
- `ckpt_local/value_model/20000/` 仅为 `value_model/20000` 的**模型权重副本**（无 optimizer），
  用于在慢速挂载上加速加载；同一文件系统下可替换为软链接（见 README「目录结构」）。
- 推理用法见 `docs/inference_deployment.md` § 7 与 [README § 10（推理与部署）](../README.md#10-推理与部署)。

---

## 4. 阶段分割评估

价值模型输出的 `absolute_value`（0~1 累计进度）可进一步做**阶段分割**（把任务切成 K 个阶段）。
以下为两轮评估（`outputs/eval_stage_seg_results/` 与 `outputs/eval_叠衣服_round1/`）：

### 4.1 折纸箱（Fold Box，K=16，100 episodes，seed=42）

| 指标 | 值 |
| --- | --- |
| Frame Accuracy（mean） | **51.0%**（std ±12.2%，median 51.1%） |
| Stage IoU（overall mean） | **48.1%** |
| Boundary MAE | **60.1 帧**（~2.0 秒） |
| Boundary AUC（1–50 帧） | **0.549** |
| Boundary Acc @±30f | **66.8%** |
| Boundary Acc @±50f | **77.7%** |
| Episode Full Overlap @≥0.5 | 0/100（受首尾阶段拖累） |

**逐阶段 IoU 要点**：

- 中间阶段效果好：阶段 4 / 7 / 9 / 12 的 IoU 均 > 72%（模型对塑形、折叠等核心动作识别准确）；
- 首尾阶段差：阶段 0（2.4%）、1（6.7%）、14（9.9%）、15（0.0%）——**等宽分段对边界处理不准确**
  是主因，而非模型本身。

### 4.2 叠衣服（K=3~5，70 episodes，seed=42）

| 指标 | 值 |
| --- | --- |
| Frame Accuracy（mean） | **18.9%**（std ±17.1%） |
| Stage IoU（mean） | **6.2%** |
| Boundary MAE | **571.4 帧**（~19 秒） |
| Boundary AUC（0-50 帧） | **0.001** |

> 叠衣服任务各子动作时长差异大，等宽分段（`equal_width`）严重失效；K=3 的 episodes
> 表现相对较好（50.7%~87.1%），说明**阶段越少误差累积越小**。

### 4.3 结论与改进方向

1. **KAI0 价值模型对细粒度阶段（K=16）的进度预测能力良好**，帧级准确率 51.0% 显著优于
   叠衣服任务（18.9%）；边界定位在 ±30/±50 帧容差下分别达 66.8% / 77.7%。
2. **瓶颈在分割算法而非模型**：等宽分段（`equal_width`）无法应对时长不均的阶段，导致首尾
   阶段 IoU≈0 且 Full Overlap 恒为 0。
3. **改进方向**（报告建议）：排除首尾非动作帧、等分位数分段（`equal_count`）、Viterbi 时序
   平滑、边界微调、按 episode 自身 `absolute_value` 范围归一化。

---

## 5. 结论与局限

**积极结论**

- 以极低数据量（~20h）与算力（8×A100）实现超越 π₀.₅ 基线约 **250%** 的成功率。
- 价值模型（Advantage Estimator）无需人工标注即可为任意轨迹预测阶段进度，支持 AWBC 与
  阶段分割两大下游任务。
- 模型算术在权重空间融合多子集 checkpoint，避免 MoE 的部署开销，训练-部署对齐模块
  （增广 / DAgger / RTC / 时序平滑）显著降低实机抖振。

**已知局限**

- 阶段分割采用简单等宽分段，首尾边界误差大、Full Overlap 为 0（见 § 4.3 改进方向）。
- 价值模型批量推理较慢（A800-40GB 上约 11 分钟/episode），可通过增大 batch 提速。
- `data_infer` 为 TOS 只读挂载，报告中的 `/kai0/...` 路径为 Docker 内路径，与当前仓库布局不同。

---

## 6. 引用

```bibtex
@article{sima2026kai0,
  title={$\chi_{0}$: Resource-Aware Robust Manipulation via Taming Distributional Inconsistencies},
  author={Yu, Checheng and Sima, Chonghao and Jiang, Gangcheng and Zhang, Hai and Mai, Haoguang
          and Li, Hongyang and Wang, Huijie and Chen, Jin and Wu, Kaiyang and Chen, Li
          and Zhao, Lirui and Shi, Modi and Luo, Ping and Bu, Qingwen and Peng, Shijia
          and Li, Tianyu and Yuan, Yibo},
  journal={arXiv preprint arXiv:2602.09021},
  year={2026}
}
```

- 博客：https://mmlab.hk/research/kai0
- 数据集 / 模型：https://huggingface.co/OpenDriveLab-org/Kai0 、https://www.modelscope.cn/models/OpenDriveLab/Kai0
