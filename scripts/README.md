## Scripts Index

现在 `scripts/` 已经按用途做了物理拆分：

- 训练脚本保留在 `scripts/` 根目录
- Docker 相关保留在 `scripts/docker/`
- `scripts/infer/`
- `scripts/viz/`
- `scripts/tools/`

使用时请直接走现在的真实路径：

- 训练：`scripts/*.py`
- 推理：`scripts/infer/*`
- 可视化：`scripts/viz/*`
- 工具：`scripts/tools/*`

---

## 1. 训练

### 训练入口

- `train.py`
  - JAX 训练入口。
- `train_pytorch.py`
  - PyTorch / DDP 训练入口。
- `train_test.py`
  - 训练脚本相关测试。

### 训练前检查 / 统计

- `compute_norm_stats.py`
  - 标准版归一化统计计算。
- `compute_norm_states_fast.py`
  - 快速版归一化统计计算。
- `scan_train_dataloader.py`
  - 按真实训练 dataloader 扫描数据，检查 decode / transform / batch。

### 训练评估

- `eval_train_l1_jax.py`
  - 用 JAX checkpoint 在训练数据上评估动作 L1。
- `eval_train_l1_pytorch.py`
  - 用 PyTorch checkpoint 在训练数据上评估动作 L1。

---

## 2. 推理

### 推理服务

- `serve_policy.py`
  - 启动 policy websocket 服务，给远端或本地推理调用。

### 数据集推理 / checkpoint 评估

- `infer_from_dataset_viewer.py`
  - 从数据集取样做推理，并把结果推到 web viewer。
- `eval_local_episode_metrics.py`
  - 在本地数据集 episode 上评估 checkpoint。
- `eval_task_a_v2_manual.py`
  - Task_A v2 的手工评估脚本。

---

## 3. 可视化

### GT / 数据集回放

- `rerun_dataset_viz.py`
  - 用 Rerun 回放 GT 数据集，支持图像、关节、URDF/FK 轨迹。
- `view_gt_replay.sh`
  - GT 回放的轻量启动脚本，直接调用 `rerun_dataset_viz.py`。

### 推理结果可视化

- `infer_dataset_loss_viz.py`
  - 跑 dataset inference，并可视化 loss / ee error / 图像。
- `infer_dataset_loss_viz.sh`
  - 上面脚本的 shell 启动器。
- `infer_viewer_server.py`
  - 浏览器 viewer 的后端服务。
- `infer_viewer_replay.py`
  - 回放 viewer 记录的 JSONL 事件。

### 前端 viewer 资源

- `infer_viewer/`
  - 浏览器可视化前端。
  - `index.html`: 主 viewer 页面
  - `app.js`: 主前端逻辑
  - `styles.css`: 样式
  - `ee_viewer/`: 3D 末端轨迹 viewer
  - `vendor/`: three.js / URDFLoader 等依赖

### 常见可视化命令

1. GT 回放

```bash
bash scripts/viz/view_gt_replay.sh data/Task_B/base 0
```

2. 启动 web viewer

```bash
uv run python scripts/viz/infer_viewer_server.py --host 0.0.0.0 --port 9001
```

3. 浏览器访问

- Loss viewer: `http://127.0.0.1:9001/?mode=live`
- 3D viewer: `http://127.0.0.1:9001/ee_viewer/index.html`

4. 推理 + 可视化

```bash
uv run python scripts/viz/infer_dataset_loss_viz.py \
  --repo-id data/Task_B/base \
  --dataset-episodes 0,1 \
  --local-policy-config pi05_flatten_fold_normal \
  --local-policy-dir model/Task_B/mixed_9 \
  --viz none \
  --web-viewer-url http://127.0.0.1:9001 \
  --web-reset-first \
  --web-log-images
```

---

## 4. 其他工具

### 数据集下载 / 转换 / 清洗

- `download_dataset.py`
  - 下载 Kai0 数据集。
- `download_checkpoints.py`
  - 下载公开 checkpoint。
- `convert_dataset_v30_to_v21.py`
  - LeRobot v3 布局转 v2.1。
- `repair_lerobot_videos.py`
  - 检查并修复损坏的视频。
- `remove_bad_episodes.py`
  - 按 episode 清洗数据集，生成新数据集。

### 数据集拆分 / 合并

- `split_lerobot.py`
  - 按 episode 拆分 LeRobot 数据集。
- `split_lerobot_dataset.py`
  - 将数据集拆成多个固定子集。
- `merge_lerobot.py`
  - 较通用的 LeRobot 合并脚本。
- `merge_lerobot_datasets.py`
  - 多数据集按 episode 合并，支持按比例近似混合。

### 环境 / 部署 / 同步

- `docker/`
  - Docker 相关脚本和配置，保留在 `scripts/` 根目录。
  - `compose.yml`
  - `serve_policy.Dockerfile`
  - `install_docker_ubuntu22.sh`
  - `install_nvidia_container_toolkit.sh`
  - `docker_README.md`
- `sync_folders.sh`
  - 文件夹同步工具。

### 备注

- `__init__.py`
  - 供 `scripts` 作为 Python 模块导入。
- `__pycache__/`
  - Python 缓存文件，可忽略。
