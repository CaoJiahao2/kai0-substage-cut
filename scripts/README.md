## 可视化数据代码

**step1**: 

git clone https://github.com/agilexrobotics/mobile_aloha_sim_ros2.git


**step2**:
修改 rerun_dataset_viz.py 中的变量 DEFAULT_URDF_ROOT 为上面clone 的仓库的路径

```shell
python scripts/rerun_dataset_viz.py \
  --repo-id "/path/to/Task_A/base" \
  --episode-index 0 \
  --max-frames 1500 \
  --show-ee-traj
```

## 可视化代码

```shell
python scripts/infer_dataset_loss_viz.py \
  --repo-id data/Task_B/base \
  --host 127.0.0.1 \
  --port 8000 \
  --viz rerun \
  --save-csv outputs/loss_eval/task_b_loss.csv \
  --save-plot outputs/loss_eval/task_b_loss.png \
  --log-images
```
