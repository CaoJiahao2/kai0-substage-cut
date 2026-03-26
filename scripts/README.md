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

step1 运行可视化服务

```shell
uv run python scripts/infer_viewer_server.py --host 0.0.0.0 --port 9001
```

step2 浏览器访问

loss地址: http://127.0.0.1:9001/?mode=live

3d 可视化: http://127.0.0.1:9001/ee_viewer/index.html



step3 运行模型推理

```shell
uv run python scripts/infer_dataset_loss_viz.py \
  --repo-id data/Task_B/base \
  --dataset-episodes 0,1 \
  --local-policy-config pi05_flatten_fold_normal \
  --local-policy-dir model/Task_B/mixed_9 \
  --viz none \
  --web-viewer-url http://127.0.0.1:9001 \
  --web-reset-first \
  --web-log-images
```


