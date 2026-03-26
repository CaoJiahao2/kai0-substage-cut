import json
import time
import traceback

print("step=import_start", flush=True)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print("step=import_ok", flush=True)

repo_id = "OpenDriveLab-org/Kai0"
root = "/magicflow/chenxuhua/Data/Kai0_newdataset/Task_A/base_dagger_merge"

print("step=init_start", flush=True)
t0 = time.time()
ds = LeRobotDataset(repo_id=repo_id, root=root,video_backend="pyav")
print(f"step=init_ok seconds={time.time() - t0:.2f}", flush=True)
print(f"len={len(ds)}", flush=True)

def try_get(idx, name):
    print(f"step={name}_start idx={idx}", flush=True)
    t = time.time()
    try:
        sample = ds[idx]
        out = {
            "ok": True,
            "seconds": round(time.time() - t, 2),
            "keys": sorted(sample.keys()),
            "action_shape": list(sample["action"].shape) if hasattr(sample["action"], "shape") else None,
            "state_shape": list(sample["observation.state"].shape) if hasattr(sample["observation.state"], "shape")
else None,
        }
        print(json.dumps({name: out}, ensure_ascii=False, indent=2), flush=True)
    except Exception as e:
        out = {
            "ok": False,
            "seconds": round(time.time() - t, 2),
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps({name: out}, ensure_ascii=False, indent=2), flush=True)

try_get(0, "getitem_first")
try_get(len(ds) - 1, "getitem_last")