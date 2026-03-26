#!/usr/bin/env python3
"""Load a LeRobot dataset, run policy inference, and push live visualization payloads.

Example:
  uv run python scripts/infer_from_dataset_viewer.py \
    --dataset_repo_id /home/magiclab/kai0/data/Task_A/base \
    --host 127.0.0.1 \
    --port 8000 \
    --num_samples 200 \
    --auto_launch_viewer
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
try:
    import torch
except Exception:  # pragma: no cover - runtime env dependent
    torch = None  # type: ignore[assignment]

_OPENPI_CLIENT_IMPORT_ERROR: Exception | None = None
try:
    from openpi_client import websocket_client_policy
except Exception as exc:  # pragma: no cover - runtime env dependent
    websocket_client_policy = None  # type: ignore[assignment]
    _OPENPI_CLIENT_IMPORT_ERROR = exc

_LEROBOT_IMPORT_ERROR: Exception | None = None
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except Exception as exc:  # pragma: no cover - runtime env dependent
    LeRobotDataset = Any  # type: ignore[assignment,misc]
    _LEROBOT_IMPORT_ERROR = exc

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
except Exception:  # pragma: no cover - depends on lerobot version
    LeRobotDatasetMetadata = None

try:
    from lerobot.model.kinematics import RobotKinematics
except Exception:  # pragma: no cover - optional dependency (placo)
    RobotKinematics = None

try:
    from PIL import Image

    PIL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PIL_AVAILABLE = False


DEFAULT_URDF_ROOT = Path("/home/magiclab/projects/mobile_aloha_sim_ros2")
DEFAULT_URDF_PATH = DEFAULT_URDF_ROOT / "aloha_new_description/urdf/aloha_new.urdf"
DEFAULT_VIEWER_SERVER_SCRIPT = Path("/home/magiclab/projects/lerobot/tools/infer_viewer_server.py")
DEFAULT_VIEWER_STATIC_DIR = Path("/home/magiclab/projects/lerobot/tools/infer_viewer")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run websocket policy inference on local dataset samples and push to infer_viewer."
    )
    parser.add_argument("--dataset_repo_id", required=True, help="LeRobot dataset repo_id or local dataset path.")
    parser.add_argument("--dataset_root", default=None, help="Optional dataset root.")
    parser.add_argument(
        "--dataset_episodes",
        default=None,
        help="Episode subset. Accepts JSON list (e.g. '[0,1]') or comma/range syntax (e.g. '0,1,3-8').",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Websocket policy server host.")
    parser.add_argument("--port", type=int, default=8000, help="Websocket policy server port.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=200, help="<=0 means full dataset.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames_per_episode", type=int, default=0)
    parser.add_argument("--fps", type=float, default=0.0, help="If >0, sleep to keep this loop rate.")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--state_key", default="observation.state")
    parser.add_argument("--action_key", default="action")
    parser.add_argument(
        "--camera_map",
        default="top_head=observation.images.top_head,hand_left=observation.images.hand_left,hand_right=observation.images.hand_right",
        help="Policy camera key to dataset key mapping, comma-separated.",
    )
    parser.add_argument("--prompt", default=None, help="Override prompt. If omitted, use sample task when present.")
    parser.add_argument("--target_action_index", type=int, default=0, help="GT action offset for error compute.")
    parser.add_argument("--push_every", type=int, default=1, help="Push every N frames.")
    parser.add_argument("--push_timeout_s", type=float, default=1.0)
    parser.add_argument("--viewer_host", default="127.0.0.1")
    parser.add_argument("--viewer_port", type=int, default=9001)
    parser.add_argument("--push_url", default=None, help="infer_viewer push endpoint. Auto-built if omitted.")
    parser.add_argument("--viewer_max_frames", type=int, default=2000)
    parser.add_argument(
        "--viewer_server_script",
        default=str(DEFAULT_VIEWER_SERVER_SCRIPT),
        help="Path to infer_viewer_server.py (used when --auto_launch_viewer).",
    )
    parser.add_argument(
        "--viewer_static_dir",
        default=str(DEFAULT_VIEWER_STATIC_DIR),
        help="infer_viewer static directory (used when --auto_launch_viewer).",
    )
    parser.add_argument("--auto_launch_viewer", action="store_true")
    parser.add_argument("--viewer_startup_wait_s", type=float, default=1.5)
    parser.add_argument(
        "--push_trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Push trajectory arrays to viewer.",
    )
    parser.add_argument("--traj_indices", default="0,1", help="Fallback 2D trajectory dimensions.")
    parser.add_argument("--traj_max_len", type=int, default=4000)
    parser.add_argument("--urdf_path", default=str(DEFAULT_URDF_PATH), help="URDF path for FK trajectory.")
    parser.add_argument("--urdf_root", default=str(DEFAULT_URDF_ROOT), help="Root for package:// URDF resources.")
    parser.add_argument("--ee_links", default="fl_link8,fr_link8", help="Comma-separated end-effector link names.")
    parser.add_argument("--joint_unit", choices=["rad", "deg"], default="rad")
    parser.add_argument("--save_path", default=None, help="Optional JSONL output path.")
    return parser.parse_args()


def _parse_dataset_episodes(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    values: list[int] = []
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("--dataset_episodes JSON must be a list.")
        values = [int(v) for v in parsed]
    else:
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                a_str, b_str = token.split("-", 1)
                a = int(a_str)
                b = int(b_str)
                if b < a:
                    raise ValueError(f"Invalid range: {token}")
                values.extend(range(a, b + 1))
            else:
                values.append(int(token))
    out = sorted(set(values))
    if any(v < 0 for v in out):
        raise ValueError("Episode indices must be non-negative.")
    return out


def _parse_camera_map(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid camera mapping '{item}'. Expected key=value.")
        dst, src = item.split("=", 1)
        dst = dst.strip()
        src = src.strip()
        if not dst or not src:
            raise ValueError(f"Invalid camera mapping '{item}'.")
        result[dst] = src
    if not result:
        raise ValueError("--camera_map is empty.")
    return result


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_scalar(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value
    return value


def _select_last_frame(arr: np.ndarray) -> np.ndarray:
    if arr.ndim <= 1:
        return arr
    if arr.shape[0] > 1:
        return arr[-1]
    return arr


def _to_uint8_chw(image: Any) -> np.ndarray:
    img = _to_numpy(image)
    if img.ndim == 4:
        img = img[-1]
    if img.ndim != 3:
        raise ValueError(f"Image must have ndim=3, got {img.shape}.")
    if img.shape[-1] in (1, 3, 4) and img.shape[0] not in (1, 3, 4):
        img = np.transpose(img, (2, 0, 1))
    if np.issubdtype(img.dtype, np.floating):
        scale = 255.0 if float(np.max(img)) <= 1.5 else 1.0
        img = np.clip(img * scale, 0, 255).round().astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _to_uint8_hwc(image: Any) -> np.ndarray:
    chw = _to_uint8_chw(image)
    if chw.shape[0] in (1, 3, 4):
        return np.transpose(chw, (1, 2, 0))
    return chw


def _encode_image_data_url(image: Any) -> str | None:
    if not PIL_AVAILABLE:
        return None
    try:
        img = _to_uint8_hwc(image)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=2)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def _extract_optional_int(sample: dict[str, Any], key: str) -> int | None:
    if key not in sample:
        return None
    value = _to_scalar(sample[key])
    try:
        return int(value)
    except Exception:
        return None


def _extract_timestamp(sample: dict[str, Any]) -> float:
    if "timestamp" in sample:
        try:
            return float(_to_scalar(sample["timestamp"]))
        except Exception:
            return time.time()
    return time.time()


def _extract_text(value: Any) -> str | None:
    value = _to_scalar(value)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S", "O"} and value.size == 1:
        item = value.item()
        return _extract_text(item)
    text = str(value).strip()
    return text if text else None


def _flatten_feature_names(names: Any) -> list[str]:
    if names is None:
        return []
    if isinstance(names, dict):
        out: list[str] = []
        for v in names.values():
            out.extend(_flatten_feature_names(v))
        return out
    if isinstance(names, (list, tuple)):
        out = []
        for v in names:
            out.extend(_flatten_feature_names(v))
        return out
    return [str(names)]


def _extract_feature_names(features: dict[str, Any], key: str) -> list[str]:
    if key not in features or not isinstance(features[key], dict):
        return []
    return _flatten_feature_names(features[key].get("names"))


def _extract_action_sequence(action: Any) -> list[list[float]] | None:
    if action is None:
        return None
    arr = _to_numpy(action)
    if arr.ndim == 0:
        return [[float(arr)]]
    if arr.ndim == 1:
        return [[float(v) for v in arr.reshape(-1)]]
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim == 1:
        return [[float(v) for v in arr.reshape(-1)]]
    return [[float(v) for v in row.reshape(-1)] for row in arr]


def _collect_target_action_steps(
    dataset: LeRobotDataset,
    current_idx: int,
    current_sample: dict[str, Any],
    action_key: str,
    start_step: int,
    max_steps: int,
) -> list[list[float]]:
    if max_steps <= 0 or action_key not in current_sample:
        return []
    steps: list[list[float]] = []
    seq = _extract_action_sequence(current_sample.get(action_key)) or []
    base_len = len(seq)
    cur_episode = _extract_optional_int(current_sample, "episode_index")
    cur_frame = _extract_optional_int(current_sample, "frame_index")
    for rel in range(max_steps):
        offset = max(0, start_step) + rel
        if offset < base_len:
            steps.append(seq[offset])
            continue
        ds_idx = current_idx + (offset - base_len + 1)
        if ds_idx >= len(dataset):
            break
        nxt = dataset[ds_idx]
        if cur_episode is not None:
            nxt_episode = _extract_optional_int(nxt, "episode_index")
            if nxt_episode is None or nxt_episode != cur_episode:
                break
        if cur_frame is not None:
            nxt_frame = _extract_optional_int(nxt, "frame_index")
            expected = cur_frame + (offset - base_len + 1)
            if nxt_frame is not None and nxt_frame != expected:
                break
        nxt_seq = _extract_action_sequence(nxt.get(action_key)) if action_key in nxt else None
        if not nxt_seq:
            break
        steps.append(nxt_seq[0])
    return steps


def _compute_l1(pred: list[float] | None, target: list[float] | None) -> float | None:
    if pred is None or target is None or len(pred) != len(target):
        return None
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(target))))


def _infer_side_from_name(name: str | None) -> str | None:
    if not name:
        return None
    text = str(name).lower()
    if "right" in text or text.startswith("r_") or text.endswith("_r"):
        return "right"
    if "left" in text or text.startswith("l_") or text.endswith("_l"):
        return "left"
    if text.startswith("fr_") or text.startswith("br_"):
        return "right"
    if text.startswith("fl_") or text.startswith("bl_"):
        return "left"
    return None


def _split_indices_by_side(names: list[str]) -> tuple[list[int], list[int]]:
    left: list[int] = []
    right: list[int] = []
    for idx, name in enumerate(names):
        side = _infer_side_from_name(name)
        if side == "left":
            left.append(idx)
        elif side == "right":
            right.append(idx)
    if not left and not right and len(names) >= 14:
        left = list(range(0, 7))
        right = list(range(7, 14))
    return left, right


def _compute_l1_on_indices(pred: list[float] | None, target: list[float] | None, indices: list[int]) -> float | None:
    if pred is None or target is None or len(pred) != len(target) or not indices:
        return None
    vals = [abs(float(pred[i]) - float(target[i])) for i in indices if 0 <= i < len(pred)]
    if not vals:
        return None
    return float(np.mean(vals))


def _compute_l2_distance(vec_a: list[float] | None, vec_b: list[float] | None) -> float | None:
    if vec_a is None or vec_b is None or len(vec_a) != len(vec_b):
        return None
    return float(np.linalg.norm(np.asarray(vec_a) - np.asarray(vec_b)))


def _mean_errors_by_side(errors_by_name: dict[str, float]) -> tuple[float | None, float | None]:
    left_vals: list[float] = []
    right_vals: list[float] = []
    for name, val in errors_by_name.items():
        side = _infer_side_from_name(name)
        if side == "left":
            left_vals.append(float(val))
        elif side == "right":
            right_vals.append(float(val))
    left_mean = float(np.mean(left_vals)) if left_vals else None
    right_mean = float(np.mean(right_vals)) if right_vals else None
    return left_mean, right_mean


def _resolve_resource_path(resource: str, urdf_path: Path, urdf_root: Path | None) -> tuple[Path | None, str]:
    if resource.startswith("package://"):
        parts = resource[len("package://") :].split("/", 1)
        if len(parts) != 2:
            return None, resource
        package, rel_path = parts
        candidates = [parent / package / rel_path for parent in urdf_path.parents]
        if urdf_root is not None:
            candidates.append(urdf_root / package / rel_path)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve(), resource
        return None, resource
    if resource.startswith("file://"):
        candidate = Path(resource[len("file://") :]).expanduser()
        return candidate if candidate.exists() else None, resource
    candidate = Path(resource)
    if not candidate.is_absolute():
        candidate = (urdf_path.parent / candidate).resolve()
    return candidate if candidate.exists() else None, resource


def _rewrite_urdf_for_fk(urdf_path: Path, urdf_root: Path | None, out_dir: Path) -> Path:
    text = urdf_path.read_text(encoding="utf-8")
    rewritten = text
    for match in set(re.findall(r'filename="([^"]+)"', text)):
        src, _ = _resolve_resource_path(match, urdf_path, urdf_root)
        if src is not None:
            rewritten = rewritten.replace(match, str(src))
    out_path = out_dir / "robot_fk.urdf"
    out_path.write_text(rewritten, encoding="utf-8")
    return out_path


def _build_urdf_to_dataset_map(dataset_joint_names: list[str], urdf_joint_names: list[str]) -> list[int | None]:
    mapping: list[int | None] = [None] * len(urdf_joint_names)
    if not urdf_joint_names:
        return mapping

    if dataset_joint_names:
        for ds_idx, name in enumerate(dataset_joint_names):
            if name in urdf_joint_names:
                mapping[urdf_joint_names.index(name)] = ds_idx
        if any(v is not None for v in mapping):
            return mapping

    # Heuristic for mobile-aloha two-arm datasets: first 14 dimensions map to fl/fr joints.
    fallback_names = [f"fl_joint{i}" for i in range(1, 8)] + [f"fr_joint{i}" for i in range(1, 8)]
    if len(dataset_joint_names) >= 14 and all(name in urdf_joint_names for name in fallback_names):
        for ds_idx, urdf_name in enumerate(fallback_names):
            urdf_idx = urdf_joint_names.index(urdf_name)
            mapping[urdf_idx] = ds_idx
        return mapping

    # Heuristic for explicit left/right naming (e.g. left_arm_joint1 ... right_arm_joint7).
    left_pat = re.compile(r"left[_-]?arm[_-]?joint([1-7])")
    right_pat = re.compile(r"right[_-]?arm[_-]?joint([1-7])")
    for ds_idx, name in enumerate(dataset_joint_names):
        lower = name.lower()
        m_left = left_pat.fullmatch(lower)
        m_right = right_pat.fullmatch(lower)
        if m_left:
            urdf_name = f"fl_joint{int(m_left.group(1))}"
        elif m_right:
            urdf_name = f"fr_joint{int(m_right.group(1))}"
        else:
            continue
        if urdf_name in urdf_joint_names:
            mapping[urdf_joint_names.index(urdf_name)] = ds_idx
    return mapping


def _vector_to_fk_point(
    vec: list[float] | None,
    solver: Any,
    urdf_to_dataset: list[int | None],
    joint_unit: str,
) -> list[float] | None:
    if vec is None:
        return None
    full = np.zeros(len(urdf_to_dataset), dtype=float)
    for urdf_idx, ds_idx in enumerate(urdf_to_dataset):
        if ds_idx is None or ds_idx >= len(vec):
            continue
        full[urdf_idx] = float(vec[ds_idx])
    if joint_unit == "rad":
        full = np.rad2deg(full)
    fk = solver.forward_kinematics(full)
    return fk[:3, 3].tolist()


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        resp.read()


def _launch_viewer_server(args: argparse.Namespace) -> subprocess.Popen[str] | None:
    script = Path(args.viewer_server_script).expanduser().resolve()
    static_dir = Path(args.viewer_static_dir).expanduser().resolve()
    if not script.exists():
        logging.warning("Viewer server script does not exist: %s", script)
        return None
    if not static_dir.exists():
        logging.warning("Viewer static dir does not exist: %s", static_dir)
        return None
    cmd = [
        sys.executable,
        str(script),
        "--host",
        args.viewer_host,
        "--port",
        str(args.viewer_port),
        "--static_dir",
        str(static_dir),
        "--max_frames",
        str(args.viewer_max_frames),
    ]
    if args.urdf_path:
        cmd.extend(["--urdf_path", str(Path(args.urdf_path).expanduser())])
    if args.urdf_root:
        cmd.extend(["--urdf_root", str(Path(args.urdf_root).expanduser())])
    proc = subprocess.Popen(cmd)
    time.sleep(max(0.0, args.viewer_startup_wait_s))
    return proc


def _safe_prompt(sample: dict[str, Any], prompt_override: str | None, tasks_map: dict[int, str]) -> str | None:
    if prompt_override:
        return prompt_override
    for key in ("task", "prompt", "instruction", "language_instruction"):
        if key in sample:
            text = _extract_text(sample[key])
            if text:
                return text
    task_index = _extract_optional_int(sample, "task_index")
    if task_index is not None and task_index in tasks_map:
        return tasks_map[task_index]
    return None


def _load_features(dataset_repo_id: str, dataset_root: str | None, dataset: LeRobotDataset) -> dict[str, Any]:
    if LeRobotDatasetMetadata is not None:
        try:
            return dict(LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root).features)
        except Exception:
            pass
    meta = getattr(dataset, "meta", None)
    features = getattr(meta, "features", None)
    if isinstance(features, dict):
        return dict(features)
    return {}


def _load_tasks_map(dataset: LeRobotDataset) -> dict[int, str]:
    meta = getattr(dataset, "meta", None)
    tasks = getattr(meta, "tasks", None)
    if not isinstance(tasks, dict):
        return {}
    out: dict[int, str] = {}
    for key, value in tasks.items():
        try:
            idx = int(key)
        except Exception:
            continue
        text = _extract_text(value)
        if text:
            out[idx] = text
    return out


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if websocket_client_policy is None:
        raise RuntimeError(
            "openpi_client import failed. Install project runtime deps first."
        ) from _OPENPI_CLIENT_IMPORT_ERROR
    if _LEROBOT_IMPORT_ERROR is not None:
        raise RuntimeError(
            "lerobot import failed. Install project runtime deps first."
        ) from _LEROBOT_IMPORT_ERROR

    selected_episodes = _parse_dataset_episodes(args.dataset_episodes)
    camera_map = _parse_camera_map(args.camera_map)
    push_url = args.push_url or f"http://{args.viewer_host}:{args.viewer_port}/api/push"
    traj_indices = [int(x) for x in args.traj_indices.split(",") if x.strip().isdigit()]
    if len(traj_indices) < 2:
        traj_indices = [0, 1]
    ee_links = [s.strip() for s in (args.ee_links or "").split(",") if s.strip()]

    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, episodes=selected_episodes)
    features = _load_features(args.dataset_repo_id, args.dataset_root, dataset)
    tasks_map = _load_tasks_map(dataset)
    state_names = _extract_feature_names(features, args.state_key)
    action_names = _extract_feature_names(features, args.action_key)
    left_indices, right_indices = _split_indices_by_side(action_names or state_names)

    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info("Connected to websocket policy server %s:%s", args.host, args.port)
    logging.info("Server metadata: %s", policy.get_server_metadata())

    viewer_proc: subprocess.Popen[str] | None = None
    if args.auto_launch_viewer:
        viewer_proc = _launch_viewer_server(args)
        if viewer_proc is not None:
            logging.info("Viewer server started at http://%s:%s/?mode=live", args.viewer_host, args.viewer_port)

    fk_tempdir: tempfile.TemporaryDirectory[str] | None = None
    fk_solvers: dict[str, Any] | None = None
    fk_maps: dict[str, list[int | None]] | None = None
    traj_mode = "2d"
    if args.push_trajectory and args.urdf_path and ee_links:
        if RobotKinematics is None:
            logging.warning("RobotKinematics unavailable (placo not installed), fallback to 2D trajectory.")
        else:
            urdf_path = Path(args.urdf_path).expanduser().resolve()
            urdf_root = Path(args.urdf_root).expanduser().resolve() if args.urdf_root else None
            if not urdf_path.exists():
                logging.warning("URDF not found: %s. fallback to 2D trajectory.", urdf_path)
            else:
                fk_tempdir = tempfile.TemporaryDirectory(prefix="openpi_fk_")
                fk_path = _rewrite_urdf_for_fk(urdf_path, urdf_root, Path(fk_tempdir.name))
                fk_solvers = {}
                fk_maps = {}
                dataset_joint_names = action_names or state_names
                for link_name in ee_links:
                    solver = RobotKinematics(str(fk_path), target_frame_name=link_name, joint_names=None)
                    fk_map = _build_urdf_to_dataset_map(dataset_joint_names, solver.joint_names)
                    if not any(idx is not None for idx in fk_map):
                        logging.warning("FK map is empty for link %s, skip.", link_name)
                        continue
                    fk_solvers[link_name] = solver
                    fk_maps[link_name] = fk_map
                if fk_solvers:
                    traj_mode = "fk3d"
                    logging.info("Enabled FK trajectory mode with links: %s", ", ".join(fk_solvers.keys()))
                else:
                    fk_solvers = None
                    fk_maps = None
                    logging.warning("FK initialization failed, fallback to 2D trajectory.")

    if args.num_samples <= 0:
        end_index = len(dataset)
    else:
        end_index = min(len(dataset), args.start_index + args.num_samples * args.stride)
    indices = list(range(args.start_index, end_index, args.stride))
    logging.info("Dataset size=%s, selected_samples=%s", len(dataset), len(indices))

    trajectory_pred: Any = {name: [] for name in fk_solvers} if fk_solvers else []
    trajectory_gt: Any = {name: [] for name in fk_solvers} if fk_solvers else []
    save_file = Path(args.save_path).expanduser().resolve().open("w", encoding="utf-8") if args.save_path else None
    fps_sleep = 0.0 if args.fps <= 0 else 1.0 / args.fps
    warned_episode_cap = False
    episode_counts: dict[int, int] = {}

    try:
        for step, ds_idx in enumerate(indices, start=1):
            t0 = time.perf_counter()
            sample = dataset[ds_idx]

            if args.max_frames_per_episode > 0:
                epi = _extract_optional_int(sample, "episode_index")
                if epi is None:
                    if not warned_episode_cap:
                        logging.warning("episode_index missing; max_frames_per_episode ignored for such samples.")
                        warned_episode_cap = True
                else:
                    used = episode_counts.get(epi, 0)
                    if used >= args.max_frames_per_episode:
                        continue
                    episode_counts[epi] = used + 1

            state_vec = _select_last_frame(_to_numpy(sample[args.state_key])).reshape(-1).astype(np.float32)
            images_payload: dict[str, np.ndarray] = {}
            for dst_key, src_key in camera_map.items():
                if src_key not in sample:
                    raise KeyError(f"Sample idx={ds_idx} missing camera key '{src_key}'.")
                images_payload[dst_key] = _to_uint8_chw(sample[src_key])

            prompt = _safe_prompt(sample, args.prompt, tasks_map)
            policy_obs: dict[str, Any] = {"state": state_vec, "images": images_payload}
            if prompt is not None:
                policy_obs["prompt"] = prompt

            result = policy.infer(policy_obs)
            pred_steps = _extract_action_sequence(result.get("actions")) or []
            pred_vec = pred_steps[0] if pred_steps else None
            target_steps = _collect_target_action_steps(
                dataset=dataset,
                current_idx=ds_idx,
                current_sample=sample,
                action_key=args.action_key,
                start_step=args.target_action_index,
                max_steps=max(1, len(pred_steps)),
            )
            target_vec = target_steps[0] if target_steps else None

            loss_l1 = _compute_l1(pred_vec, target_vec)
            loss_l1_left = _compute_l1_on_indices(pred_vec, target_vec, left_indices)
            loss_l1_right = _compute_l1_on_indices(pred_vec, target_vec, right_indices)

            ee_pred_by_link: dict[str, list[float]] = {}
            ee_target_by_link: dict[str, list[float]] = {}
            ee_error_by_link: dict[str, float] = {}
            loss_ee: float | None = None
            ee_error_left: float | None = None
            ee_error_right: float | None = None
            if fk_solvers is not None and fk_maps is not None and pred_vec is not None and target_vec is not None:
                for link_name, solver in fk_solvers.items():
                    fk_map = fk_maps[link_name]
                    pred_point = _vector_to_fk_point(pred_vec, solver, fk_map, args.joint_unit)
                    tgt_point = _vector_to_fk_point(target_vec, solver, fk_map, args.joint_unit)
                    if pred_point is not None:
                        ee_pred_by_link[link_name] = pred_point
                    if tgt_point is not None:
                        ee_target_by_link[link_name] = tgt_point
                    ee_err = _compute_l2_distance(pred_point, tgt_point)
                    if ee_err is not None:
                        ee_error_by_link[link_name] = ee_err
                if ee_error_by_link:
                    loss_ee = float(np.mean(list(ee_error_by_link.values())))
                    ee_error_left, ee_error_right = _mean_errors_by_side(ee_error_by_link)

            if args.push_trajectory:
                if fk_solvers is not None:
                    for link_name in fk_solvers:
                        point_pred = ee_pred_by_link.get(link_name)
                        point_gt = ee_target_by_link.get(link_name)
                        if point_pred:
                            trajectory_pred.setdefault(link_name, []).append(point_pred)
                        if point_gt:
                            trajectory_gt.setdefault(link_name, []).append(point_gt)
                        if args.traj_max_len > 0:
                            if len(trajectory_pred.get(link_name, [])) > args.traj_max_len:
                                trajectory_pred[link_name] = trajectory_pred[link_name][-args.traj_max_len :]
                            if len(trajectory_gt.get(link_name, [])) > args.traj_max_len:
                                trajectory_gt[link_name] = trajectory_gt[link_name][-args.traj_max_len :]
                else:
                    if pred_vec is not None and len(pred_vec) > max(traj_indices):
                        trajectory_pred.append([pred_vec[traj_indices[0]], pred_vec[traj_indices[1]]])
                    if target_vec is not None and len(target_vec) > max(traj_indices):
                        trajectory_gt.append([target_vec[traj_indices[0]], target_vec[traj_indices[1]]])
                    if args.traj_max_len > 0:
                        if len(trajectory_pred) > args.traj_max_len:
                            trajectory_pred = trajectory_pred[-args.traj_max_len :]
                        if len(trajectory_gt) > args.traj_max_len:
                            trajectory_gt = trajectory_gt[-args.traj_max_len :]

            image_data: dict[str, str] = {}
            for key, img in images_payload.items():
                encoded = _encode_image_data_url(img)
                if encoded:
                    image_data[key] = encoded

            qpos = [float(v) for v in state_vec.tolist()]
            frame_payload = {
                "global_index": ds_idx,
                "episode_index": _extract_optional_int(sample, "episode_index"),
                "frame_index": _extract_optional_int(sample, "frame_index"),
                "timestamp": _extract_timestamp(sample),
                "loss_l1": loss_l1 if loss_l1 is not None else 0.0,
                "loss_l1_global": loss_l1,
                "loss_l1_left": loss_l1_left,
                "loss_l1_right": loss_l1_right,
                "loss_ee": loss_ee,
                "loss_ee_global": loss_ee,
                "ee_error_mean": loss_ee,
                "ee_error_left": ee_error_left,
                "ee_error_right": ee_error_right,
                "image_data": image_data or None,
                "detail": {
                    "qpos": qpos,
                    "action_pred": pred_vec or [],
                    "action_target": target_vec or [],
                    "ee_error_by_link": ee_error_by_link or {},
                    "ee_pred_by_link": ee_pred_by_link or {},
                    "ee_target_by_link": ee_target_by_link or {},
                },
            }

            payload = {
                "meta": {
                    "task_name": prompt,
                    "action_index": args.target_action_index,
                    "trajectory_mode": traj_mode,
                    "ee_links": ee_links if traj_mode == "fk3d" else None,
                },
                "frame": frame_payload,
                "trajectory_pred": trajectory_pred if args.push_trajectory else None,
                "trajectory_gt": trajectory_gt if args.push_trajectory else None,
            }

            if step % max(1, args.push_every) == 0:
                try:
                    _post_json(push_url, payload, timeout_s=args.push_timeout_s)
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    logging.warning("Push failed idx=%s: %s", ds_idx, exc)

            if save_file is not None:
                save_file.write(json.dumps(payload, ensure_ascii=True) + "\n")

            if step % max(1, args.log_every) == 0:
                logging.info(
                    "step=%s idx=%s loss_l1=%s loss_l1_left=%s loss_l1_right=%s loss_ee=%s",
                    step,
                    ds_idx,
                    f"{loss_l1:.6f}" if loss_l1 is not None else "None",
                    f"{loss_l1_left:.6f}" if loss_l1_left is not None else "None",
                    f"{loss_l1_right:.6f}" if loss_l1_right is not None else "None",
                    f"{loss_ee:.6f}" if loss_ee is not None else "None",
                )

            elapsed = time.perf_counter() - t0
            if fps_sleep > 0:
                time.sleep(max(0.0, fps_sleep - elapsed))
    finally:
        if save_file is not None:
            save_file.close()
        if viewer_proc is not None and viewer_proc.poll() is None:
            viewer_proc.terminate()
            try:
                viewer_proc.wait(timeout=2.0)
            except Exception:
                pass
        if fk_tempdir is not None:
            fk_tempdir.cleanup()


if __name__ == "__main__":
    main()
