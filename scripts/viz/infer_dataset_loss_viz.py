#!/usr/bin/env python3
"""Run dataset inference on Kai0 and visualize loss (Rerun optional).

Reference:
  /home/magiclab/projects/lerobot/tools/infer_from_dataset_service.py
"""

from __future__ import annotations

import argparse
import base64
from bisect import bisect_right
import csv
import importlib.util
import io
import json
import logging
import re
import sys
import tempfile
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_AV_IMPORT_ERROR: Exception | None = None
try:
    import av
except Exception as exc:  # pragma: no cover - runtime dependency
    av = None  # type: ignore[assignment]
    _AV_IMPORT_ERROR = exc

_PYARROW_IMPORT_ERROR: Exception | None = None
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover - runtime dependency
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]
    _PYARROW_IMPORT_ERROR = exc


def _ensure_lerobot_stub() -> None:
    try:
        if importlib.util.find_spec("lerobot.datasets.lerobot_dataset") is not None:
            return
    except ModuleNotFoundError:
        pass

    lerobot_pkg = sys.modules.setdefault("lerobot", types.ModuleType("lerobot"))
    if not hasattr(lerobot_pkg, "__path__"):
        lerobot_pkg.__path__ = []  # type: ignore[attr-defined]

    datasets_pkg = sys.modules.setdefault("lerobot.datasets", types.ModuleType("lerobot.datasets"))
    if not hasattr(datasets_pkg, "__path__"):
        datasets_pkg.__path__ = []  # type: ignore[attr-defined]

    dataset_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")

    class _DummyLeRobotDataset:
        pass

    dataset_mod.LeRobotDataset = _DummyLeRobotDataset
    sys.modules["lerobot.datasets.lerobot_dataset"] = dataset_mod

_TORCH_IMPORT_ERROR: Exception | None = None
try:
    import torch
except Exception as exc:  # pragma: no cover - runtime dependency
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

_OPENPI_CLIENT_IMPORT_ERROR: Exception | None = None
try:
    from openpi_client import websocket_client_policy
except Exception as exc:  # pragma: no cover - runtime dependency
    websocket_client_policy = None  # type: ignore[assignment]
    _OPENPI_CLIENT_IMPORT_ERROR = exc

_LEROBOT_IMPORT_ERROR: Exception | None = None
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except Exception as exc:  # pragma: no cover - runtime dependency
    LeRobotDataset = Any  # type: ignore[assignment,misc]
    _LEROBOT_IMPORT_ERROR = exc

_ROBOT_KIN_IMPORT_ERROR: Exception | None = None
try:
    from lerobot.model.kinematics import RobotKinematics
except Exception as exc:  # pragma: no cover - runtime dependency
    RobotKinematics = None  # type: ignore[assignment]
    _ROBOT_KIN_IMPORT_ERROR = exc

_RERUN_IMPORT_ERROR: Exception | None = None
try:
    import rerun as rr
except Exception as exc:  # pragma: no cover - runtime dependency
    rr = None  # type: ignore[assignment]
    _RERUN_IMPORT_ERROR = exc

_MATPLOTLIB_IMPORT_ERROR: Exception | None = None
try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - runtime dependency
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_IMPORT_ERROR = exc


DEFAULT_URDF_ROOT = Path("/home/magiclab/projects/mobile_aloha_sim_ros2")
DEFAULT_URDF_PATH = DEFAULT_URDF_ROOT / "aloha_new_description/urdf/aloha_new.urdf"
DEFAULT_CAMERA_MAP = (
    "top_head=observation.images.top_head,"
    "hand_left=observation.images.hand_left,"
    "hand_right=observation.images.hand_right"
)
magiclab_CAMERA_MAP = (
    "top_head=observation.images.headf_rgbd_color,"
    "hand_left=observation.images.hand_l_rgb,"
    "hand_right=observation.images.hand_r_rgb,"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset inference + loss visualization for Kai0.")
    parser.add_argument("--dataset-repo-id", "--repo-id", dest="dataset_repo_id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dataset-episodes", default=None, help="e.g. '[0,1]' or '0,1,3-8'")
    parser.add_argument("--host", default="127.0.0.1", help="Policy websocket host.")
    parser.add_argument("--port", type=int, default=8000, help="Policy websocket port.")
    parser.add_argument("--local-policy-config", default=None, help="Load local policy config directly (no websocket).")
    parser.add_argument("--local-policy-dir", type=Path, default=None, help="Checkpoint dir for --local-policy-config.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=500, help="<=0 means full dataset")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames-per-episode", type=int, default=0)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--camera-map", default=magiclab_CAMERA_MAP, help="policy_cam=dataset_key,...")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--target-action-index", type=int, default=0)
    parser.add_argument("--viz", choices=["rerun", "none"], default="rerun")
    parser.add_argument("--no-spawn-rerun", action="store_true")
    parser.add_argument("--rerun-name", default="kai0/dataset_infer_loss")
    parser.add_argument("--log-images", action="store_true", help="Log camera images to rerun.")
    parser.add_argument("--save-csv", type=Path, default=None)
    parser.add_argument("--save-plot", type=Path, default=None, help="Save loss curve figure (.png).")
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--urdf-root", type=Path, default=DEFAULT_URDF_ROOT)
    parser.add_argument("--ee-links", default="fl_link8,fr_link8", help="left,right ee link names")
    parser.add_argument("--joint-unit", choices=["rad", "deg"], default="rad")
    parser.add_argument("--web-viewer-url", default=None, help="Push live frames to viewer server, e.g. http://127.0.0.1:9001")
    parser.add_argument(
        "--web-record-path",
        type=Path,
        default=None,
        help="Record web viewer reset/push payloads to a local JSONL file for offline replay.",
    )
    parser.add_argument("--web-push-every", type=int, default=1, help="Batch N frames before POST to /api/push")
    parser.add_argument("--web-log-images", action="store_true", help="Embed camera images into web viewer payloads.")
    parser.add_argument("--web-reset-first", action="store_true", help="POST /api/reset before starting live push.")
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
            raise ValueError("--dataset-episodes must be a JSON list.")
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
        raise ValueError("Episode index must be non-negative.")
    return out


def _parse_camera_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid camera map item: {item}")
        dst, src = item.split("=", 1)
        out[dst.strip()] = src.strip()
    if not out:
        raise ValueError("--camera-map is empty.")
    return out


class _SequentialVideoReader:
    def __init__(self, video_path: Path):
        if av is None:
            raise RuntimeError("PyAV is required for local Kai0 dataset video reading.") from _AV_IMPORT_ERROR
        self._video_path = video_path
        self._container = None
        self._frame_iter = None
        self._current_index = -1
        self._open()

    def _open(self) -> None:
        if self._container is not None:
            self._container.close()
        self._container = av.open(str(self._video_path))
        stream = self._container.streams.video[0]
        stream.thread_type = "NONE"
        stream.thread_count = 1

        def _iter():
            for packet in self._container.demux(stream):
                for frame in packet.decode():
                    yield frame.to_rgb().to_ndarray()

        self._frame_iter = _iter()
        self._current_index = -1

    def read(self, frame_index: int) -> np.ndarray:
        if frame_index < self._current_index:
            self._open()
        assert self._frame_iter is not None
        frame = None
        while self._current_index < frame_index:
            frame = next(self._frame_iter)
            self._current_index += 1
        if frame is None:
            raise IndexError(f"Frame {frame_index} not available in {self._video_path}")
        return np.asarray(frame)

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None


class _LocalKai0Dataset:
    def __init__(self, dataset_root: Path, episodes: list[int] | None = None):
        if pq is None or pa is None:
            raise RuntimeError("pyarrow is required for local Kai0 dataset loading.") from _PYARROW_IMPORT_ERROR
        self.root = dataset_root
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.features = self.info.get("features", {})
        self.fps = int(self.info["fps"])
        self._tasks_map = self._load_tasks_map()
        self.meta = SimpleNamespace(tasks=self._tasks_map)
        self._episode_rows = self._read_episode_rows()
        self._row_by_episode = {int(row["episode_index"]): row for row in self._episode_rows}
        if episodes is None:
            self.episodes = sorted(self._row_by_episode)
        else:
            self.episodes = episodes
        self._data_file_starts, self._video_file_starts = self._build_file_start_indices()
        self._spans: list[dict[str, Any]] = []
        running = 0
        for episode_index in self.episodes:
            row = self._row_by_episode.get(int(episode_index))
            if row is None:
                raise ValueError(f"Episode {episode_index} not found in dataset metadata.")
            length = int(row["dataset_to_index"]) - int(row["dataset_from_index"])
            self._spans.append({"episode_index": int(episode_index), "row": row, "start": running, "length": length})
            running += length
        self._len = running
        self._table_cache: dict[Path, dict[str, list[Any]]] = {}
        self._video_cache: dict[Path, _SequentialVideoReader] = {}

    def _load_tasks_map(self) -> dict[int, str]:
        tasks_parquet = self.root / "meta/tasks.parquet"
        if tasks_parquet.exists():
            table = pq.read_table(tasks_parquet).to_pydict()
            if "task_index" in table and "__index_level_0__" in table:
                return {int(k): str(v) for k, v in zip(table["task_index"], table["__index_level_0__"], strict=True)}
        tasks_jsonl = self.root / "meta/tasks.jsonl"
        if tasks_jsonl.exists():
            out: dict[int, str] = {}
            for line in tasks_jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                out[int(item["task_index"])] = str(item["task"])
            return out
        return {}

    def _read_episode_rows(self) -> list[dict[str, Any]]:
        files = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No episode parquet files found under {self.root / 'meta/episodes'}")
        rows: list[dict[str, Any]] = []
        for file in files:
            table = pq.read_table(file)
            columns = table.to_pydict()
            for idx in range(table.num_rows):
                rows.append({key: value[idx] for key, value in columns.items()})
        rows.sort(key=lambda item: int(item["episode_index"]))
        return rows

    def _build_file_start_indices(self) -> tuple[dict[tuple[int, int], int], dict[tuple[str, int, int], int]]:
        data_starts: dict[tuple[int, int], int] = {}
        video_starts: dict[tuple[str, int, int], int] = {}
        camera_keys = [key for key, spec in self.features.items() if spec.get("dtype") == "video"]
        for row in self._episode_rows:
            data_key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
            data_start = int(row["dataset_from_index"])
            data_starts[data_key] = min(data_starts.get(data_key, data_start), data_start)
            for camera_key in camera_keys:
                chunk_key = f"videos/{camera_key}/chunk_index"
                file_key = f"videos/{camera_key}/file_index"
                from_key = f"videos/{camera_key}/from_timestamp"
                video_key = (camera_key, int(row[chunk_key]), int(row[file_key]))
                video_start = int(round(float(row[from_key]) * self.fps))
                video_starts[video_key] = min(video_starts.get(video_key, video_start), video_start)
        return data_starts, video_starts

    def _data_file_path(self, row: dict[str, Any]) -> Path:
        return self.root / self.info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )

    def _video_file_path(self, row: dict[str, Any], camera_key: str) -> Path:
        return self.root / self.info["video_path"].format(
            video_key=camera_key,
            chunk_index=int(row[f"videos/{camera_key}/chunk_index"]),
            file_index=int(row[f"videos/{camera_key}/file_index"]),
        )

    def _load_table(self, data_path: Path) -> dict[str, list[Any]]:
        if data_path in self._table_cache:
            return self._table_cache[data_path]
        table = pq.read_table(data_path)
        out: dict[str, list[Any]] = {}
        for column_name in table.column_names:
            column = table[column_name]
            assert isinstance(column, pa.ChunkedArray)
            combined = column.combine_chunks()
            if isinstance(combined, pa.FixedSizeListArray):
                values = combined.flatten().to_numpy()
                out[column_name] = values.reshape(len(combined), combined.type.list_size).tolist()
            elif isinstance(combined, pa.ListArray):
                out[column_name] = combined.to_pylist()
            else:
                out[column_name] = combined.to_pylist()
        self._table_cache[data_path] = out
        return out

    def _find_span(self, index: int) -> tuple[dict[str, Any], int]:
        if index < 0:
            index += self._len
        if index < 0 or index >= self._len:
            raise IndexError(index)
        starts = [span["start"] for span in self._spans]
        span_idx = max(0, bisect_right(starts, index) - 1)
        span = self._spans[span_idx]
        return span, index - int(span["start"])

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> dict[str, Any]:
        span, rel_index = self._find_span(index)
        row = span["row"]
        data_path = self._data_file_path(row)
        data = self._load_table(data_path)

        data_key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        data_base = self._data_file_starts[data_key]
        row_index = int(row["dataset_from_index"]) - data_base + rel_index

        sample: dict[str, Any] = {key: value[row_index] for key, value in data.items()}
        task_index = sample.get("task_index")
        if task_index is not None and int(task_index) in self._tasks_map:
            sample["task"] = self._tasks_map[int(task_index)]

        frame_index = int(sample.get("frame_index", rel_index))
        for camera_key, spec in self.features.items():
            if spec.get("dtype") != "video":
                continue
            video_path = self._video_file_path(row, camera_key)
            reader = self._video_cache.get(video_path)
            if reader is None:
                reader = _SequentialVideoReader(video_path)
                self._video_cache[video_path] = reader
            video_key = (camera_key, int(row[f"videos/{camera_key}/chunk_index"]), int(row[f"videos/{camera_key}/file_index"]))
            frame_base = self._video_file_starts[video_key]
            start_frame = int(round(float(row[f"videos/{camera_key}/from_timestamp"]) * self.fps)) - frame_base
            sample[camera_key] = reader.read(start_frame + frame_index)

        return sample

    def close(self) -> None:
        for reader in self._video_cache.values():
            reader.close()
        self._video_cache.clear()


def _load_dataset(dataset_repo_id: str, dataset_root: str | None, episodes: list[int] | None):
    dataset_path = Path(dataset_repo_id)
    if dataset_path.exists():
        return _LocalKai0Dataset(dataset_path.resolve(), episodes=episodes)
    if _LEROBOT_IMPORT_ERROR is not None:
        raise RuntimeError("lerobot is required for non-local dataset loading.") from _LEROBOT_IMPORT_ERROR
    return LeRobotDataset(repo_id=dataset_repo_id, root=dataset_root, episodes=episodes)


def _load_local_norm_stats(checkpoint_dir: Path) -> dict[str, Any]:
    direct = checkpoint_dir / "norm_stats.json"
    if direct.exists():
        from openpi.shared import normalize as _normalize

        return _normalize.deserialize_json(direct.read_text())
    matches = sorted(checkpoint_dir.glob("assets/**/norm_stats.json"))
    if matches:
        from openpi.shared import normalize as _normalize

        return _normalize.deserialize_json(matches[0].read_text())
    raise FileNotFoundError(f"Could not find norm_stats.json under {checkpoint_dir}")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_scalar(value: Any, default: float = 0.0) -> float:
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return default
    arr = np.asarray(value)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return default


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
        raise ValueError(f"Expected image ndim=3, got {img.shape}")
    if img.shape[-1] in (1, 3, 4) and img.shape[0] not in (1, 3, 4):
        img = np.transpose(img, (2, 0, 1))
    if np.issubdtype(img.dtype, np.floating):
        scale = 255.0 if float(np.max(img)) <= 1.5 else 1.0
        img = np.clip(img * scale, 0, 255).round().astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _to_uint8_hwc(image_chw: np.ndarray) -> np.ndarray:
    img = np.asarray(image_chw)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        return np.transpose(img, (1, 2, 0))
    return img


def _extract_optional_int(sample: dict[str, Any], key: str) -> int | None:
    if key not in sample:
        return None
    try:
        return int(_to_scalar(sample[key], default=0.0))
    except Exception:
        return None


def _extract_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        t = value.strip()
        return t if t else None
    if isinstance(value, bytes):
        t = value.decode("utf-8", errors="ignore").strip()
        return t if t else None
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _extract_text(value.item())
        return None
    arr = np.asarray(value)
    if arr.size == 1:
        return _extract_text(arr.reshape(-1)[0].item())
    return None


def _load_tasks_map(dataset: LeRobotDataset) -> dict[int, str]:
    tasks = getattr(getattr(dataset, "meta", None), "tasks", None)
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


def _extract_feature_names(dataset: LeRobotDataset, key: str) -> list[str]:
    features = getattr(dataset, "features", {}) or {}
    if key not in features:
        return []
    names = features[key].get("names")
    if not isinstance(names, list):
        return []
    if len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    return [str(name) for name in names]


def _extract_action_sequence(action: Any) -> list[list[float]] | None:
    if action is None:
        return None
    arr = _to_numpy(action)
    if arr.ndim == 0:
        return [[float(arr)]]
    if arr.ndim == 1:
        return [[float(x) for x in arr.reshape(-1)]]
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim == 1:
        return [[float(x) for x in arr.reshape(-1)]]
    return [[float(x) for x in row.reshape(-1)] for row in arr]


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

    start_step = max(0, int(start_step))
    steps: list[list[float]] = []
    current_seq = _extract_action_sequence(current_sample.get(action_key)) or []
    base_len = len(current_seq)
    current_episode = _extract_optional_int(current_sample, "episode_index")
    current_frame = _extract_optional_int(current_sample, "frame_index")

    for rel in range(max_steps):
        offset = start_step + rel
        if offset < base_len:
            steps.append(current_seq[offset])
            continue

        next_idx = current_idx + (offset - base_len + 1)
        if next_idx >= len(dataset):
            break
        next_sample = dataset[next_idx]

        if current_episode is not None:
            next_episode = _extract_optional_int(next_sample, "episode_index")
            if next_episode is None or next_episode != current_episode:
                break
        if current_frame is not None:
            next_frame = _extract_optional_int(next_sample, "frame_index")
            expected = current_frame + (offset - base_len + 1)
            if next_frame is not None and next_frame != expected:
                break

        next_seq = _extract_action_sequence(next_sample.get(action_key)) if action_key in next_sample else None
        if not next_seq:
            break
        steps.append(next_seq[0])

    return steps


def _compute_l1(pred: list[float] | None, target: list[float] | None) -> float | None:
    if pred is None or target is None or len(pred) != len(target):
        return None
    return float(np.mean(np.abs(np.asarray(pred, dtype=float) - np.asarray(target, dtype=float))))


def _infer_side_from_name(name: str | None) -> str | None:
    if not name:
        return None
    text = str(name).strip().lower()
    if not text:
        return None
    if "right" in text or text.startswith("r_") or text.endswith("_r") or text.startswith("fr_") or text.startswith("br_"):
        return "right"
    if "left" in text or text.startswith("l_") or text.endswith("_l") or text.startswith("fl_") or text.startswith("bl_"):
        return "left"
    return None


def _split_indices_by_side(names: list[str]) -> tuple[list[int], list[int]]:
    left: list[int] = []
    right: list[int] = []
    for i, name in enumerate(names):
        side = _infer_side_from_name(name)
        if side == "left":
            left.append(i)
        elif side == "right":
            right.append(i)
    if not left and not right and len(names) >= 14:
        left = list(range(0, 7))
        right = list(range(7, 14))
    return left, right


def _default_side_indices(dataset: Any, state_key: str, action_key: str) -> tuple[list[int], list[int]]:
    features = getattr(dataset, "features", {}) or {}
    for key in (action_key, state_key):
        spec = features.get(key)
        if not isinstance(spec, dict):
            continue
        shape = spec.get("shape")
        if isinstance(shape, list) and shape and int(shape[0]) >= 14:
            return list(range(0, 7)), list(range(7, 14))
    return [], []


def _compute_l1_on_indices(pred: list[float] | None, target: list[float] | None, indices: list[int]) -> float | None:
    if pred is None or target is None or len(pred) != len(target) or not indices:
        return None
    vals = [abs(float(pred[i]) - float(target[i])) for i in indices if 0 <= i < len(pred)]
    if not vals:
        return None
    return float(np.mean(vals))


def _compute_l2_distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None or len(a) != len(b):
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _mean_errors_by_side(errors_by_name: dict[str, float]) -> tuple[float | None, float | None]:
    left_vals: list[float] = []
    right_vals: list[float] = []
    for name, val in errors_by_name.items():
        side = _infer_side_from_name(name)
        if side == "left":
            left_vals.append(float(val))
        elif side == "right":
            right_vals.append(float(val))
    return (float(np.mean(left_vals)) if left_vals else None, float(np.mean(right_vals)) if right_vals else None)


def _looks_like_magiclab_names(names: list[str]) -> bool:
    return {"left_arm_joint1", "right_arm_joint1"}.issubset(set(names))


def _resolve_urdf_resource(resource: str, urdf_path: Path, urdf_root: Path | None) -> Path | None:
    if resource.startswith("package://"):
        parts = resource[len("package://") :].split("/", 1)
        if len(parts) != 2:
            return None
        package, rel_path = parts
        candidates = [parent / package / rel_path for parent in urdf_path.parents]
        if urdf_root is not None:
            candidates.append(urdf_root / package / rel_path)
        for cand in candidates:
            if cand.exists():
                return cand.resolve()
        return None
    if resource.startswith("file://"):
        cand = Path(resource[len("file://") :]).expanduser()
        return cand.resolve() if cand.exists() else None
    cand = Path(resource)
    if not cand.is_absolute():
        cand = (urdf_path.parent / cand).resolve()
    return cand if cand.exists() else None


def _rewrite_urdf_for_fk(urdf_path: Path, urdf_root: Path | None, out_dir: Path) -> Path:
    text = urdf_path.read_text(encoding="utf-8")
    rewritten = text
    for match in set(re.findall(r'filename="([^"]+)"', text)):
        resolved = _resolve_urdf_resource(match, urdf_path, urdf_root)
        if resolved is not None:
            rewritten = rewritten.replace(match, str(resolved))
    out_path = out_dir / "robot_fk.urdf"
    out_path.write_text(rewritten, encoding="utf-8")
    return out_path


def _build_urdf_to_dataset_map(dataset_joint_names: list[str], urdf_joint_names: list[str]) -> list[int | None]:
    # import pdb;pdb.set_trace()
    mapping: list[int | None] = [None] * len(urdf_joint_names)
    if dataset_joint_names and any(name in urdf_joint_names for name in dataset_joint_names):
        for ds_idx, name in enumerate(dataset_joint_names):
            if name in urdf_joint_names:
                mapping[urdf_joint_names.index(name)] = ds_idx
        return mapping

    if _looks_like_magiclab_names(dataset_joint_names):
        if 'left_hip_roll_joint' in urdf_joint_names:
            left_arm = [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_roll_joint",
                "left_wrist_yaw_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
            ]
            right_arm = [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_roll_joint",
                "right_wrist_yaw_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
            ]
            waist = ["waist_yaw_joint", "waist_roll_joint"]
            head = ["head_yaw_joint", "head_pitch_joint"]
            for idx, name in enumerate(left_arm):
                if idx < len(mapping):
                    mapping[urdf_joint_names.index(name)] = idx
            for idx, name in enumerate(right_arm):
                if 7 + idx < len(mapping):
                    mapping[urdf_joint_names.index(name)]=7+idx
            for idx, name in enumerate(waist):
                if 26 + idx < len(mapping):
                    mapping[urdf_joint_names.index(name)]=26+idx
            for idx, name in enumerate(head):
                if 28 + idx < len(mapping):
                    mapping[urdf_joint_names.index(name)]=28+idx
            return mapping
        else:
            for i in range(7):
                left_ds = f"left_arm_joint{i + 1}"
                right_ds = f"right_arm_joint{i + 1}"
                left_urdf = f"fl_joint{i + 1}"
                right_urdf = f"fr_joint{i + 1}"
                if left_ds in dataset_joint_names and left_urdf in urdf_joint_names:
                    mapping[urdf_joint_names.index(left_urdf)] = dataset_joint_names.index(left_ds)
                if right_ds in dataset_joint_names and right_urdf in urdf_joint_names:
                    mapping[urdf_joint_names.index(right_urdf)] = dataset_joint_names.index(right_ds)
            return mapping

    if len(dataset_joint_names) >= 14 or not dataset_joint_names:
        defaults = [f"fl_joint{i}" for i in range(1, 8)] + [f"fr_joint{i}" for i in range(1, 8)]
        for ds_idx, urdf_name in enumerate(defaults):
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
    if hasattr(solver, "forward_kinematics"):
        full = np.zeros(len(urdf_to_dataset), dtype=float)
        for urdf_idx, ds_idx in enumerate(urdf_to_dataset):
            if ds_idx is None or ds_idx >= len(vec):
                continue
            full[urdf_idx] = float(vec[ds_idx])
        if joint_unit == "rad":
            full = np.rad2deg(full)
        fk = solver.forward_kinematics(full)
        return fk[:3, 3].tolist()

    if isinstance(solver, tuple) and len(solver) == 2:
        from scripts.viz.rerun_dataset_viz import _joint_map_from_state as _joint_map_from_state_rerun

        kin, link_name = solver
        joint_map = _joint_map_from_state_rerun(
            np.asarray(vec, dtype=np.float64),
            kin.joint_names,
            {name: idx for name, idx in zip(kin.joint_names, urdf_to_dataset, strict=True) if idx is not None},
            add_symmetric_gripper=True,
        )
        point = kin.compute_link_position(joint_map, link_name)
        return point.tolist() if point is not None else None

    full = np.zeros(len(urdf_to_dataset), dtype=float)
    for urdf_idx, ds_idx in enumerate(urdf_to_dataset):
        if ds_idx is None or ds_idx >= len(vec):
            continue
        full[urdf_idx] = float(vec[ds_idx])
    if joint_unit == "rad":
        full = np.rad2deg(full)
    fk = solver.forward_kinematics(full)
    return fk[:3, 3].tolist()


def _init_fk(
    dataset_joint_names: list[str],
    urdf_path: Path,
    urdf_root: Path | None,
    ee_links: list[str],
) -> tuple[dict[str, Any] | None, dict[str, list[int | None]] | None, tempfile.TemporaryDirectory[str] | None]:
    if not ee_links:
        return None, None, None
    if not urdf_path.exists():
        logging.warning("URDF not found: %s, skip ee loss.", urdf_path)
        return None, None, None

    tmp = tempfile.TemporaryDirectory(prefix="kai0_fk_")
    fk_urdf = _rewrite_urdf_for_fk(urdf_path, urdf_root, Path(tmp.name))
    solvers: dict[str, Any] = {}
    maps: dict[str, list[int | None]] = {}
    if RobotKinematics is not None:
        for link_name in ee_links:
            solver = RobotKinematics(str(fk_urdf), target_frame_name=link_name, joint_names=None)
            mapping = _build_urdf_to_dataset_map(dataset_joint_names, solver.joint_names)
            if not any(idx is not None for idx in mapping):
                logging.warning("FK mapping is empty for link %s, skip.", link_name)
                continue
            solvers[link_name] = solver
            maps[link_name] = mapping
    else:
        from scripts.viz.rerun_dataset_viz import URDFKinematics

        logging.warning("RobotKinematics unavailable, falling back to URDFKinematics.")
        kin = URDFKinematics(fk_urdf)
        mapping = _build_urdf_to_dataset_map(dataset_joint_names, kin.joint_names)
        if not any(idx is not None for idx in mapping):
            logging.warning("FK mapping is empty for fallback URDFKinematics.")
        else:
            for link_name in ee_links:
                if not kin.has_link(link_name):
                    logging.warning("Link %s not found in URDF, skip.", link_name)
                    continue
                solvers[link_name] = (kin, link_name)
                maps[link_name] = mapping

    if not solvers:
        tmp.cleanup()
        return None, None, None
    return solvers, maps, tmp


def _init_rerun(name: str, spawn: bool) -> None:
    rr.init(name, spawn=spawn)


def _log_rerun_scalars(
    frame_index: int,
    timestamp: float,
    loss_l1: float | None,
    loss_l1_left: float | None,
    loss_l1_right: float | None,
    loss_ee: float | None,
    pred_vec: list[float] | None,
    target_vec: list[float] | None,
) -> None:
    rr.set_time("frame_index", sequence=frame_index)
    rr.set_time("timestamp", timestamp=timestamp)
    rr.log("loss/l1", rr.Scalars(float(loss_l1) if loss_l1 is not None else np.nan))
    rr.log("loss/l1_left", rr.Scalars(float(loss_l1_left) if loss_l1_left is not None else np.nan))
    rr.log("loss/l1_right", rr.Scalars(float(loss_l1_right) if loss_l1_right is not None else np.nan))
    rr.log("loss/ee", rr.Scalars(float(loss_ee) if loss_ee is not None else np.nan))
    if pred_vec is not None:
        for i, v in enumerate(pred_vec):
            rr.log(f"pred_action/{i}", rr.Scalars(float(v)))
    if target_vec is not None:
        for i, v in enumerate(target_vec):
            rr.log(f"target_action/{i}", rr.Scalars(float(v)))


def _plot_loss_curves(path: Path, x: list[int], l1: list[float], l1_left: list[float], l1_right: list[float], ee: list[float]) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for --save-plot") from _MATPLOTLIB_IMPORT_ERROR
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=140)
    ax.plot(x, l1, label="l1", linewidth=1.6)
    ax.plot(x, l1_left, label="l1_left", linewidth=1.2)
    ax.plot(x, l1_right, label="l1_right", linewidth=1.2)
    ax.plot(x, ee, label="ee", linewidth=1.2)
    ax.set_xlabel("sample")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _image_to_data_url(image_chw: np.ndarray) -> str:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Pillow is required for --web-log-images") from exc

    image_hwc = _to_uint8_hwc(image_chw)
    buffer = io.BytesIO()
    Image.fromarray(image_hwc).save(buffer, format="JPEG", quality=82)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _post_json(url: str, payload: dict[str, Any], timeout_s: float = 5.0) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout_s) as resp:
        if int(getattr(resp, "status", 200)) >= 400:
            raise RuntimeError(f"Viewer POST failed with status {resp.status}")


def _append_jsonl_event(path: Path, event_type: str, payload: dict[str, Any]) -> None:
    record = {
        "event": event_type,
        "recorded_at": time.time(),
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=True) + "\n")


def _safe_mean(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def _push_viewer_payload(
    viewer_url: str | None = None,
    *,
    record_path: Path | None = None,
    meta: dict[str, Any] | None = None,
    frames: list[dict[str, Any]] | None = None,
    trajectory_pred: dict[str, list[list[float]]] | None = None,
    trajectory_gt: dict[str, list[list[float]]] | None = None,
    reset_first: bool = False,
) -> None:
    if reset_first:
        if viewer_url is not None:
            base = viewer_url.rstrip("/")
            _post_json(f"{base}/api/reset", {})
        if record_path is not None:
            _append_jsonl_event(record_path, "reset", {})
    payload: dict[str, Any] = {}
    if meta:
        payload["meta"] = meta
    if frames:
        payload["frames"] = frames
    if trajectory_pred is not None:
        payload["trajectory_pred"] = trajectory_pred
    if trajectory_gt is not None:
        payload["trajectory_gt"] = trajectory_gt
    if not payload:
        return
    payload["timestamp"] = time.time()
    if viewer_url is not None:
        base = viewer_url.rstrip("/")
        _post_json(f"{base}/api/push", payload)
    if record_path is not None:
        _append_jsonl_event(record_path, "push", payload)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if torch is None:
        raise RuntimeError("torch is required.") from _TORCH_IMPORT_ERROR
    if _LEROBOT_IMPORT_ERROR is not None and not Path(args.dataset_repo_id).exists():
        raise RuntimeError("lerobot is required.") from _LEROBOT_IMPORT_ERROR
    if args.viz == "rerun" and rr is None:
        raise RuntimeError("rerun is required when --viz=rerun.") from _RERUN_IMPORT_ERROR

    selected_episodes = _parse_dataset_episodes(args.dataset_episodes)
    camera_map = _parse_camera_map(args.camera_map)
    dataset_repo_id = str(Path(args.dataset_repo_id).resolve()) if Path(args.dataset_repo_id).exists() else args.dataset_repo_id
    ee_links = [s.strip() for s in args.ee_links.split(",") if s.strip()]

    dataset = _load_dataset(dataset_repo_id, args.dataset_root, selected_episodes)
    tasks_map = _load_tasks_map(dataset)
    state_names = _extract_feature_names(dataset, args.state_key)
    action_names = _extract_feature_names(dataset, args.action_key)
    left_indices, right_indices = _split_indices_by_side(action_names or state_names)
    if not left_indices and not right_indices:
        left_indices, right_indices = _default_side_indices(dataset, args.state_key, args.action_key)
    dataset_joint_names = action_names or state_names

    use_local_policy = args.local_policy_config is not None or args.local_policy_dir is not None
    viewer_url = args.web_viewer_url.rstrip("/") if args.web_viewer_url else None
    viewer_record_path = args.web_record_path.expanduser().resolve() if args.web_record_path else None
    policy_source = None
    if use_local_policy:
        if not args.local_policy_config or args.local_policy_dir is None:
            raise ValueError("--local-policy-config and --local-policy-dir must be provided together.")
        _ensure_lerobot_stub()
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        policy_cfg = _config.get_config(args.local_policy_config)
        policy_dir = args.local_policy_dir.expanduser().resolve()
        norm_stats = _load_local_norm_stats(policy_dir)
        policy = _policy_config.create_trained_policy(
            policy_cfg,
            policy_dir,
            norm_stats=norm_stats,
            default_prompt=getattr(policy_cfg.data, "default_prompt", None),
        )
        policy_source = str(policy_dir)
        logging.info("Loaded local policy config=%s dir=%s", args.local_policy_config, policy_dir)
        logging.info("Policy metadata: %s", policy.metadata)
    else:
        if websocket_client_policy is None:
            raise RuntimeError("openpi_client is required for websocket policy mode.") from _OPENPI_CLIENT_IMPORT_ERROR
        policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        policy_source = f"ws://{args.host}:{args.port}"
        logging.info("Connected policy server %s:%s", args.host, args.port)
        logging.info("Server metadata: %s", policy.get_server_metadata())

    fk_solvers: dict[str, Any] | None = None
    fk_maps: dict[str, list[int | None]] | None = None
    fk_tmp: tempfile.TemporaryDirectory[str] | None = None
    if args.urdf_path and ee_links:
        fk_solvers, fk_maps, fk_tmp = _init_fk(
            dataset_joint_names=dataset_joint_names,
            urdf_path=args.urdf_path.expanduser().resolve(),
            urdf_root=args.urdf_root.expanduser().resolve() if args.urdf_root else None,
            ee_links=ee_links,
        )
        if fk_solvers is not None:
            logging.info("FK enabled for links: %s", ", ".join(fk_solvers.keys()))

    if args.viz == "rerun":
        _init_rerun(args.rerun_name, spawn=not args.no_spawn_rerun)

    if viewer_record_path is not None:
        viewer_record_path.parent.mkdir(parents=True, exist_ok=True)
        if viewer_record_path.exists():
            viewer_record_path.unlink()
        _append_jsonl_event(
            viewer_record_path,
            "session_start",
            {
                "dataset_repo_id": dataset_repo_id,
                "policy_source": policy_source,
            },
        )
        logging.info("Viewer recording enabled: %s", viewer_record_path)

    if viewer_url is not None or viewer_record_path is not None:
        viewer_meta = {
            "task_name": dataset_repo_id,
            "ckpt_path": policy_source,
            "num_frames": len(dataset),
            "action_index": args.target_action_index,
            "mode": "live" if viewer_url is not None else "offline_recorded",
            "camera_keys": list(camera_map.keys()),
        }
        try:
            _push_viewer_payload(
                viewer_url,
                record_path=viewer_record_path,
                meta=viewer_meta,
                reset_first=args.web_reset_first,
            )
            if viewer_url is not None:
                logging.info("Viewer push enabled: %s", viewer_url)
        except (OSError, urllib_error.URLError, RuntimeError) as exc:
            raise RuntimeError(f"Failed to initialize viewer push to {viewer_url}: {exc}") from exc

    csv_writer = None
    csv_fp = None
    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_fp = args.save_csv.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_fp,
            fieldnames=[
                "sample_idx",
                "dataset_index",
                "episode_index",
                "frame_index",
                "timestamp",
                "loss_l1",
                "loss_l1_left",
                "loss_l1_right",
                "loss_ee",
            ],
        )
        csv_writer.writeheader()

    if args.num_samples <= 0:
        end_index = len(dataset)
    else:
        end_index = min(len(dataset), args.start_index + args.num_samples * args.stride)
    indices = list(range(args.start_index, end_index, args.stride))
    logging.info("Total samples to process: %d", len(indices))

    fps_sleep = 0.0 if args.fps <= 0 else 1.0 / args.fps
    warned_episode_cap = False
    episode_counts: dict[int, int] = {}
    xs: list[int] = []
    loss_l1_series: list[float] = []
    loss_l1_left_series: list[float] = []
    loss_l1_right_series: list[float] = []
    loss_ee_series: list[float] = []
    viewer_frames_batch: list[dict[str, Any]] = []
    viewer_traj_pred: dict[str, list[list[float]]] = {}
    viewer_traj_gt: dict[str, list[list[float]]] = {}

    try:
        for step, ds_idx in enumerate(indices, start=1):
            t0 = time.perf_counter()
            sample = dataset[ds_idx]

            if args.max_frames_per_episode > 0:
                epi = _extract_optional_int(sample, "episode_index")
                if epi is None:
                    if not warned_episode_cap:
                        logging.warning("episode_index missing, max-frames-per-episode ignored for these samples.")
                        warned_episode_cap = True
                else:
                    used = episode_counts.get(epi, 0)
                    if used >= args.max_frames_per_episode:
                        continue
                    episode_counts[epi] = used + 1

            state_arr = _to_numpy(sample[args.state_key])
            state_vec = _select_last_frame(state_arr).reshape(-1).astype(np.float32)
            images_payload: dict[str, np.ndarray] = {}
            for policy_key, ds_key in camera_map.items():
                if ds_key not in sample:
                    raise KeyError(f"Sample idx={ds_idx} missing camera key '{ds_key}'")
                images_payload[policy_key] = _to_uint8_chw(sample[ds_key])

            prompt = _safe_prompt(sample, args.prompt, tasks_map)
            obs: dict[str, Any] = {"state": state_vec, "images": images_payload}
            if prompt is not None:
                obs["prompt"] = prompt

            out = policy.infer(obs)
            pred_steps = _extract_action_sequence(out.get("actions")) or []
            #import pdb;pdb.set_trace()
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

            ee_error_by_link: dict[str, float] = {}
            ee_pred_points: dict[str, list[float]] = {}
            ee_target_points: dict[str, list[float]] = {}
            loss_ee: float | None = None
            if fk_solvers is not None and fk_maps is not None and pred_vec is not None and target_vec is not None:
                for link_name, solver in fk_solvers.items():
                    mapping = fk_maps.get(link_name)
                    if mapping is None:
                        continue
                    pred_point = _vector_to_fk_point(pred_vec, solver, mapping, args.joint_unit)
                    tgt_point = _vector_to_fk_point(target_vec, solver, mapping, args.joint_unit)
                    if pred_point is not None:
                        ee_pred_points[link_name] = pred_point
                    if tgt_point is not None:
                        ee_target_points[link_name] = tgt_point
                    dist = _compute_l2_distance(pred_point, tgt_point)
                    if dist is not None:
                        ee_error_by_link[link_name] = dist
                if ee_error_by_link:
                    loss_ee = float(np.mean(list(ee_error_by_link.values())))
                    _ = _mean_errors_by_side(ee_error_by_link)

            frame_index = int(_to_scalar(sample["frame_index"], default=float(step))) if "frame_index" in sample else step
            timestamp = float(_to_scalar(sample["timestamp"], default=time.time())) if "timestamp" in sample else time.time()
            episode_index = _extract_optional_int(sample, "episode_index")

            if args.viz == "rerun":
                _log_rerun_scalars(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    loss_l1=loss_l1,
                    loss_l1_left=loss_l1_left,
                    loss_l1_right=loss_l1_right,
                    loss_ee=loss_ee,
                    pred_vec=pred_vec,
                    target_vec=target_vec,
                )
                if args.log_images:
                    for key, img in images_payload.items():
                        rr.log(f"images/{key}", rr.Image(_to_uint8_hwc(img)))

            row = {
                "sample_idx": step,
                "dataset_index": ds_idx,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "timestamp": timestamp,
                "loss_l1": loss_l1,
                "loss_l1_left": loss_l1_left,
                "loss_l1_right": loss_l1_right,
                "loss_ee": loss_ee,
            }
            if csv_writer is not None:
                csv_writer.writerow(row)

            if viewer_url is not None or viewer_record_path is not None:
                ee_left, ee_right = _mean_errors_by_side(ee_error_by_link)
                detail_payload: dict[str, Any] = {
                    "qpos": state_vec.astype(float).tolist(),
                    "action_pred": [float(v) for v in pred_vec] if pred_vec is not None else [],
                    "action_target": [float(v) for v in target_vec] if target_vec is not None else [],
                    "ee_error_by_link": {key: float(val) for key, val in ee_error_by_link.items()},
                    "ee_pred_points": ee_pred_points,
                    "ee_target_points": ee_target_points,
                }
                viewer_frame = {
                    "global_index": ds_idx,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "loss_l1": float(loss_l1) if loss_l1 is not None else None,
                    "loss_l1_left": float(loss_l1_left) if loss_l1_left is not None else None,
                    "loss_l1_right": float(loss_l1_right) if loss_l1_right is not None else None,
                    "loss_l1_global": float(loss_l1) if loss_l1 is not None else None,
                    "loss_ee": float(loss_ee) if loss_ee is not None else None,
                    "ee_error_mean": float(loss_ee) if loss_ee is not None else None,
                    "ee_error_left": ee_left,
                    "ee_error_right": ee_right,
                    "prompt": prompt,
                    "detail": detail_payload,
                }
                if args.web_log_images:
                    viewer_frame["image_data"] = {
                        key: _image_to_data_url(img) for key, img in images_payload.items()
                    }
                viewer_frames_batch.append(viewer_frame)

                for link_name, point in ee_pred_points.items():
                    viewer_traj_pred.setdefault(link_name, []).append(point)
                for link_name, point in ee_target_points.items():
                    viewer_traj_gt.setdefault(link_name, []).append(point)

                if len(viewer_frames_batch) >= max(1, args.web_push_every):
                    _push_viewer_payload(
                        viewer_url,
                        record_path=viewer_record_path,
                        frames=viewer_frames_batch,
                        trajectory_pred=viewer_traj_pred,
                        trajectory_gt=viewer_traj_gt,
                    )
                    viewer_frames_batch = []

            xs.append(step)
            loss_l1_series.append(float(loss_l1) if loss_l1 is not None else np.nan)
            loss_l1_left_series.append(float(loss_l1_left) if loss_l1_left is not None else np.nan)
            loss_l1_right_series.append(float(loss_l1_right) if loss_l1_right is not None else np.nan)
            loss_ee_series.append(float(loss_ee) if loss_ee is not None else np.nan)

            if step % max(1, args.log_every) == 0:
                logging.info(
                    "step=%d idx=%d l1=%s l1_left=%s l1_right=%s ee=%s",
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

        if args.save_plot is not None:
            _plot_loss_curves(
                path=args.save_plot,
                x=xs,
                l1=loss_l1_series,
                l1_left=loss_l1_left_series,
                l1_right=loss_l1_right_series,
                ee=loss_ee_series,
            )
            logging.info("Saved loss plot to %s", args.save_plot)
    finally:
        if (viewer_url is not None or viewer_record_path is not None) and viewer_frames_batch:
            try:
                _push_viewer_payload(
                    viewer_url,
                    record_path=viewer_record_path,
                    frames=viewer_frames_batch,
                    trajectory_pred=viewer_traj_pred,
                    trajectory_gt=viewer_traj_gt,
                )
            except (OSError, urllib_error.URLError, RuntimeError) as exc:
                logging.warning("Failed to flush viewer frames: %s", exc)
        if csv_fp is not None:
            csv_fp.close()
        if fk_tmp is not None:
            fk_tmp.cleanup()
        close_fn = getattr(dataset, "close", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    main()
