#!/usr/bin/env python3
"""Visualize LeRobot dataset frames in Rerun with images, joint curves, and URDF-based end-effector tracks.

Default target dataset:
  data/Task_B/base

Example:
  python scripts/rerun_dataset_viz.py --episode-index 0
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
_TORCH_IMPORT_ERROR: Exception | None = None
try:
    import torch
    import torch.utils.data
except Exception as exc:  # pragma: no cover - runtime dependency
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

_TQDM_IMPORT_ERROR: Exception | None = None
try:
    import tqdm
except Exception as exc:  # pragma: no cover - runtime dependency
    tqdm = None  # type: ignore[assignment]
    _TQDM_IMPORT_ERROR = exc

_RERUN_IMPORT_ERROR: Exception | None = None
try:
    import rerun as rr
except Exception as exc:  # pragma: no cover - runtime dependency
    rr = None  # type: ignore[assignment]
    _RERUN_IMPORT_ERROR = exc

_LEROBOT_IMPORT_ERROR: Exception | None = None
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except Exception as exc:  # pragma: no cover - runtime dependency
    LeRobotDataset = Any  # type: ignore[assignment,misc]
    _LEROBOT_IMPORT_ERROR = exc

try:
    from urdf_parser_py import urdf as urdf_parser
except Exception as exc:  # pragma: no cover - runtime dependency
    urdf_parser = None  # type: ignore[assignment]
    _URDF_IMPORT_ERROR = exc
else:
    _URDF_IMPORT_ERROR = None

_TRIMESH_IMPORT_ERROR: Exception | None = None
try:
    import trimesh
except Exception as exc:  # pragma: no cover - runtime dependency
    trimesh = None  # type: ignore[assignment]
    _TRIMESH_IMPORT_ERROR = exc


DEFAULT_DATASET = "data/Task_B/base"
DEFAULT_URDF_ROOT = Path("/home/ubuntu/Codes/mobile_aloha_sim_ros2")
DEFAULT_URDF_PATH = DEFAULT_URDF_ROOT / "aloha_new_description/urdf/aloha_new.urdf"
_SamplerBase = torch.utils.data.Sampler[int] if torch is not None else object


class EpisodeSampler(_SamplerBase):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        frame_ids: list[int] = []
        for rel_idx, ep_idx in enumerate(dataset.hf_dataset["episode_index"]):
            value = int(ep_idx.item()) if (torch is not None and isinstance(ep_idx, torch.Tensor)) else int(ep_idx)
            if value == int(episode_index):
                frame_ids.append(rel_idx)
        if not frame_ids:
            raise ValueError(f"Episode {episode_index} not found in dataset.")
        self.frame_ids = frame_ids

    def __iter__(self) -> Iterator[int]:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def _to_hwc_uint8(image: Any) -> np.ndarray:
    if torch is not None and isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)

    if arr.ndim != 3:
        raise ValueError(f"Expected image with shape [C,H,W] or [H,W,C], got {arr.shape}")

    if arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if float(np.max(arr)) <= 1.5 else 1.0
        arr = np.clip(arr * scale, 0, 255).round().astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def _extract_scalar(value: Any, default: float) -> float:
    if torch is not None and isinstance(value, torch.Tensor):
        return float(value.item()) if value.numel() == 1 else default
    arr = np.asarray(value)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return default


def _extract_feature_names(dataset: LeRobotDataset, key: str) -> list[str]:
    names = dataset.features.get(key, {}).get("names")
    if not isinstance(names, list):
        return []
    if len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    return [str(name) for name in names]


def _looks_like_magiclab_names(names: list[str]) -> bool:
    required = {"left_arm_joint1", "right_arm_joint1"}
    return required.issubset(set(names))


def _rpy_to_matrix(rpy: list[float] | tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _supports_rr_timeless() -> bool:
    try:
        import inspect

        return "timeless" in inspect.signature(rr.log).parameters
    except Exception:
        return False


_RR_SUPPORTS_TIMELESS = _supports_rr_timeless() if rr is not None else False


def _validate_rerun_module() -> None:
    if rr is None:
        raise RuntimeError("rerun is required. Please install `rerun-sdk`.") from _RERUN_IMPORT_ERROR
    required_attrs = ("init", "log", "set_time", "Image", "Points3D")
    missing = [name for name in required_attrs if not hasattr(rr, name)]
    if missing:
        module_path = getattr(rr, "__file__", "<unknown>")
        raise RuntimeError(
            "Imported `rerun` module is incompatible (missing: "
            f"{', '.join(missing)}). Loaded from: {module_path}. "
            "This usually means a wrong package named `rerun` is installed. "
            "Fix with: `pip uninstall -y rerun && pip install -U rerun-sdk`."
        )


def _rr_log(entity_path: str, data: Any, *, timeless: bool = False) -> None:
    if timeless and _RR_SUPPORTS_TIMELESS:
        rr.log(entity_path, data, timeless=True)
    else:
        rr.log(entity_path, data)


def _log_trimesh_mesh(entity_path: str, mesh: Any) -> None:
    _rr_log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=np.asarray(mesh.vertices),
            triangle_indices=np.asarray(mesh.faces),
            vertex_normals=np.asarray(mesh.vertex_normals) if mesh.vertex_normals is not None else None,
        ),
        timeless=True,
    )


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


class URDFKinematics:
    def __init__(self, urdf_path: Path, root_path: str = "robot") -> None:
        if urdf_parser is None:
            raise RuntimeError("urdf_parser_py is required for URDF kinematics.") from _URDF_IMPORT_ERROR
        if trimesh is None:
            raise RuntimeError("trimesh is required for URDF mesh logging.") from _TRIMESH_IMPORT_ERROR
        self.urdf = urdf_parser.URDF.from_xml_file(str(urdf_path))
        self.urdf_path = urdf_path
        self.root_link = self.urdf.get_root()
        self.root_path = root_path.rstrip("/")
        self.link_names = {link.name for link in self.urdf.links}
        self.mat_name_to_mat = {mat.name: mat for mat in self.urdf.materials}
        self.joint_info: dict[str, dict[str, Any]] = {}
        self.joint_names: list[str] = []
        self._prepare_joints()

    def link_entity_path(self, link: Any) -> str:
        link_names = self.urdf.get_chain(self.root_link, link.name)[0::2]
        return "/".join(link_names)

    def joint_entity_path(self, joint: Any) -> str:
        link_names = self.urdf.get_chain(self.root_link, joint.child)[0::2]
        return "/".join(link_names)

    def _prepare_joints(self) -> None:
        for joint in self.urdf.joints:
            translation = np.zeros(3, dtype=np.float64)
            rotation = np.eye(3, dtype=np.float64)
            if joint.origin is not None:
                if joint.origin.xyz is not None:
                    translation = np.asarray(joint.origin.xyz, dtype=np.float64)
                if joint.origin.rpy is not None:
                    rotation = _rpy_to_matrix(joint.origin.rpy)
            axis = np.asarray(joint.axis if joint.axis is not None else [0.0, 0.0, 1.0], dtype=np.float64)
            self.joint_info[joint.name] = {
                "entity_path": f"{self.root_path}/{self.joint_entity_path(joint)}",
                "base_translation": translation,
                "base_rotation": rotation,
                "axis": axis,
                "type": joint.type,
            }
            if joint.type != "fixed":
                self.joint_names.append(joint.name)

    def log_static_geometry(self) -> None:
        _rr_log(self.root_path, rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
        for info in self.joint_info.values():
            _rr_log(
                info["entity_path"],
                rr.Transform3D(
                    translation=np.asarray(info["base_translation"], dtype=np.float64),
                    mat3x3=np.asarray(info["base_rotation"], dtype=np.float64),
                ),
                timeless=True,
            )

        for link in self.urdf.links:
            self._log_link_mesh(link)

    def _log_link_mesh(self, link: Any) -> None:
        visuals = list(link.visuals) if getattr(link, "visuals", None) else []
        if not visuals and getattr(link, "visual", None) is not None:
            visuals = [link.visual]
        if not visuals:
            return

        entity_path = self.link_entity_path(link)
        for vis_idx, visual in enumerate(visuals):
            vis_path = f"{self.root_path}/{entity_path}/visual_{vis_idx}"
            self._log_visual_mesh(vis_path, visual)

    def _log_visual_mesh(self, entity_path: str, visual: Any) -> None:
        transform = np.eye(4, dtype=np.float64)
        if visual.origin is not None:
            if visual.origin.xyz is not None:
                transform[:3, 3] = np.asarray(visual.origin.xyz, dtype=np.float64)
            if visual.origin.rpy is not None:
                transform[:3, :3] = _rpy_to_matrix(visual.origin.rpy)

        geom = visual.geometry
        mesh_or_scene: Any
        if isinstance(geom, urdf_parser.Mesh):
            resolved = _resolve_urdf_resource(geom.filename, self.urdf_path, None)
            if resolved is None:
                logging.warning("Skip unresolved mesh resource: %s", geom.filename)
                return
            mesh_or_scene = trimesh.load_mesh(str(resolved))
            if geom.scale is not None:
                scale = np.asarray(geom.scale, dtype=np.float64)
                transform[:3, :3] *= scale
        elif isinstance(geom, urdf_parser.Box):
            mesh_or_scene = trimesh.creation.box(extents=np.asarray(geom.size, dtype=np.float64))
        elif isinstance(geom, urdf_parser.Cylinder):
            mesh_or_scene = trimesh.creation.cylinder(radius=float(geom.radius), height=float(geom.length))
        elif isinstance(geom, urdf_parser.Sphere):
            mesh_or_scene = trimesh.creation.icosphere(radius=float(geom.radius))
        else:
            return

        mesh_or_scene.apply_transform(transform)
        if isinstance(mesh_or_scene, trimesh.Scene):
            for mesh_idx, mesh in enumerate(mesh_or_scene.dump()):
                _log_trimesh_mesh(f"{entity_path}/{mesh_idx}", mesh)
        else:
            _log_trimesh_mesh(entity_path, mesh_or_scene)

    def has_link(self, link_name: str) -> bool:
        return link_name in self.link_names

    def log_joints(self, joint_values: dict[str, float]) -> None:
        for joint_name, info in self.joint_info.items():
            value = joint_values.get(joint_name)
            if value is None:
                continue
            translation = np.array(info["base_translation"], dtype=np.float64)
            rotation = np.array(info["base_rotation"], dtype=np.float64)
            axis = np.array(info["axis"], dtype=np.float64)
            if info["type"] in ("revolute", "continuous"):
                rotation = rotation @ _axis_angle_to_matrix(axis, float(value))
            elif info["type"] == "prismatic":
                norm = np.linalg.norm(axis)
                if norm > 0.0:
                    translation = translation + axis / norm * float(value)

            rr.log(
                info["entity_path"],
                rr.Transform3D(translation=translation, mat3x3=rotation),
            )

    def compute_link_position(self, joint_values: dict[str, float], link_name: str) -> np.ndarray | None:
        if link_name not in self.link_names:
            return None
        chain = self.urdf.get_chain(self.root_link, link_name)
        joint_names = chain[1::2]
        transform = np.eye(4, dtype=np.float64)
        for joint_name in joint_names:
            info = self.joint_info.get(joint_name)
            if info is None:
                continue
            value = float(joint_values.get(joint_name, 0.0))
            origin = np.eye(4, dtype=np.float64)
            origin[:3, :3] = np.asarray(info["base_rotation"], dtype=np.float64)
            origin[:3, 3] = np.asarray(info["base_translation"], dtype=np.float64)

            motion = np.eye(4, dtype=np.float64)
            axis = np.asarray(info["axis"], dtype=np.float64)
            if info["type"] in ("revolute", "continuous"):
                motion[:3, :3] = _axis_angle_to_matrix(axis, value)
            elif info["type"] == "prismatic":
                norm = np.linalg.norm(axis)
                if norm > 0:
                    motion[:3, 3] = axis / norm * value

            transform = transform @ origin @ motion
        point = transform @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return point[:3]


def _build_urdf_to_dataset_map(dataset_joint_names: list[str], urdf_joint_names: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    if dataset_joint_names and any(name in urdf_joint_names for name in dataset_joint_names):
        for idx, name in enumerate(dataset_joint_names):
            if name in urdf_joint_names:
                mapping[name] = idx
        return mapping

    if _looks_like_magiclab_names(dataset_joint_names):
        left_arm = [f"left_arm_joint{i}" for i in range(1, 8)]
        right_arm = [f"right_arm_joint{i}" for i in range(1, 8)]
        for i, ds_name in enumerate(left_arm):
            urdf_name = f"fl_joint{i + 1}"
            if ds_name in dataset_joint_names and urdf_name in urdf_joint_names:
                mapping[urdf_name] = dataset_joint_names.index(ds_name)
        for i, ds_name in enumerate(right_arm):
            urdf_name = f"fr_joint{i + 1}"
            if ds_name in dataset_joint_names and urdf_name in urdf_joint_names:
                mapping[urdf_name] = dataset_joint_names.index(ds_name)
        return mapping

    # Kai0 Task_B (agilex) fallback: state/action are 14-dim without names.
    if len(dataset_joint_names) >= 14 or not dataset_joint_names:
        defaults = [f"fl_joint{i}" for i in range(1, 8)] + [f"fr_joint{i}" for i in range(1, 8)]
        for ds_idx, urdf_name in enumerate(defaults):
            if urdf_name in urdf_joint_names:
                mapping[urdf_name] = ds_idx

    return mapping


def _gripper_to_prismatic(value: float) -> float:
    v = abs(float(value))
    if v <= 0.04:
        return v
    if v <= 1.0:
        return v * 0.04
    return min(v, 40.0) * 0.001


def _joint_map_from_state(
    state_values: np.ndarray,
    urdf_joint_names: list[str],
    urdf_to_dataset: dict[str, int],
    add_symmetric_gripper: bool = True,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for urdf_name in urdf_joint_names:
        ds_idx = urdf_to_dataset.get(urdf_name)
        if ds_idx is None or ds_idx >= len(state_values):
            continue
        out[urdf_name] = float(state_values[ds_idx])

    if not add_symmetric_gripper:
        return out

    # For piper gripper, one dataset dim controls two opposite prismatic joints.
    for side in ("fl", "fr"):
        j7 = f"{side}_joint7"
        j8 = f"{side}_joint8"
        if j7 in out and j8 in urdf_joint_names and j8 not in out:
            disp = _gripper_to_prismatic(out[j7])
            out[j7] = disp
            out[j8] = -disp
    return out


def _select_tip_link(kin: URDFKinematics, candidates: list[str]) -> str | None:
    for name in candidates:
        if kin.has_link(name):
            return name
    return None


def _log_scalars(prefix: str, values: np.ndarray) -> None:
    for idx, val in enumerate(values.reshape(-1)):
        rr.log(f"{prefix}/{idx}", rr.Scalars(float(val)))


def _visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    urdf_path: Path | None,
    urdf_root: Path | None,
    state_key: str,
    action_key: str,
    ee_links: tuple[str, str],
    batch_size: int,
    num_workers: int,
    mode: str,
    web_port: int,
    ws_port: int,
    save: bool,
    output_dir: Path | None,
    playback_sleep_s: float,
    max_frames: int,
    show_skeleton: bool,
    show_ee_traj: bool,
) -> Path | None:
    if save and output_dir is None:
        raise ValueError("--output-dir is required when --save is set.")

    repo_id = dataset.repo_id
    sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    spawn_local_viewer = (mode == "local") and (not save)
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)
    gc.collect()
    if mode == "distant":
        rr.serve_web_viewer(open_browser=False, web_port=web_port)

    camera_keys = list(getattr(dataset.meta, "camera_keys", []))
    if not camera_keys:
        camera_keys = [key for key in dataset.features.keys() if key.startswith("observation.images.")]

    kin: URDFKinematics | None = None
    urdf_to_dataset: dict[str, int] = {}
    left_ee, right_ee = ee_links
    if urdf_path is not None:
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        with tempfile.TemporaryDirectory(prefix="kai0_rerun_fk_") as tempdir:
            fk_urdf_path = _rewrite_urdf_for_fk(urdf_path, urdf_root, Path(tempdir))
            kin = URDFKinematics(fk_urdf_path)
            kin.log_static_geometry()
            ds_names = _extract_feature_names(dataset, state_key) or _extract_feature_names(dataset, action_key)
            urdf_to_dataset = _build_urdf_to_dataset_map(ds_names, kin.joint_names)
            state_shape = dataset.features.get(state_key, {}).get("shape", [])
            if (
                not urdf_to_dataset
                and isinstance(state_shape, (list, tuple))
                and len(state_shape) > 0
                and int(state_shape[0]) == 14
            ):
                urdf_to_dataset = _build_urdf_to_dataset_map([], kin.joint_names)
            if not urdf_to_dataset:
                logging.warning("Failed to build URDF joint mapping; ee points will be skipped.")
            _run_loop(
                dataloader=dataloader,
                camera_keys=camera_keys,
                state_key=state_key,
                action_key=action_key,
                kin=kin,
                urdf_to_dataset=urdf_to_dataset,
                left_ee=left_ee,
                right_ee=right_ee,
                playback_sleep_s=playback_sleep_s,
                max_frames=max_frames,
                show_skeleton=show_skeleton,
                show_ee_traj=show_ee_traj,
            )
    else:
        _run_loop(
            dataloader=dataloader,
            camera_keys=camera_keys,
            state_key=state_key,
            action_key=action_key,
            kin=None,
            urdf_to_dataset={},
            left_ee=left_ee,
            right_ee=right_ee,
            playback_sleep_s=playback_sleep_s,
            max_frames=max_frames,
            show_skeleton=show_skeleton,
            show_ee_traj=show_ee_traj,
        )

    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{str(repo_id).replace('/', '_')}_episode_{episode_index}.rrd"
        rr.save(out_path)
        return out_path

    if mode == "distant":
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            logging.info("Stopped by user.")
    return None


def _run_loop(
    dataloader: torch.utils.data.DataLoader,
    camera_keys: list[str],
    state_key: str,
    action_key: str,
    kin: URDFKinematics | None,
    urdf_to_dataset: dict[str, int],
    left_ee: str,
    right_ee: str,
    playback_sleep_s: float,
    max_frames: int,
    show_skeleton: bool,
    show_ee_traj: bool,
) -> None:
    left_traj: list[np.ndarray] = []
    right_traj: list[np.ndarray] = []
    frame_count = 0
    has_left_link = kin.has_link(left_ee) if kin is not None else False
    has_right_link = kin.has_link(right_ee) if kin is not None else False
    if kin is not None:
        if not has_left_link:
            logging.warning("Left ee link not found in URDF: %s", left_ee)
        if not has_right_link:
            logging.warning("Right ee link not found in URDF: %s", right_ee)

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        batch_size = len(batch["index"])
        for i in range(batch_size):
            frame_value = batch["frame_index"][i] if "frame_index" in batch else i
            ts_value = batch["timestamp"][i] if "timestamp" in batch else float(i)
            frame_index = int(_extract_scalar(frame_value, default=float(i)))
            timestamp = float(_extract_scalar(ts_value, default=float(i)))
            rr.set_time("frame_index", sequence=frame_index)
            rr.set_time("timestamp", timestamp=timestamp)

            for key in camera_keys:
                if key not in batch:
                    continue
                rr.log(key, rr.Image(_to_hwc_uint8(batch[key][i])))

            if action_key in batch:
                values = batch[action_key][i].detach().cpu().numpy()
                _log_scalars("action", values)
            if state_key in batch:
                values = batch[state_key][i].detach().cpu().numpy()
                _log_scalars("state", values)

            if kin is not None and urdf_to_dataset and state_key in batch:
                state_values = batch[state_key][i].detach().cpu().numpy().reshape(-1)
                joint_map = _joint_map_from_state(state_values, kin.joint_names, urdf_to_dataset, add_symmetric_gripper=True)
                kin.log_joints(joint_map)

                if has_left_link:
                    left = kin.compute_link_position(joint_map, left_ee)
                    if left is not None:
                        left_traj.append(left)
                        rr.log("robot/ee_left", rr.Points3D([left], colors=[(255, 80, 80)]))
                        if show_ee_traj and len(left_traj) >= 2:
                            rr.log("robot/ee_left/traj", rr.LineStrips3D([left_traj], colors=[(255, 80, 80)]))

                if has_right_link:
                    right = kin.compute_link_position(joint_map, right_ee)
                    if right is not None:
                        right_traj.append(right)
                        rr.log("robot/ee_right", rr.Points3D([right], colors=[(80, 120, 255)]))
                        if show_ee_traj and len(right_traj) >= 2:
                            rr.log("robot/ee_right/traj", rr.LineStrips3D([right_traj], colors=[(80, 120, 255)]))

                if show_skeleton:
                    left_chain = [f"fl_link{i}" for i in range(1, 9)]
                    right_chain = [f"fr_link{i}" for i in range(1, 9)]
                    left_points = [kin.compute_link_position(joint_map, name) for name in left_chain if kin.has_link(name)]
                    right_points = [kin.compute_link_position(joint_map, name) for name in right_chain if kin.has_link(name)]
                    left_points = [p for p in left_points if p is not None]
                    right_points = [p for p in right_points if p is not None]
                    if len(left_points) >= 2:
                        rr.log("robot/arm_left", rr.LineStrips3D([left_points], colors=[(255, 160, 90)]))
                    if len(right_points) >= 2:
                        rr.log("robot/arm_right", rr.LineStrips3D([right_points], colors=[(90, 180, 255)]))

            if playback_sleep_s > 0:
                time.sleep(playback_sleep_s)
            frame_count += 1
            if max_frames > 0 and frame_count >= max_frames:
                return


def _resolve_dataset_repo_id(path_or_repo: str) -> str:
    path = Path(path_or_repo)
    if path.exists():
        return str(path.resolve())
    return path_or_repo


def _load_first_task(meta_tasks_path: Path) -> str | None:
    if not meta_tasks_path.exists():
        return None
    try:
        first = meta_tasks_path.read_text(encoding="utf-8").strip().splitlines()[0]
        return json.loads(first).get("task")
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun visualization for LeRobot dataset with URDF-based end-effector trajectories."
    )
    parser.add_argument(
        "--dataset-repo-id",
        "--repo-id",
        dest="dataset_repo_id",
        default=DEFAULT_DATASET,
        help="Dataset repo_id or local dataset path.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Optional LeRobot dataset root.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--mode", choices=["local", "distant"], default="local")
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--ws-port", type=int, default=9087)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rerun"))
    parser.add_argument("--playback-sleep-s", type=float, default=0.0, help="Optional sleep per frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 means full episode).")
    parser.add_argument("--show-skeleton", action="store_true", help="Also render arm skeleton line-strips.")
    parser.add_argument("--show-ee-traj", action="store_true", help="Show ee history trajectory line-strips.")
    parser.add_argument("--no-urdf", action="store_true", help="Disable URDF-based ee/skeleton visualization.")
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--urdf-root", type=Path, default=DEFAULT_URDF_ROOT)
    parser.add_argument(
        "--ee-links",
        default="fl_link8,fr_link8",
        help="End-effector links in URDF: 'left,right'.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    _validate_rerun_module()
    if torch is None:
        raise RuntimeError("torch is required in this environment.") from _TORCH_IMPORT_ERROR
    if tqdm is None:
        raise RuntimeError("tqdm is required in this environment.") from _TQDM_IMPORT_ERROR
    if _LEROBOT_IMPORT_ERROR is not None:
        raise RuntimeError("lerobot is required in this environment.") from _LEROBOT_IMPORT_ERROR
    if not args.no_urdf and urdf_parser is None:
        raise RuntimeError("urdf_parser_py is required for URDF visualization.") from _URDF_IMPORT_ERROR
    if not args.no_urdf and trimesh is None:
        raise RuntimeError("trimesh is required for URDF mesh visualization.") from _TRIMESH_IMPORT_ERROR
    repo_id = _resolve_dataset_repo_id(args.dataset_repo_id)
    ee_parts = [s.strip() for s in args.ee_links.split(",") if s.strip()]
    if len(ee_parts) != 2:
        raise ValueError("--ee-links must provide two links: left,right")

    dataset = LeRobotDataset(repo_id=repo_id, root=args.root, episodes=[args.episode_index])
    if Path(repo_id).exists():
        task = _load_first_task(Path(repo_id) / "meta" / "tasks.jsonl")
        if task:
            logging.info("Task: %s", task)
    logging.info("Loaded dataset=%s episode=%s", repo_id, args.episode_index)
    logging.info("Camera keys: %s", list(getattr(dataset.meta, "camera_keys", [])))

    urdf_path = None if args.no_urdf else args.urdf_path.expanduser().resolve()
    urdf_root = None if args.no_urdf else args.urdf_root.expanduser().resolve()
    out_path = _visualize_dataset(
        dataset=dataset,
        episode_index=args.episode_index,
        urdf_path=urdf_path,
        urdf_root=urdf_root,
        state_key=args.state_key,
        action_key=args.action_key,
        ee_links=(ee_parts[0], ee_parts[1]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.mode,
        web_port=args.web_port,
        ws_port=args.ws_port,
        save=args.save,
        output_dir=args.output_dir,
        playback_sleep_s=max(0.0, args.playback_sleep_s),
        max_frames=max(0, args.max_frames),
        show_skeleton=bool(args.show_skeleton),
        show_ee_traj=bool(args.show_ee_traj),
    )
    if out_path is not None:
        logging.info("Saved Rerun file: %s", out_path)


if __name__ == "__main__":
    main()
