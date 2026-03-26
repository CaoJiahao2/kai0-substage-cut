"""Policy transforms for the MagicG1 robot."""

import dataclasses
from typing import ClassVar

import numpy as np
import torch

import openpi.models.model as _model
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class MagicG1Inputs(transforms.DataTransformFn):
    """Inputs for the MagicG1 policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. For normal pi05
      training, names must be exactly the keys of required_rename_map. For advantage
      estimator, optional_rename_map keys may be included as well.
    - state: [N]
    - actions: [action_horizon, N]
    """

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    required_rename_map = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb",
    }
    optional_rename_map = {
        "his_-100_top_head": "base_-100_rgb",
        "his_-100_hand_left": "left_wrist_-100_rgb",
        "his_-100_hand_right": "right_wrist_-100_rgb",
    }

    all_rename_map = {**required_rename_map, **optional_rename_map}

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = tuple(required_rename_map.keys())
    EXTRA_CAMERAS: ClassVar[tuple[str, ...]] = tuple(optional_rename_map.keys())

    mask_state: bool = False

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type in (_model.ModelType.PI0, _model.ModelType.PI0_RTC)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS) - set(self.EXTRA_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        state = transforms.pad_to_dim(data["state"], self.action_dim)
        state = state.squeeze()

        images = {}
        image_masks = {}
        for camera in self.EXPECTED_CAMERAS + self.EXTRA_CAMERAS:
            if camera in in_images:
                img = in_images[camera]
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[self.all_rename_map[camera]] = img
                image_masks[self.all_rename_map[camera]] = np.True_
            elif camera not in in_images and camera in self.EXTRA_CAMERAS:
                continue
            else:
                raise ValueError(f"Camera {camera} not found in data")

        state = np.where(state > np.pi, 0, state)
        state = np.where(state < -np.pi, 0, state)

        masked_state = np.zeros_like(state) if self.mask_state else state
        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": masked_state,
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            actions = np.where(actions > np.pi, 0, actions)
            actions = np.where(actions < -np.pi, 0, actions)
            if mask_padding:
                action_mask = np.ones_like(actions, dtype=bool)
                action_mask[:, self.action_dim:] = False
                inputs["action_mask"] = action_mask
            inputs["actions"] = actions.squeeze()

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        for key in ("frame_index", "episode_length", "progress", "image_original", "episode_index"):
            if key in data:
                inputs[key] = data[key]

        def _to_tensor(x, default=None):
            if x is None and default is not None:
                return default
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            if isinstance(x, torch.Tensor):
                return x.detach().clone()
            raise NotImplementedError(f"Unsupported type: {type(x)}")

        if "action_advantage" in data:
            inputs["action_advantage"] = _to_tensor(data["action_advantage"], default=torch.tensor(1.0))
        if "action_advantage_original" in data:
            inputs["action_advantage_original"] = _to_tensor(data["action_advantage_original"])
        return inputs


@dataclasses.dataclass(frozen=True)
class MagicG1Outputs(transforms.DataTransformFn):
    """Outputs for the MagicG1 policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :30])}
