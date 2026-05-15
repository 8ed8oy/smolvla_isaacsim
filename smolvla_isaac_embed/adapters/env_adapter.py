"""Minimal Isaac observation adapter for SmolVLA integration.

This adapter intentionally supports only one observation path for now:

- state: ``policy.robot_joint_pos`` -> ``observation.state``
- image: ``camera_obs.robot_pov_cam_rgb`` -> ``observation.images.robot_pov_cam``

The goal is to keep the first embedded integration explicit and easy to inspect
before we broaden the schema surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

SOURCE_STATE_KEY = "robot_joint_pos"
SOURCE_CAMERA_KEY = "robot_pov_cam_rgb"
TARGET_STATE_KEY = "observation.state"
TARGET_IMAGE_KEY = "observation.images.robot_pov_cam"


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping, got {type(value).__name__}.")
    return value


def _require_tensor(value: Any, *, context: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{context} must be a torch.Tensor, got {type(value).__name__}.")
    return value


def _ensure_batched_state(state: torch.Tensor) -> torch.Tensor:
    if state.dim() == 1:
        state = state.unsqueeze(0)
    if state.dim() < 2:
        raise ValueError(f"Expected state tensor with at least 2 dims, got {tuple(state.shape)}.")
    if state.dim() > 2:
        state = state.reshape(state.shape[0], -1)
    return state.float().contiguous()


def _ensure_batched_image(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError(f"Expected image tensor with shape (B, H, W, C), got {tuple(image.shape)}.")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with last dim 3, got {tuple(image.shape)}.")

    image = image.permute(0, 3, 1, 2).contiguous()
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    else:
        image = image.float()
    return image


@dataclass(slots=True)
class MinimalIsaacEnvAdapter:
    """Adapt a raw IsaacLab-Arena observation into the minimal SmolVLA input schema."""

    task: str | None = None

    @property
    def state_key(self) -> str:
        return SOURCE_STATE_KEY

    @property
    def camera_key(self) -> str:
        return SOURCE_CAMERA_KEY

    @property
    def rename_map(self) -> dict[str, str]:
        return {f"observation.images.{SOURCE_CAMERA_KEY}": TARGET_IMAGE_KEY}

    def adapt(self, raw_observation: Mapping[str, Any], task: str | None = None) -> dict[str, Any]:
        observation = _require_mapping(raw_observation, context="raw_observation")
        policy_obs = _require_mapping(observation.get("policy"), context="raw_observation['policy']")
        camera_obs = _require_mapping(observation.get("camera_obs"), context="raw_observation['camera_obs']")

        state = _require_tensor(policy_obs.get(SOURCE_STATE_KEY), context=f"raw_observation['policy']['{SOURCE_STATE_KEY}']")
        image = _require_tensor(
            camera_obs.get(SOURCE_CAMERA_KEY),
            context=f"raw_observation['camera_obs']['{SOURCE_CAMERA_KEY}']",
        )

        adapted = {
            TARGET_STATE_KEY: _ensure_batched_state(state),
            TARGET_IMAGE_KEY: _ensure_batched_image(image),
        }

        task_text = task if task is not None else self.task
        if task_text is not None:
            adapted["task"] = task_text

        return adapted

    def describe(self) -> dict[str, str]:
        return {
            "state_source": f"policy.{SOURCE_STATE_KEY}",
            "state_target": TARGET_STATE_KEY,
            "image_source": f"camera_obs.{SOURCE_CAMERA_KEY}",
            "image_target": TARGET_IMAGE_KEY,
        }


def adapt_minimal_observation(raw_observation: Mapping[str, Any], task: str | None = None) -> dict[str, Any]:
    """Convenience wrapper for the current one-path adapter."""

    return MinimalIsaacEnvAdapter(task=task).adapt(raw_observation)
