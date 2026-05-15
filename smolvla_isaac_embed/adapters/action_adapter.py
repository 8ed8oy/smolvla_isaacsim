"""Minimal Isaac action adapter for SmolVLA integration.

At this stage we only support the GR1 Pink 36-dim continuous action used by the
documented microwave environment path. No reordering, clipping, or safety
filtering happens here yet; the adapter only normalizes shape and validates the
expected action dimension.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

EXPECTED_ACTION_DIM = 36

ACTION_ORDER = (
    "left_eef_pos_x",
    "left_eef_pos_y",
    "left_eef_pos_z",
    "left_eef_quat_w",
    "left_eef_quat_x",
    "left_eef_quat_y",
    "left_eef_quat_z",
    "right_eef_pos_x",
    "right_eef_pos_y",
    "right_eef_pos_z",
    "right_eef_quat_w",
    "right_eef_quat_x",
    "right_eef_quat_y",
    "right_eef_quat_z",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
)


def _as_action_tensor(action: torch.Tensor | Sequence[float]) -> torch.Tensor:
    if isinstance(action, torch.Tensor):
        return action
    return torch.as_tensor(action, dtype=torch.float32)


@dataclass(slots=True)
class MinimalIsaacActionAdapter:
    """Normalize policy actions into the Isaac environment action shape."""

    expected_action_dim: int = EXPECTED_ACTION_DIM

    def adapt(self, action: torch.Tensor | Sequence[float]) -> torch.Tensor:
        action_tensor = _as_action_tensor(action)

        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.dim() != 2:
            raise ValueError(f"Expected action tensor with shape (B, D), got {tuple(action_tensor.shape)}.")
        if action_tensor.shape[-1] != self.expected_action_dim:
            raise ValueError(
                f"Expected action dim {self.expected_action_dim}, got {action_tensor.shape[-1]}."
            )

        return action_tensor.float().contiguous()

    def zero_action(self, *, batch_size: int = 1, device: str | torch.device | None = None) -> torch.Tensor:
        return torch.zeros((batch_size, self.expected_action_dim), dtype=torch.float32, device=device)

    def describe(self) -> dict[str, object]:
        return {
            "action_dim": self.expected_action_dim,
            "action_order": list(ACTION_ORDER),
            "adapter_behavior": "identity",
        }


def adapt_minimal_action(action: torch.Tensor | Sequence[float]) -> torch.Tensor:
    """Convenience wrapper for the current one-path action adapter."""

    return MinimalIsaacActionAdapter().adapt(action)
