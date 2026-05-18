"""Formal minimal action adapter for SmolVLA -> Isaac integration.

This module intentionally keeps the current GR1 Pink action path as an identity
adapter:
- no reordering
- no clipping
- no safety filtering

The adapter only normalizes rank, validates the expected action dimension, and
casts the result to a stable floating-point output dtype while preserving the
caller-provided device.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
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

LEFT_EEF_ACTION_ORDER = ACTION_ORDER[:7]
RIGHT_EEF_ACTION_ORDER = ACTION_ORDER[7:14]
HAND_JOINT_ACTION_ORDER = ACTION_ORDER[14:]

ActionInput = torch.Tensor | np.ndarray | Sequence[float] | Sequence[Sequence[float]]


def validate_action_order(
    action_order: Sequence[str] = ACTION_ORDER,
    *,
    expected_action_dim: int = EXPECTED_ACTION_DIM,
) -> dict[str, object]:
    """Validate the current 36-dim GR1 Pink action order contract.

    The current minimal path assumes:
    - 7 left end-effector target values
    - 7 right end-effector target values
    - 22 hand joint targets
    """

    action_order_tuple = tuple(str(name) for name in action_order)
    is_expected_dim = len(action_order_tuple) == expected_action_dim
    has_unique_names = len(set(action_order_tuple)) == len(action_order_tuple)
    left_matches = action_order_tuple[:7] == LEFT_EEF_ACTION_ORDER
    right_matches = action_order_tuple[7:14] == RIGHT_EEF_ACTION_ORDER
    hand_matches = action_order_tuple[14:] == HAND_JOINT_ACTION_ORDER
    matches_reference = action_order_tuple == ACTION_ORDER

    return {
        "is_valid": bool(
            is_expected_dim
            and has_unique_names
            and left_matches
            and right_matches
            and hand_matches
            and matches_reference
        ),
        "expected_action_dim": int(expected_action_dim),
        "actual_action_dim": int(len(action_order_tuple)),
        "has_unique_names": bool(has_unique_names),
        "matches_reference": bool(matches_reference),
        "group_checks": {
            "left_eef": {
                "expected_span": [0, 6],
                "actual_names": list(action_order_tuple[:7]),
                "expected_names": list(LEFT_EEF_ACTION_ORDER),
                "matches": bool(left_matches),
            },
            "right_eef": {
                "expected_span": [7, 13],
                "actual_names": list(action_order_tuple[7:14]),
                "expected_names": list(RIGHT_EEF_ACTION_ORDER),
                "matches": bool(right_matches),
            },
            "hand_joints": {
                "expected_span": [14, 35],
                "actual_names": list(action_order_tuple[14:]),
                "expected_names": list(HAND_JOINT_ACTION_ORDER),
                "matches": bool(hand_matches),
            },
        },
    }


def describe_action_order(
    action_order: Sequence[str] = ACTION_ORDER,
    *,
    expected_action_dim: int = EXPECTED_ACTION_DIM,
) -> list[dict[str, object]]:
    """Return a flat per-dimension action order report for inspection."""

    action_order_tuple = tuple(str(name) for name in action_order)
    report: list[dict[str, object]] = []
    for index, name in enumerate(action_order_tuple):
        if index < 7:
            group = "left_eef"
        elif index < 14:
            group = "right_eef"
        else:
            group = "hand_joints"
        report.append(
            {
                "index": int(index),
                "name": name,
                "group": group,
                "matches_expected_dim": bool(len(action_order_tuple) == expected_action_dim),
            }
        )
    return report


def _as_action_tensor(action: ActionInput) -> torch.Tensor:
    """Convert supported action inputs into a tensor without changing device.

    Supported inputs are:
    - ``torch.Tensor``
    - ``numpy.ndarray``
    - numeric 1D sequences
    - numeric 2D sequences representing batched actions
    """

    if isinstance(action, torch.Tensor):
        action_tensor = action
    elif isinstance(action, np.ndarray):
        action_tensor = torch.from_numpy(action)
    elif isinstance(action, Sequence) and not isinstance(action, (str, bytes, bytearray)):
        action_tensor = torch.as_tensor(action)
    else:
        raise TypeError(
            "Action must be a torch.Tensor, numpy.ndarray, or a numeric sequence; "
            f"got {type(action).__name__}."
        )

    if action_tensor.dtype == torch.bool:
        raise TypeError("Boolean action values are not supported.")
    if action_tensor.is_complex():
        raise TypeError("Complex-valued actions are not supported.")

    return action_tensor


@dataclass(slots=True)
class MinimalIsaacActionAdapter:
    """Identity adapter for the current GR1 Pink 36-dim action path.

    Input contract:
    - accepts a single action vector with shape ``(36,)``
    - accepts a batched action tensor with shape ``(B, 36)``
    - accepts tensors, numpy arrays, or numeric Python sequences

    Output contract:
    - always returns a contiguous ``torch.float32`` tensor
    - single actions are promoted to shape ``(1, 36)``
    - batched actions keep their batch dimension unchanged
    - the caller's device is preserved

    Behavior contract:
    - identity adapter only
    - no action reordering
    - no clipping
    - no safety filtering
    """

    expected_action_dim: int = EXPECTED_ACTION_DIM
    expected_action_dtype: torch.dtype = torch.float32
    action_order: tuple[str, ...] = ACTION_ORDER

    def adapt(self, action: ActionInput) -> torch.Tensor:
        action_tensor = _as_action_tensor(action)

        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
        elif action_tensor.dim() != 2:
            raise ValueError(
                "Expected action rank 1 or 2, "
                f"got rank {action_tensor.dim()} with shape {tuple(action_tensor.shape)}."
            )

        if action_tensor.shape[-1] != self.expected_action_dim:
            raise ValueError(
                f"Expected action dim {self.expected_action_dim}, got {action_tensor.shape[-1]}."
            )

        return action_tensor.to(dtype=self.expected_action_dtype).contiguous()

    def zero_action(self, *, batch_size: int = 1, device: str | torch.device | None = None) -> torch.Tensor:
        return torch.zeros(
            (batch_size, self.expected_action_dim),
            dtype=self.expected_action_dtype,
            device=device,
        )

    def describe(self) -> dict[str, object]:
        return {
            "adapter_type": self.__class__.__name__,
            "adapter_behavior": "identity",
            "supported_input_types": [
                "torch.Tensor",
                "numpy.ndarray",
                "Sequence[float]",
                "Sequence[Sequence[float]]",
            ],
            "supported_ranks": [1, 2],
            "input_rank_behavior": {
                "1": "promote_to_batch",
                "2": "preserve_batch",
            },
            "output_shape": "(B, 36)",
            "output_dtype": str(self.expected_action_dtype),
            "output_contiguous": True,
            "device_policy": "preserve_input_device",
            "action_dim": self.expected_action_dim,
            "action_order": list(self.action_order),
            "action_order_validation": validate_action_order(
                self.action_order,
                expected_action_dim=self.expected_action_dim,
            ),
        }

    def preview_action_with_order(
        self,
        action: ActionInput,
        *,
        limit: int = 8,
    ) -> list[tuple[str, float]]:
        """Return the leading action values paired with ``ACTION_ORDER`` names."""

        if limit < 0:
            raise ValueError("limit must be non-negative.")

        adapted_action = self.adapt(action)
        if adapted_action.shape[0] == 0 or limit == 0:
            return []

        preview_values = adapted_action[0, : min(limit, adapted_action.shape[-1])]
        preview_names = self.action_order[: preview_values.shape[0]]
        preview_values_list = preview_values.detach().cpu().tolist()
        return [(name, float(value)) for name, value in zip(preview_names, preview_values_list)]


def adapt_minimal_action(action: ActionInput) -> torch.Tensor:
    """Convenience wrapper around the current identity action adapter."""

    return MinimalIsaacActionAdapter().adapt(action)
