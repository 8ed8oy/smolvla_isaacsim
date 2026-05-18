"""针对 action adapter 的最小测试。

目的：
- 验证当前 36 维 GR1 Pink 的 action 路径保持 identity 语义。
- 捕捉 action 维度、rank、描述信息和调试预览的意外改动。

用法：
- 建议在 `./.conda/lerobot-arena` 环境中运行，这个环境同时具备 `torch` 和 `pytest`。
- 在本地检查时用 pytest 运行。
- 典型示例：``python -m pytest smolvla_isaac_embed/tests/test_action_adapter.py -q``。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from smolvla_isaac_embed.adapters import (
    ACTION_ORDER,
    EXPECTED_ACTION_DIM,
    MinimalIsaacActionAdapter,
    describe_action_order,
    validate_action_order,
)


def test_minimal_action_adapter_promotes_1d_input_to_batch() -> None:
    adapter = MinimalIsaacActionAdapter()
    source_action = torch.arange(36, dtype=torch.float64)

    adapted = adapter.adapt(source_action)

    assert adapted.shape == (1, 36)
    assert adapted.dtype == torch.float32
    assert adapted.is_contiguous()
    assert torch.equal(adapted[0], source_action.to(torch.float32))


def test_minimal_action_adapter_preserves_2d_input() -> None:
    adapter = MinimalIsaacActionAdapter()
    source_action = torch.arange(72, dtype=torch.float32).reshape(2, 36)

    adapted = adapter.adapt(source_action)

    assert adapted.shape == (2, 36)
    assert adapted.dtype == torch.float32
    assert adapted.is_contiguous()
    assert torch.equal(adapted, source_action)


def test_minimal_action_adapter_accepts_numpy_input() -> None:
    adapter = MinimalIsaacActionAdapter()
    source_action = np.arange(36, dtype=np.float32)

    adapted = adapter.adapt(source_action)

    assert adapted.shape == (1, 36)
    assert adapted.dtype == torch.float32
    assert adapted.is_contiguous()
    assert torch.equal(adapted[0], torch.from_numpy(source_action))


def test_minimal_action_adapter_rejects_wrong_rank() -> None:
    adapter = MinimalIsaacActionAdapter()

    with pytest.raises(ValueError, match="Expected action rank 1 or 2"):
        adapter.adapt(torch.zeros(1, 2, 36))


def test_minimal_action_adapter_rejects_wrong_dim() -> None:
    adapter = MinimalIsaacActionAdapter()

    with pytest.raises(ValueError, match="Expected action dim 36"):
        adapter.adapt(torch.zeros(1, 35))


def test_minimal_action_adapter_rejects_unsupported_type() -> None:
    adapter = MinimalIsaacActionAdapter()

    with pytest.raises(TypeError, match="numeric sequence"):
        adapter.adapt("not-an-action")


def test_minimal_action_adapter_describe_and_preview_align_with_action_order() -> None:
    adapter = MinimalIsaacActionAdapter()
    source_action = torch.arange(36, dtype=torch.float32)

    description = adapter.describe()
    preview = adapter.preview_action_with_order(source_action, limit=4)

    assert description["adapter_behavior"] == "identity"
    assert description["output_shape"] == "(B, 36)"
    assert description["output_dtype"] == "torch.float32"
    assert description["action_dim"] == EXPECTED_ACTION_DIM == 36
    assert len(ACTION_ORDER) == EXPECTED_ACTION_DIM
    assert description["action_order"] == list(ACTION_ORDER)
    assert description["action_order_validation"]["is_valid"] is True
    assert preview == [
        (ACTION_ORDER[0], 0.0),
        (ACTION_ORDER[1], 1.0),
        (ACTION_ORDER[2], 2.0),
        (ACTION_ORDER[3], 3.0),
    ]


def test_validate_action_order_reports_expected_groups() -> None:
    validation = validate_action_order()

    assert validation["is_valid"] is True
    assert validation["actual_action_dim"] == EXPECTED_ACTION_DIM
    assert validation["has_unique_names"] is True
    assert validation["matches_reference"] is True
    assert validation["group_checks"]["left_eef"]["matches"] is True
    assert validation["group_checks"]["right_eef"]["matches"] is True
    assert validation["group_checks"]["hand_joints"]["matches"] is True


def test_validate_action_order_detects_wrong_order() -> None:
    broken_order = list(ACTION_ORDER)
    broken_order[0], broken_order[1] = broken_order[1], broken_order[0]

    validation = validate_action_order(broken_order)

    assert validation["is_valid"] is False
    assert validation["matches_reference"] is False
    assert validation["group_checks"]["left_eef"]["matches"] is False


def test_describe_action_order_returns_flat_dimension_report() -> None:
    report = describe_action_order()

    assert len(report) == EXPECTED_ACTION_DIM
    assert report[0] == {
        "index": 0,
        "name": ACTION_ORDER[0],
        "group": "left_eef",
        "matches_expected_dim": True,
    }
    assert report[7]["group"] == "right_eef"
    assert report[14]["group"] == "hand_joints"
