"""针对 action adapter 的最小测试。

目的：
- 验证当前 36 维 GR1 Pink 的 action 路径保持恒等映射。
- 捕捉 action 维度检查或文档顺序的意外改动。

用法：
- 在本地检查时用 pytest 运行。
- 典型示例：``python -m pytest smolvla_isaac_embed/tests/test_action_adapter.py -q``。
"""

from __future__ import annotations

import torch
import pytest

from smolvla_isaac_embed.adapters import ACTION_ORDER, MinimalIsaacActionAdapter


def test_minimal_action_adapter_preserves_36d_order() -> None:
    adapter = MinimalIsaacActionAdapter()
    source_action = torch.arange(36, dtype=torch.float32)

    adapted = adapter.adapt(source_action)

    assert len(ACTION_ORDER) == 36
    assert ACTION_ORDER[0] == "left_eef_pos_x"
    assert ACTION_ORDER[-1] == "R_thumb_distal_joint"
    assert adapted.shape == (1, 36)
    assert adapted.dtype == torch.float32
    assert torch.equal(adapted[0], source_action)


def test_minimal_action_adapter_rejects_wrong_dim() -> None:
    adapter = MinimalIsaacActionAdapter()

    with pytest.raises(ValueError, match="Expected action dim 36"):
        adapter.adapt(torch.zeros(1, 35))
