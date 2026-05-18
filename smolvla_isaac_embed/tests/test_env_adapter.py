"""针对 observation adapter 的最小测试。

目的：
- 验证最小观测路径是否能把 Isaac 的键正确映射到 policy 键。
- 捕捉 state 形状处理或相机张量转换中的回归。

用法：
- 建议在 `./.conda/lerobot-arena` 环境中运行，这个环境同时具备 `torch` 和 `pytest`。
- 在本地检查时用 pytest 运行。
- 典型示例：``python -m pytest smolvla_isaac_embed/tests/test_env_adapter.py -q``。
"""

from __future__ import annotations

import torch

from smolvla_isaac_embed.adapters import MinimalIsaacEnvAdapter


def test_minimal_env_adapter_maps_joint_state_and_camera() -> None:
    adapter = MinimalIsaacEnvAdapter(task="open microwave")
    raw_observation = {
        "policy": {
            "robot_joint_pos": torch.arange(54, dtype=torch.float32).reshape(1, 54),
        },
        "camera_obs": {
            "robot_pov_cam_rgb": torch.tensor(
                [[[[0, 64, 255], [255, 128, 0]]]],
                dtype=torch.uint8,
            ),
        },
    }

    adapted = adapter.adapt(raw_observation)

    assert set(adapted) == {"observation.state", "observation.images.robot_pov_cam", "task"}
    assert adapted["task"] == "open microwave"

    state = adapted["observation.state"]
    assert state.shape == (1, 54)
    assert state.dtype == torch.float32
    assert torch.equal(state, raw_observation["policy"]["robot_joint_pos"])

    image = adapted["observation.images.robot_pov_cam"]
    assert image.shape == (1, 3, 1, 2)
    assert image.dtype == torch.float32
    assert torch.isclose(image[0, 0, 0, 1], torch.tensor(1.0))
    assert torch.isclose(image[0, 1, 0, 0], torch.tensor(64.0 / 255.0))
    assert torch.isclose(image[0, 2, 0, 0], torch.tensor(1.0))
