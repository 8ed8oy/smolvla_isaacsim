"""Pure-protocol tests for the dual-environment Arena <-> SmolVLA bridge."""

from __future__ import annotations

import io

import numpy as np
import torch

from smolvla_isaac_embed.bridge_protocol import (
    numpy_to_torch_tree,
    read_message,
    summarize_array_tree,
    tensors_to_numpy_tree,
    write_message,
)


def test_bridge_message_round_trip() -> None:
    stream = io.BytesIO()
    payload = {
        "type": "action",
        "action": np.arange(36, dtype=np.float32).reshape(1, 36),
    }

    write_message(stream, payload)
    stream.seek(0)
    decoded = read_message(stream)

    assert decoded["type"] == "action"
    assert isinstance(decoded["action"], np.ndarray)
    assert decoded["action"].dtype == np.float32
    assert decoded["action"].shape == (1, 36)


def test_bridge_message_round_trip_preserves_tuple_and_scalar_arrays() -> None:
    stream = io.BytesIO()
    payload = {
        "type": "mixed",
        "items": (
            np.array(7, dtype=np.int64),
            np.arange(6, dtype=np.float16).reshape(2, 3),
            "ok",
        ),
    }

    write_message(stream, payload)
    stream.seek(0)
    decoded = read_message(stream)

    assert decoded["type"] == "mixed"
    assert isinstance(decoded["items"], tuple)
    assert decoded["items"][0].shape == ()
    assert decoded["items"][0].dtype == np.int64
    assert decoded["items"][0].item() == 7
    assert decoded["items"][1].shape == (2, 3)
    assert decoded["items"][1].dtype == np.float16
    assert decoded["items"][2] == "ok"


def test_tensor_tree_converts_to_numpy_and_back() -> None:
    source = {
        "observation.state": torch.arange(6, dtype=torch.float32).reshape(1, 6),
        "observation.images.robot_pov_cam": torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2),
        "task": "open microwave",
    }

    encoded = tensors_to_numpy_tree(source)
    decoded = numpy_to_torch_tree(encoded)

    assert isinstance(encoded["observation.state"], np.ndarray)
    assert isinstance(encoded["observation.images.robot_pov_cam"], np.ndarray)
    assert decoded["task"] == "open microwave"
    assert isinstance(decoded["observation.state"], torch.Tensor)
    assert isinstance(decoded["observation.images.robot_pov_cam"], torch.Tensor)
    assert torch.equal(decoded["observation.state"], source["observation.state"])
    assert torch.equal(decoded["observation.images.robot_pov_cam"], source["observation.images.robot_pov_cam"])


def test_summarize_array_tree_reports_shape_and_dtype() -> None:
    payload = {
        "action": np.zeros((1, 36), dtype=np.float32),
        "nested": {"image": np.zeros((1, 3, 224, 224), dtype=np.float32)},
    }

    summary = summarize_array_tree(payload)

    assert summary == {
        "action": {"shape": (1, 36), "dtype": "float32"},
        "nested": {"image": {"shape": (1, 3, 224, 224), "dtype": "float32"}},
    }
