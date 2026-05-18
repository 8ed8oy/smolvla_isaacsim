"""Shared bridge protocol for the dual-environment Isaac <-> SmolVLA rollout.

The bridge intentionally stays lightweight and local-process only:

- parent process: Python 3.11 Arena / Isaac rollout loop
- worker process: Python 3.12 lerobot / SmolVLA policy inference

Messages are exchanged over stdio using a length-prefixed pickle frame. The
payload only contains builtins, ``numpy.ndarray`` objects, and small metadata
dicts so the protocol stays explicit and easy to debug.
"""

from __future__ import annotations

import pickle
import struct
from collections.abc import Mapping
from typing import Any, BinaryIO

import numpy as np
import torch

FRAME_HEADER_STRUCT = struct.Struct(">Q")
NDARRAY_MARKER = "__bridge_ndarray__"
TUPLE_MARKER = "__bridge_tuple__"


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if chunk is None or len(chunk) == 0:
            raise EOFError("Unexpected EOF while reading bridge message.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _encode_for_wire(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            NDARRAY_MARKER: True,
            "dtype": value.dtype.str,
            "shape": tuple(int(dim) for dim in value.shape),
            "data": value.tobytes(order="C"),
        }
    if isinstance(value, Mapping):
        return {key: _encode_for_wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_for_wire(item) for item in value]
    if isinstance(value, tuple):
        return {
            TUPLE_MARKER: [_encode_for_wire(item) for item in value],
        }
    return value


def _decode_from_wire(value: Any) -> Any:
    if isinstance(value, Mapping):
        if value.get(NDARRAY_MARKER) is True:
            dtype = np.dtype(value["dtype"])
            shape = tuple(int(dim) for dim in value["shape"])
            data = value["data"]
            array = np.frombuffer(data, dtype=dtype)
            return array.reshape(shape).copy()
        if TUPLE_MARKER in value:
            return tuple(_decode_from_wire(item) for item in value[TUPLE_MARKER])
        return {key: _decode_from_wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_from_wire(item) for item in value]
    return value


def write_message(stream: BinaryIO, payload: Mapping[str, Any]) -> None:
    encoded_payload = _encode_for_wire(dict(payload))
    encoded = pickle.dumps(encoded_payload, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(FRAME_HEADER_STRUCT.pack(len(encoded)))
    stream.write(encoded)
    stream.flush()


def read_message(stream: BinaryIO) -> dict[str, Any]:
    header = stream.read(FRAME_HEADER_STRUCT.size)
    if header is None or len(header) == 0:
        raise EOFError("Bridge stream closed before a message header was read.")
    if len(header) != FRAME_HEADER_STRUCT.size:
        raise EOFError("Bridge stream ended mid-header.")

    (payload_size,) = FRAME_HEADER_STRUCT.unpack(header)
    payload = _read_exact(stream, payload_size)
    decoded = _decode_from_wire(pickle.loads(payload))  # nosec: local child-process protocol only
    if not isinstance(decoded, dict):
        raise TypeError(f"Bridge payload must decode to dict, got {type(decoded).__name__}.")
    return decoded


def tensors_to_numpy_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Mapping):
        return {key: tensors_to_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [tensors_to_numpy_tree(item) for item in value]
        return converted if isinstance(value, list) else tuple(converted)
    return value


def numpy_to_torch_tree(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, Mapping):
        return {key: numpy_to_torch_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [numpy_to_torch_tree(item) for item in value]
        return converted if isinstance(value, list) else tuple(converted)
    return value


def summarize_array_tree(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": tuple(int(dim) for dim in value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        return {key: summarize_array_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [summarize_array_tree(item) for item in value]
        return converted if isinstance(value, list) else tuple(converted)
    return value
