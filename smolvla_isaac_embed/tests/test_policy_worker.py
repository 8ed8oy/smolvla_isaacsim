"""Focused tests for stdout/stderr separation in the policy worker."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "policy_worker.py"
)


def _load_policy_worker_module():
    spec = importlib.util.spec_from_file_location("test_policy_worker_module", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load policy_worker module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_stdout_guard_redirects_prints_to_stderr(monkeypatch) -> None:
    module = _load_policy_worker_module()

    stdout_bytes = io.BytesIO()
    stderr_bytes = io.BytesIO()
    fake_stdout = io.TextIOWrapper(stdout_bytes, encoding="utf-8", write_through=True)
    fake_stderr = io.TextIOWrapper(stderr_bytes, encoding="utf-8", write_through=True)

    monkeypatch.setattr(sys, "stdout", fake_stdout)
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    protocol_stdout = module._install_protocol_stdout_guard()
    print("Loading weights from local directory")
    fake_stderr.flush()

    assert protocol_stdout is stdout_bytes
    assert stdout_bytes.getvalue() == b""
    assert b"Loading weights from local directory" in stderr_bytes.getvalue()
