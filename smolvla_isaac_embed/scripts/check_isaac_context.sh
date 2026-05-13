#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${1:-./.conda/lerobot-arena/bin/python}"

echo "== basic =="
date -Is
pwd
id
echo

echo "== env =="
for key in \
  DISPLAY \
  WAYLAND_DISPLAY \
  XDG_SESSION_TYPE \
  XDG_CURRENT_DESKTOP \
  XDG_RUNTIME_DIR \
  CUDA_VISIBLE_DEVICES \
  NVIDIA_VISIBLE_DEVICES \
  NVIDIA_DRIVER_CAPABILITIES \
  VK_ICD_FILENAMES \
  LD_LIBRARY_PATH \
  XDG_CACHE_HOME \
  MPLCONFIGDIR \
  OMNI_KIT_ACCEPT_EULA \
  ACCEPT_EULA \
  PRIVACY_CONSENT
do
  printf '%s=%s\n' "$key" "${!key-}"
done
echo

echo "== gpu devices =="
for path in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/dri/renderD128 /dev/dri/card0; do
  if [ -e "$path" ]; then
    ls -l "$path"
  else
    echo "$path missing"
  fi
done
echo

echo "== host tools =="
command -v nvidia-smi || true
command -v xrandr || true
command -v vulkaninfo || true
echo

echo "== nvidia-smi -L =="
nvidia-smi -L 2>&1 || true
echo

echo "== xrandr --listproviders =="
xrandr --listproviders 2>&1 || true
echo

echo "== python / torch =="
"$PYTHON_BIN" - <<'PY'
import os
import platform
import sys

print("python", sys.version)
print("platform", platform.platform())

try:
    import torch
    print("torch", torch.__version__)
    print("torch.cuda.is_available", torch.cuda.is_available())
    print("torch.cuda.device_count", torch.cuda.device_count())
    print("torch.version.cuda", torch.version.cuda)
except Exception as exc:
    print("torch_probe_error", repr(exc))

for key in ["HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR"]:
    print(f"{key}={os.environ.get(key)}")
PY
