#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_DIR="${WORKSPACE_ROOT}/.conda/lerobot-arena"
CACHE_DIR="${WORKSPACE_ROOT}/.cache"
LEROBOT_DIR="${WORKSPACE_ROOT}/lerobot"
ISAACLAB_DIR="${WORKSPACE_ROOT}/IsaacLab"
ARENA_DIR="${WORKSPACE_ROOT}/IsaacLab-Arena"

mkdir -p "${CACHE_DIR}" "${WORKSPACE_ROOT}/.conda/pkgs"

echo "[1/6] Create or reuse workspace-local conda env"
if [ ! -x "${ENV_DIR}/bin/python" ]; then
  XDG_CACHE_HOME="${CACHE_DIR}" \
  CONDA_PKGS_DIRS="${WORKSPACE_ROOT}/.conda/pkgs" \
  conda create -y -p "${ENV_DIR}" python=3.11 ffmpeg=7.1.1 \
    --solver classic \
    --override-channels \
    -c https://conda.anaconda.org/conda-forge
fi

echo "[2/6] Accept NVIDIA EULA for Isaac Sim"
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

echo "[3/6] Install Isaac Sim 5.1.0"
PIP_CACHE_DIR="${CACHE_DIR}/pip" \
  "${ENV_DIR}/bin/pip" install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com

echo "[4/6] Clone and install IsaacLab 2.3.0"
if [ ! -d "${ISAACLAB_DIR}" ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
fi
git -C "${ISAACLAB_DIR}" fetch --tags
git -C "${ISAACLAB_DIR}" checkout v2.3.0
"${ISAACLAB_DIR}/isaaclab.sh" -i

echo "[5/6] Clone and install IsaacLab-Arena release/0.1.1"
if [ ! -d "${ARENA_DIR}" ]; then
  git clone https://github.com/isaac-sim/IsaacLab-Arena.git "${ARENA_DIR}"
fi
git -C "${ARENA_DIR}" fetch --tags
git -C "${ARENA_DIR}" checkout release/0.1.1
"${ENV_DIR}/bin/pip" install -e "${ARENA_DIR}"

echo "[6/6] Install local lerobot extras and Arena companion deps"
"${ENV_DIR}/bin/pip" install -e "${LEROBOT_DIR}[evaluation,smolvla]"
"${ENV_DIR}/bin/pip" install \
  onnxruntime==1.23.2 \
  lightwheel-sdk==1.0.1 \
  "vuer[all]==0.0.70" \
  qpsolvers==4.8.1 \
  numpy==1.26.0

cat <<EOF

Setup completed.

Activate with:
  conda activate ${ENV_DIR}

Suggested next checks:
  ${ENV_DIR}/bin/python --version
  ${ENV_DIR}/bin/pip show isaacsim
  nvidia-smi
EOF
