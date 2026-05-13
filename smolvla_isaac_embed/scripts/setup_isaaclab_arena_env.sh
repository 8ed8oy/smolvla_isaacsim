#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_DIR="${WORKSPACE_ROOT}/.conda/lerobot-arena"
LEROBOT_DIR="${WORKSPACE_ROOT}/lerobot"
SMOKE_SCRIPT="${WORKSPACE_ROOT}/smolvla_isaac_embed/scripts/arena_smoke_check.py"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "Missing Arena environment: ${ENV_DIR}" >&2
  exit 1
fi

export OMNI_KIT_ACCEPT_EULA=YES

echo "[1/4] Verify Python version"
"${ENV_DIR}/bin/python" --version

echo "[2/4] Verify Isaac imports"
"${ENV_DIR}/bin/python" -c "import isaacsim, isaaclab, isaaclab_arena; import torch; print('imports_ok'); print(torch.__version__)"

echo "[3/4] Report current lerobot compatibility"
echo "lerobot source tree: ${LEROBOT_DIR}"
echo "Arena python cannot install current lerobot if it requires Python >= 3.12."

echo "[4/4] Suggested next command"
cat <<EOF
${ENV_DIR}/bin/python ${SMOKE_SCRIPT} --headless --enable_cameras --num_steps 2 gr1_open_microwave --embodiment gr1_pink --object mustard_bottle
EOF
