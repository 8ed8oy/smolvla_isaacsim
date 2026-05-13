# 已验证命令

本文件只记录“真正执行过并有结果”的命令。

不要把尚未验证的命令写成既成事实。

## 1. 环境检查

```bash
python3 --version
./.venv/bin/python --version
conda --version
nvidia-smi
curl -I https://conda.anaconda.org
curl -I https://pypi.nvidia.com
```

结果：

- 是否成功：部分成功
- 备注：`python3` 为 `3.13.12`，工作区 `.venv` 为 `3.12.13`，`conda` 为 `26.1.1`
- 备注：`conda.anaconda.org` 与 `pypi.nvidia.com` 可访问
- 备注：`nvidia-smi` 在 Codex 终端上下文中仍可能失败，但这与用户本机终端结果不一致，不能单凭这一条认定 GPU 驱动异常

## 2. Isaac 环境创建

```bash
mkdir -p .cache .conda/pkgs

XDG_CACHE_HOME=$PWD/.cache \
CONDA_PKGS_DIRS=$PWD/.conda/pkgs \
conda create -y -p $PWD/.conda/lerobot-arena python=3.11 ffmpeg=7.1.1 \
  --solver classic \
  --override-channels \
  -c https://conda.anaconda.org/conda-forge

PIP_CACHE_DIR=$PWD/.cache/pip \
  ./.conda/lerobot-arena/bin/pip install \
  "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com

git clone --depth 1 --branch v2.3.0 \
  https://github.com/isaac-sim/IsaacLab.git \
  ./IsaacLab

git clone --depth 1 --branch release/0.1.1 \
  https://github.com/isaac-sim/IsaacLab-Arena.git \
  ./IsaacLab-Arena
```

结果：

- 是否成功：成功
- 备注：`./.conda/lerobot-arena` 已成功创建，环境 Python 为 `3.11.15`
- 备注：`Isaac Sim 5.1.0`、`IsaacLab v2.3.0`、`IsaacLab-Arena release/0.1.1` 已在当前环境可导入
- 备注：ZJU 镜像源在 TLS 上不稳定，最终改用官方 `conda-forge`

## 3. Isaac 导入核查

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python -c \
  "import isaacsim, isaaclab, isaaclab_arena; import torch; print('imports_ok'); print(torch.__version__)"
```

结果：

- 是否成功：成功
- 备注：已确认 `isaacsim`、`isaaclab`、`isaaclab_arena` 能导入，`torch` 为 `2.7.0+cu128`

## 4. 当前 `lerobot` 兼容性核查

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python -m py_compile \
  lerobot/src/lerobot/motors/motors_bus.py
```

结果：

- 是否成功：失败
- 备注：失败点是 Python 3.12 专属语法 `type NameOrID = str | int`
- 备注：说明当前工作区这份 `lerobot` 不适合直接安装到 `Python 3.11` 的 Arena 环境

## 5. SmolVLA 加载

```bash
# 待填写
```

结果：

- 是否成功：
- 备注：

## 5.1 LeRobot 辅助环境创建

```bash
./.venv/bin/python -m venv .venv-lerobot
./.venv-lerobot/bin/python --version
./.venv-lerobot/bin/python -m py_compile lerobot/src/lerobot/motors/motors_bus.py
PIP_NO_BUILD_ISOLATION=1 ./.venv-lerobot/bin/pip install -e ./lerobot --no-deps --no-build-isolation
./.venv-lerobot/bin/pip show lerobot
```

结果：

- 是否成功：成功
- 备注：创建了独立辅助环境 `./.venv-lerobot`
- 备注：环境 Python 为 `3.12.13`
- 备注：已确认当前工作区 `lerobot` 源码可在该环境编译
- 备注：已确认 `lerobot` 以 editable 方式挂载到该环境
- 备注：当前只完成了源码挂载，不代表 `torch`、`transformers` 等运行依赖已经补齐

## 6. 最小闭环运行

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python \
  ./smolvla_isaac_embed/scripts/arena_smoke_check.py \
  --headless \
  --enable_cameras \
  --num_steps 2 \
  gr1_open_microwave \
  --embodiment gr1_pink \
  --object mustard_bottle
```

结果：

- 是否成功：部分执行
- 输出位置：终端标准输出
- 备注：该脚本不依赖安装 `lerobot`，用于优先确认 Arena 环境、观测键和动作维度
- 备注：历史记录中，已有一次最小检查结果被整理进 `docs/interfaces.md`
- 备注：`2026-05-13` 在 Codex 代理终端再次执行时，Isaac Sim 在 GPU / Vulkan 初始化阶段失败，未能在当前终端上下文中稳定复现 `reset_ok`

### 6.1 `setup_isaaclab_arena_env.sh`

```bash
./smolvla_isaac_embed/scripts/setup_isaaclab_arena_env.sh
```

结果：

- 是否成功：成功
- 备注：已确认 Arena 环境 Python 为 `3.11.15`
- 备注：已确认 `isaacsim`、`isaaclab`、`isaaclab_arena` 在该环境中可导入
- 备注：该脚本当前只做“环境核查 + 推荐下一条命令”，不会自动跑通 smoke test

### 6.2 `isaac_app_probe.py`

```bash
OMNI_KIT_ACCEPT_EULA=YES \
XDG_CACHE_HOME=/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/.cache \
HOME=/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace \
./.conda/lerobot-arena/bin/python \
  smolvla_isaac_embed/scripts/isaac_app_probe.py \
  --headless \
  --enable_cameras
```

结果：

- 是否成功：失败
- 备注：`2026-05-13` 在 Codex 代理终端中失败于 Isaac Sim GPU 初始化
- 备注：关键报错包括 `NVML_ERROR_DRIVER_NOT_LOADED`、`No device could be created`、`Failed to create primary CUDA context`
- 备注：该失败说明“当前代理终端上下文”还不能稳定启动 Isaac App，但不能单独用于否定用户本机终端中的 GPU 可用性

## 7. 调试命令

用于排查 observation、动作维度、显存等。

```bash
du -sh ./.conda/lerobot-arena
du -sh /tmp/pip-unpack-* 2>/dev/null | sort -h | tail
find /tmp -maxdepth 2 -type d -name 'pip-*' | sed -n '1,20p'
conda activate /media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/.conda/lerobot-arena
```

## 8. 当前推荐命令

这一节只保留“当前最推荐使用的那几条命令”。

### 6.1 推荐的环境检查命令

```bash
conda activate /media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/.conda/lerobot-arena
python --version
pip show isaacsim
nvidia-smi

./.venv-lerobot/bin/python --version
./.venv-lerobot/bin/pip show lerobot
```

### 6.2 推荐的最小运行命令

```bash
./smolvla_isaac_embed/scripts/setup_isaaclab_arena_env.sh

OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python \
  ./smolvla_isaac_embed/scripts/arena_smoke_check.py \
  --headless \
  --enable_cameras \
  --num_steps 2 \
  gr1_open_microwave \
  --embodiment gr1_pink \
  --object mustard_bottle
```
