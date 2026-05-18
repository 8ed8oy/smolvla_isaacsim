# 已验证命令

本文件只记录“真正执行过并有结果”的命令。

不要把尚未验证的命令写成既成事实。

环境背景、阻塞判断、终端差异分析等说明优先放在 `docs/environment.md`。

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

## 2. Arena Python 与 Isaac 依赖

### 2.1 Isaac 环境创建

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

### 2.2 Isaac 导入核查

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python -c \
  "import isaacsim, isaaclab, isaaclab_arena; import torch; print('imports_ok'); print(torch.__version__)"
```

结果：

- 是否成功：成功
- 备注：已确认 `isaacsim`、`isaaclab`、`isaaclab_arena` 能导入，`torch` 为 `2.7.0+cu128`

## 3. LeRobot 兼容性与辅助环境

### 3.1 当前 `lerobot` 兼容性核查

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python -m py_compile \
  lerobot/src/lerobot/motors/motors_bus.py
```

结果：

- 是否成功：失败
- 备注：失败点是 Python 3.12 专属语法 `type NameOrID = str | int`
- 备注：说明当前工作区这份 `lerobot` 不适合直接安装到 `Python 3.11` 的 Arena 环境

### 3.2 LeRobot 辅助环境创建

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

## 4. Isaac / Arena 最小运行核查

### 4.1 `setup_isaaclab_arena_env.sh`

```bash
./smolvla_isaac_embed/scripts/setup_isaaclab_arena_env.sh
```

结果：

- 是否成功：成功
- 备注：已确认 Arena 环境 Python 为 `3.11.15`
- 备注：已确认 `isaacsim`、`isaaclab`、`isaaclab_arena` 在该环境中可导入
- 备注：该脚本当前只做“环境核查 + 推荐下一条命令”，不会自动跑通 smoke test

### 4.2 `isaac_app_probe.py`

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
- 备注：当前 Codex 终端中失败于 Isaac Sim GPU 初始化
- 备注：环境差异与原因判断见 `docs/environment.md`

### 4.3 `arena_smoke_check.py` 历史最小闭环命令

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
- 备注：当前终端复核状态与环境解释见 `docs/environment.md`

### 4.4 `arena_smoke_check.py` 在当前 Codex 终端的最新复核

```bash
OMNI_KIT_ACCEPT_EULA=YES \
XDG_CACHE_HOME=/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/.cache \
HOME=/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace \
./.conda/lerobot-arena/bin/python smolvla_isaac_embed/scripts/arena_smoke_check.py \
  --headless \
  --enable_cameras \
  --num_steps 0 \
  gr1_open_microwave \
  --embodiment gr1_pink \
  --object mustard_bottle
```

结果：

- 是否成功：失败
- 备注：`2026-05-15` 在当前 Codex 终端中无法完成 Isaac Sim 启动
- 备注：环境差异与 GPU 透传相关判断见 `docs/environment.md`

### 4.5 `arena_smoke_check.py` 在用户本机终端的成功复核

```bash
OMNI_KIT_ACCEPT_EULA=YES \
XDG_CACHE_HOME=/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/.cache \
HOME=/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace \
./.conda/lerobot-arena/bin/python smolvla_isaac_embed/scripts/arena_smoke_check.py \
  --headless \
  --enable_cameras \
  --num_steps 0 \
  gr1_open_microwave \
  --embodiment gr1_pink \
  --object mustard_bottle
```

结果：

- 是否成功：成功
- 备注：用户本机终端已验证 `reset_ok`
- 备注：运行时设备为 `cuda:0`
- 备注：动作空间形状为 `(1, 36)`
- 备注：观测顶层 keys 为 `['camera_obs', 'policy']`
- 备注：最终输出 `smoke_test_ok`

## 5. 测试验证

### 5.1 观测与动作适配器测试

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python \
  -m pytest -q \
  smolvla_isaac_embed/tests/test_env_adapter.py \
  smolvla_isaac_embed/tests/test_action_adapter.py
```

结果：

- 是否成功：历史验证
- 备注：当前工作区的 `.venv` 与 `.venv-lerobot` 里都还没有 `pytest`
- 备注：Arena 环境 `./.conda/lerobot-arena` 同时具备 `torch` 与 `pytest`，因此适合作为这组单元测试的执行环境
- 备注：旧版最小测试集合曾得到 `3 passed in 4.23s`
- 备注：当前 `test_action_adapter.py` 已扩展，建议重新执行以刷新结果

### 5.2 测试目录整体回归

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python \
  -m pytest -q \
  smolvla_isaac_embed/tests
```

结果：

- 是否成功：历史验证
- 备注：旧版 `smolvla_isaac_embed/tests` 曾通过
- 备注：当前测试数量已变化，建议重新执行以刷新结果

## 6. 单帧入口脚本整理

### 6.1 `run_eval_single_frame.py`

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python \
  smolvla_isaac_embed/scripts/run_eval_single_frame.py \
  --config smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml
```

结果：

- 是否成功：待按需复核
- 备注：这是单帧验证入口，只做一次 reset、一次观测整理、一次 policy 前向和 action 打印
- 备注：它不替代 `run_eval.py` 的 rollout 主循环

### 6.2 `run_eval.py`

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python \
  smolvla_isaac_embed/scripts/run_eval.py \
  --config smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml \
  --max_steps 5 \
  --num_episodes 1
```

结果：

- 是否成功：待按需复核
- 备注：这是完整 rollout 主入口，负责 episode / step 循环、`reset`、`done`、`truncated` 和 `max_steps`

### 6.3 `run_eval.py` 打开视频录制

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python \
  smolvla_isaac_embed/scripts/run_eval.py \
  --config smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml \
  --max_steps 100 \
  --num_episodes 1 \
  --video \
  --video_length 100 \
  --video_interval 200 \
  --video_dir smolvla_isaac_embed/outputs/videos/run_eval
```

结果：

- 是否成功：待按需复核
- 备注：这是 rollout 的视频录制命令，适用于需要保存可视化轨迹时

## 7. 适配器测试命令

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python \
  -m pytest -q \
  smolvla_isaac_embed/tests/test_env_adapter.py \
  smolvla_isaac_embed/tests/test_action_adapter.py
```

结果：

- 是否成功：历史验证
- 备注：旧版最小测试集合曾得到 `3 passed in 4.23s`
- 备注：当前 `test_action_adapter.py` 已扩展，建议按本节命令重新执行完整回归
- 备注：这是纯 adapter / 参数层测试，不需要启动 Isaac App

## 8. 调试命令

用于排查 observation、动作维度、显存等。

```bash
du -sh ./.conda/lerobot-arena
du -sh /tmp/pip-unpack-* 2>/dev/null | sort -h | tail
find /tmp -maxdepth 2 -type d -name 'pip-*' | sed -n '1,20p'
conda activate /media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/.conda/lerobot-arena
```

## 9. 当前推荐命令

这一节只保留“当前最推荐使用且已验证过的那几条命令”。

当前基线配置文件：

```text
smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml
```

其中已固化：

- Arena 环境名 `gr1_open_microwave`
- LeRobot 参考环境名 `gr1_microwave`
- 在线 checkpoint id `nvidia/smolvla-arena-gr1-microwave`
- 当前推荐本地 checkpoint 路径 `smolvla_isaac_embed/models/smolvla-arena-gr1-microwave`
- `state_keys`
- `camera_keys`
- `rename_map`

### 9.1 推荐的环境检查命令

```bash
conda activate /media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/.conda/lerobot-arena
python --version
pip show isaacsim
nvidia-smi

./.venv-lerobot/bin/python --version
./.venv-lerobot/bin/pip show lerobot
./.venv-lerobot/bin/python -c "import torch, huggingface_hub, lerobot; print(torch.__version__)"
```

### 9.2 推荐的最小运行命令

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

### 9.2.1 动作顺序检查

在开始长 rollout 之前，先固定检查当前 36 维动作语义顺序：

```bash
./.conda/lerobot-arena/bin/python \
  smolvla_isaac_embed/scripts/inspect_action_order.py
```

预期输出包括两部分：

- 一行 `action_order_validation=...`
- 后续 `index=.. group=.. name=..` 的逐维展开

当前检查口径：

- 前 `7` 维应为左手末端目标位姿
- 接着 `7` 维应为右手末端目标位姿
- 最后 `22` 维应为双手手指关节目标

### 9.3 推荐的 bridge rollout 命令

当前推荐使用 `run_eval_bridge.py`，让 Isaac / Arena 留在 Python 3.11，让 SmolVLA / LeRobot 留在 Python 3.12。

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python \
  smolvla_isaac_embed/scripts/run_eval_bridge.py \
  --config smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml \
  --policy_python ./.venv-lerobot/bin/python \
  --checkpoint smolvla_isaac_embed/models/smolvla-arena-gr1-microwave \
  --max_steps 100 \
  --num_episodes 1 \
  --video \
  --video_length 100 \
  --video_interval 200 \
  --video_dir smolvla_isaac_embed/outputs/videos/run_eval
```

注意：

- 上面命令假设模型 snapshot 已经手动下载到 `smolvla_isaac_embed/models/smolvla-arena-gr1-microwave`
- 每行续行反斜杠 `\` 后面不能有空格
- 如果 shell 已经激活了其他环境也没关系，这条命令显式指定了 Arena Python 和 policy Python

### 9.4 本地 checkpoint 下载命令

模型主页：

- https://huggingface.co/nvidia/smolvla-arena-gr1-microwave

当前工程推荐把完整 checkpoint 放到：

```text
smolvla_isaac_embed/models/smolvla-arena-gr1-microwave
```

优先推荐使用 Hugging Face 的页面下载或 `snapshot_download` 拉取完整 snapshot，不要把“直接 `git clone` 模型仓库”当作默认下载方式。

原因：

- Hugging Face 模型仓库通常通过 Git LFS 管理权重文件
- 直接 `git clone` 时如果本机没有正确执行 LFS 拉取，仓库里会留下 pointer 文本文件，而不是真正的 `.safetensors`
- 这类不完整文件最常见的特征是体积非常小，例如一百多字节
- 当前桥接加载中如果读到这类假文件，可能报 `SafetensorError: Error while deserializing header: header too large`

如果网络可访问 Hugging Face，可先手动拉取模型：

```bash
mkdir -p smolvla_isaac_embed/models

./.venv-lerobot/bin/python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="nvidia/smolvla-arena-gr1-microwave",
    local_dir="smolvla_isaac_embed/models/smolvla-arena-gr1-microwave",
)
print(path)
PY
```

如果该命令报 `Temporary failure in name resolution`，说明当前终端访问 Hugging Face 的 DNS / 网络不稳定，应换到可联网会话或使用已有本地 snapshot。

如果你是从网页手动下载，请确保把完整文件下载到本地目录，而不是只把仓库结构 clone 下来。至少应包含：

```text
README.md
config.json
model.safetensors
policy_preprocessor.json
policy_preprocessor_step_5_normalizer_processor.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
train_config.json
```

如果你已经 `git clone` 过模型目录，请额外检查小的 `.safetensors` 文件是不是 LFS pointer：

```bash
ls -lh smolvla_isaac_embed/models/smolvla-arena-gr1-microwave
sed -n '1,5p' smolvla_isaac_embed/models/smolvla-arena-gr1-microwave/policy_preprocessor_step_5_normalizer_processor.safetensors
```

如果文件内容长这样：

```text
version https://git-lfs.github.com/spec/v1
oid sha256:...
size ...
```

说明它不是真权重文件，必须重新下载对应 raw 文件或重新执行完整 snapshot 下载。

### 9.5 checkpoint 完整性快速检查

在启动 `run_eval_bridge.py` 之前，建议先检查本地 checkpoint 是否完整：

```bash
ls -lh smolvla_isaac_embed/models/smolvla-arena-gr1-microwave

file smolvla_isaac_embed/models/smolvla-arena-gr1-microwave/model.safetensors
file smolvla_isaac_embed/models/smolvla-arena-gr1-microwave/policy_preprocessor_step_5_normalizer_processor.safetensors
file smolvla_isaac_embed/models/smolvla-arena-gr1-microwave/policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

经验上：

- `model.safetensors` 应该是大文件，而不是几 KB 或几百字节
- `policy_preprocessor_step_5_normalizer_processor.safetensors` 和 `policy_postprocessor_step_0_unnormalizer_processor.safetensors` 也不应是 `130B`
- 如果它们是 `130B` 左右，基本可以判定下载到的是 Git LFS pointer
