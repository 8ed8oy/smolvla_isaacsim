# 环境说明

本文件只记录当前工作区已经确认过的事实，以及下一步应当基于这些事实采取的策略。

最后核对日期：`2026-05-13`

## 1. 主机与环境

- 操作系统：`Ubuntu 22.04.5 LTS`
- 内核版本：`6.8.0-111-generic`
- 当前 Arena conda 环境：`/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/.conda/lerobot-arena`
- 当前 Arena 环境 Python：`3.11.15`
- 当前 LeRobot 辅助环境：`/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/.venv-lerobot`
- 当前 LeRobot 辅助环境 Python：`3.12.13`
- 工作区 `lerobot/` 版本：`0.5.2`
- `lerobot/pyproject.toml` 要求：`requires-python = ">=3.12"`

## 2. 已确认可用的 Isaac 链路

以下内容已经在当前工作区的 Arena 环境中确认：

- `isaacsim 5.1.0.0`
- `torch 2.7.0+cu128`
- `isaaclab 0.47.2`
- `isaaclab_assets 0.2.3`
- `isaaclab_tasks 0.11.6`
- `isaaclab_rl 0.4.4`
- `isaaclab_mimic 1.0.15`
- `isaaclab_arena 1.0.0`

已验证命令：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python -c \
  "import isaacsim, isaaclab, isaaclab_arena; import torch; print('imports_ok'); print(torch.__version__)"
```

已知结果：

- Isaac 相关导入成功
- `torch.__version__ == 2.7.0+cu128`

## 2.1 已创建的 LeRobot 辅助环境

以下内容已经在当前工作区确认：

- `./.venv-lerobot/bin/python --version` 返回 `Python 3.12.13`
- `./.venv-lerobot/bin/python -m py_compile lerobot/src/lerobot/motors/motors_bus.py` 成功
- `./.venv-lerobot/bin/pip show lerobot` 已显示 `Editable project location: .../lerobot`

当前判断：

- 这个环境已经满足“独立承接 Python 3.12 主线 lerobot 源码”的目标
- 它不会替代 Arena 的 `Python 3.11` 主环境
- 它当前主要用于源码导入、配置检查、processor / policy 逻辑核查
- 它还没有补齐 `lerobot` 运行时依赖，不应直接拿来假定能跑完整 `SmolVLA` 推理

## 3. 当前真正的阻塞点

### 3.1 `lerobot` 与 Python 3.11 硬冲突

这不是“依赖没装全”的问题，而是语言版本级别的不兼容。

已确认：

- `lerobot/pyproject.toml` 声明 `requires-python = ">=3.12"`
- 源码中已经存在 Python 3.12 语法，例如：

```python
type NameOrID = str | int
```

在当前 Arena 环境中直接编译该文件会失败：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python -m py_compile \
  lerobot/src/lerobot/motors/motors_bus.py
```

结论：

- 当前工作区这份新 `lerobot` 不能作为 Isaac Sim 5.1.0 的 `Python 3.11` 同环境安装目标
- 因而官方文档里的 `pip install -e ".[evaluation]"` / `pip install -e ".[smolvla]"`，对当前这棵 `lerobot/` 源码并不成立

### 3.2 `nvidia-smi` 状态在不同终端上下文里不一致

在本次 Codex 终端核查中，`nvidia-smi` 仍返回：

- `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`

但用户反馈在自己的终端中 `nvidia-smi` 正常。

结论：

- 目前不能仅凭 Codex 终端中的 `nvidia-smi` 结果就断言主机 GPU 驱动损坏
- 更合理的判断是：主机侧 GPU 很可能可用，但当前代理终端上下文未能稳定继承相同的设备访问条件
- 后续是否能真正跑图形或 CUDA rollout，应以 Isaac Sim 实际启动结果为准，而不是只看 `nvidia-smi`

### 3.3 当前 Codex 终端中的 Isaac App 启动仍未打通

`2026-05-13` 在当前代理终端里补做了两类最小启动检查：

- `./smolvla_isaac_embed/scripts/setup_isaaclab_arena_env.sh`
- `isaac_app_probe.py --headless --enable_cameras`
- `arena_smoke_check.py --headless --enable_cameras --num_steps 0 ...`

当前结果：

- 环境核查脚本成功，说明 Arena Python 与基础导入链路正常
- `isaac_app_probe.py` 与 `arena_smoke_check.py` 在当前代理终端里都卡在 GPU / Vulkan 初始化阶段
- 典型报错包括 `NVML_ERROR_DRIVER_NOT_LOADED`、`No device could be created`、`Failed to create primary CUDA context`

结论：

- 当前工作区里“包可导入”与“Isaac App 可稳定启动”仍然不是同一件事
- 后续记录应区分“导入成功”与“启动成功”，避免把前者误写成后者

## 4. 当前不应继续做的事

- 不要重复安装已经能导入的 Isaac 组件
- 不要在 `Python 3.11` 的 Arena 环境里强装当前工作区这份 `lerobot`
- 不要把 `./smolvla_isaac_embed/scripts/setup_isaaclab_arena_env.sh` 当作“一键装全”的可靠入口，除非先修正其中的 `lerobot` 安装步骤

## 5. 当前最合理的技术路线

推荐顺序如下：

1. 先把 `IsaacLab-Arena` 独立跑通
2. 先确认 `gr1_open_microwave` 这类目标环境的观测键、动作维度、摄像头键
3. 在 `smolvla_isaac_embed/` 内实现最小适配脚本，先不依赖安装整个 `lerobot`
4. 等最小环境链路稳定后，再决定是：
   - 单独建立 `Python 3.12` 的 LeRobot 环境用于训练或评测 CLI
   - 还是只抽取评测所需的最小逻辑接入当前 Arena 环境

## 6. 当前推荐方案判断

### 不推荐：继续寻找“兼容 Python 3.11 的旧版 LeRobot 主线”

原因：

- 这会把工作区分裂成“本地新源码”和“外部旧版文档环境”两套语义
- 后续 SmolVLA / LeRobot 主线对齐会更难
- 你当前目标首先是跑通 Arena 环境并保留可复现性，不是回滚整个上游生态

### 也不推荐：现在立刻把全部工作切到新的 Python 3.12 环境

原因：

- Isaac Sim 5.1.0 官方文档链路本身是围绕 `Python 3.11`
- 现在 Isaac 侧已经基本装好，再平移到 3.12 成本高，而且未必兼容
- 这会把“环境是否能跑”与“LeRobot 是否能装”两个问题耦合在一起

### 当前最推荐：先绕开 `lerobot` 安装，做最小集成

原因：

- 这是当前风险最小、复用现有成果最多的路径
- 可以先完成环境实例化、观测检查、动作维度确认
- 完成这些后，再决定是否单独为 `lerobot` 建立 3.12 辅助环境

## 7. 下一步执行目标

下一步应优先完成：

- 在用户本机可用的 GPU / 显示终端里重新执行 Arena smoke test
- 复核 `gr1_open_microwave` 的 observation / action schema 是否与现有记录一致
- 固化 `camera_keys` / `state_keys` / `rename_map` 的“推荐配置”与“已实测配置”两套口径
- 与 SmolVLA 所需输入输出的最小适配脚本
- 视需要为 `./.venv-lerobot` 继续补装最小运行依赖，并验证 `SmolVLA` checkpoint 加载
