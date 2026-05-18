# 环境说明

本文件只记录当前工作区已经确认过的事实，以及下一步应当基于这些事实采取的策略。

最后核对日期：`2026-05-18`

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
- 它当前作为 `policy_worker.py` 的默认运行环境
- `2026-05-18` 已补齐 `lerobot[smolvla]` 运行时依赖
- 已确认 `./.venv-lerobot/bin/python` 中 `torch`、`huggingface_hub`、`lerobot` 均可导入
- 运行 bridge 时应通过 `--policy_python ./.venv-lerobot/bin/python` 显式指定该环境

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

补充记录：

- `2026-05-15` 再次运行 `arena_smoke_check.py --headless --enable_cameras --num_steps 0 gr1_open_microwave --embodiment gr1_pink --object mustard_bottle`
- 当前 Codex 终端报错仍停留在 GPU 初始化阶段
- 终端中能读到 `/proc/driver/nvidia/gpus/0000:0b:00.0/information`，显示 `NVIDIA GeForce RTX 3060`
- 但 `/dev/nvidia*` 设备节点不存在，`nvidia-smi` 也无法与驱动通信
- 因此更合理的结论是：这里是“代理终端的 GPU 设备透传问题”，而不是直接推翻用户本机终端上的 GPU 可用性
- 用户本机终端随后已成功跑通同一条命令，输出 `reset_ok`、`device=cuda:0`、`action_space_shape=(1, 36)`、`observation_top_level_keys=['camera_obs', 'policy']`、`smoke_test_ok`
- 这说明 Arena / IsaacLab / Isaac Sim 的主链路在正确的 GPU 会话里是可用的
- 当前记录应更新为：问题不在环境本身，而在 Codex 当前会话对 GPU 设备的访问方式
- 当前基线配置已固化到 `smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml`

## 4. 当前不应继续做的事

- 不要重复安装已经能导入的 Isaac 组件
- 不要在 `Python 3.11` 的 Arena 环境里强装当前工作区这份 `lerobot`
- 不要把 `./smolvla_isaac_embed/scripts/setup_isaaclab_arena_env.sh` 当作“一键装全”的可靠入口，除非先修正其中的 `lerobot` 安装步骤
- 不要把 `source .../activate` 接在 `run_eval_bridge.py` 命令末尾；环境激活必须作为单独 shell 命令执行
- 不要对 venv Python 入口做 `Path.resolve()` 后再作为子进程解释器使用；这会绕过 venv，落到底层 CPython 可执行文件

## 5. 当前最合理的技术路线

推荐顺序如下：

1. 先把 `IsaacLab-Arena` 独立跑通
2. 先确认 `gr1_open_microwave` 这类目标环境的观测键、动作维度、摄像头键
3. 使用 `run_eval_bridge.py` 作为当前默认 rollout 入口
4. 保持双环境分工：
   - `./.conda/lerobot-arena/bin/python` 负责 Isaac Sim / Arena
   - `./.venv-lerobot/bin/python` 负责 `lerobot` / SmolVLA policy
5. checkpoint 优先使用本地 snapshot，避免 worker 启动时依赖 Hugging Face 在线下载

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

### 当前最推荐：双环境 bridge

原因：

- Isaac Sim / Arena 继续留在已验证的 Python 3.11 环境
- 当前工作区 `lerobot` 继续留在兼容的 Python 3.12 环境
- `run_eval_bridge.py` 通过本地 `policy_worker.py` 子进程连接二者
- 不需要改动上游 `lerobot/`，也不需要把 Isaac 迁移到 Python 3.12

## 6.1 2026-05-18 bridge 排查结论

已确认的推进顺序：

- `run_eval.py` 单进程路径不适用于当前环境，因为 Arena Python 3.11 与 `lerobot >=3.12` 冲突
- `run_eval_bridge.py` 已能完成参数解析、Arena 环境创建、首次 `env.reset()`、视频帧探测
- Arena 侧在用户 GPU 终端中可创建 `gr1_open_microwave`，动作空间为 `(1, 36)`，观测顶层 keys 为 `camera_obs` 与 `policy`
- `policy_worker.py` 会在 Python 3.12 环境中加载 checkpoint 并执行 policy 推理

本轮遇到并修正的关键问题：

- Arena parser 构造会打印 `AppLauncher` warning；该 warning 不是主故障
- 早期 `_inject_example_environment()` 读取 Arena parser 内部 subparser 结构时会卡住；已改为简单追加 config 中的 example environment
- `Path.resolve()` 会把 `./.venv-lerobot/bin/python` 解析到底层 uv CPython，导致 worker 绕过 venv；已改为保留 venv 入口路径
- `./.venv-lerobot` 已安装 `lerobot[smolvla]`
- Hugging Face 在线下载 checkpoint 时曾出现 DNS / `httpx` 重试错误；当前推荐先把模型 snapshot 下载到本地目录
- shell 命令中若把 `source .../activate` 接在 bridge 命令末尾，会被 argparse 当作 Arena `example_environment`，因此会报 `invalid choice: 'source'`

当前推荐 checkpoint 路径：

```text
smolvla_isaac_embed/models/smolvla-arena-gr1-microwave
```

## 7. 下一步执行目标

下一步应优先完成：

- 使用本地 checkpoint 路径运行 `run_eval_bridge.py`
- 如果 worker 后续继续报错，优先区分 checkpoint 加载、processor 构造、首次 `select_action()` 三个阶段
- 保留 `stage=` 日志直到首个完整 rollout 稳定通过
- 将首个成功 rollout 的命令和结果补充到 `docs/commands.md`
