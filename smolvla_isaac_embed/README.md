# smolvla_isaac_embed

这是一个面向 `SmolVLA` 嵌入式直连 `Isaac Sim / IsaacLab` 的集成工程。

本工程不复制 `lerobot` 上游源码，而是在其外层增加一层“薄集成层”。你应优先把自己的工作写在这里，而不是直接散落到 `lerobot` 仓库里。

## 工程目标

当前阶段的目标不是做一个通用平台，而是完成以下最小闭环：

1. 从 Isaac 环境拿到观测。
2. 将观测适配到 LeRobot / SmolVLA 所需格式。
3. 运行 `SmolVLA` 推理。
4. 将动作送回仿真环境。
5. 加入一个最小安全测试包装层。

## 目录说明

### `configs/`

放配置文件，而不是把关键参数长期写死在命令行里。

建议后续放入：

- `arena_eval.yaml`：标准评测配置。
- `local_debug.yaml`：低资源调试配置。
- `rename_map.yaml`：观测命名映射。
- `robot_profile.yaml`：动作维度、状态维度、相机配置等。

### `scripts/`

放可直接执行的入口脚本。

建议后续至少维护这几个脚本：

- `inspect_obs.py`：打印并检查环境观测结构。
- `run_eval_single_frame.py`：单帧验证模型前向与动作输出，不进入 rollout 循环。
- `run_eval_bridge.py`：推荐的最小 rollout 主入口，使用 `Python 3.11 Arena + Python 3.12 policy worker` 的双环境桥接方式。
- `run_eval.py`：仅适用于 Arena 与 lerobot 能在同一解释器进程中共存的场景；在当前工作区基线下不作为默认入口。
- `benchmark_latency.py`：记录推理与环境步进延迟。

### `adapters/`

放“观测和动作接口适配层”。

这部分是整个工程最关键、最容易出错的地方，建议与主脚本分离。

当前已实现的最小路径：

- `env_adapter.py`：只支持 `policy.robot_joint_pos -> observation.state`
- `env_adapter.py`：只支持 `camera_obs.robot_pov_cam_rgb -> observation.images.robot_pov_cam`
- `action_adapter.py`：GR1 Pink 36 维动作的 identity 适配器，负责 rank 归一、dtype 规范和调试预览

建议后续放入：

- `env_adapter.py`：环境观测转 LeRobot 观测。
- `rename_map.py`：相机名、字段名映射。
- `action_adapter.py`：动作重排、裁剪与环境接口对齐。

### `wrappers/`

放安全测试相关包装层。

建议后续放入：

- `safety_wrapper.py`：总包装入口。
- `obs_perturb.py`：图像遮挡、噪声、状态扰动。
- `action_filter.py`：动作限幅、禁区约束、急停逻辑。

### `experiments/`

放实验文档与问题记录，不要把这些信息只留在聊天记录里。

建议后续至少维护：

- `notes.md`：日常实验笔记。
- `failures.md`：失败案例与原因。
- `milestones.md`：阶段性完成情况。

### `outputs/`

放程序生成内容。

建议子目录：

- `logs/`
- `videos/`
- `metrics/`
- `debug_dumps/`

## 当前推荐运行命令

单帧验证：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python \
  smolvla_isaac_embed/scripts/run_eval_single_frame.py \
  --config smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml
```

最小 rollout（推荐 bridge 版本）：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python \
  smolvla_isaac_embed/scripts/run_eval_bridge.py \
  --config smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml \
  --max_steps 5 \
  --policy_python ./.venv-lerobot/bin/python \
  --checkpoint smolvla_isaac_embed/models/smolvla-arena-gr1-microwave \
  --num_episodes 1
```

打开视频录制：

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

Bridge 默认会从 `./.venv-lerobot/bin/python` 启动 `smolvla_isaac_embed/scripts/policy_worker.py`。
如果你的 `lerobot` 辅助环境不在这个路径，可以通过 `--policy_python` 覆盖。
上面的命令假设你已经把 `nvidia/smolvla-arena-gr1-microwave` 下载到了
`smolvla_isaac_embed/models/smolvla-arena-gr1-microwave`，这样运行时不需要再访问 Hugging Face。
适配器测试：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python \
  -m pytest -q \
  smolvla_isaac_embed/tests/test_bridge_protocol.py \
  smolvla_isaac_embed/tests/test_env_adapter.py \
  smolvla_isaac_embed/tests/test_action_adapter.py
```

### `tests/`

放最小验证，不需要一开始写很多，但要覆盖关键接口。

建议优先写：

- 观测字段映射测试。
- 动作维度与顺序测试。
- 安全包装层输入输出测试。

## 当前开发策略

当前默认采用“嵌入式直连 + 本地 policy worker bridge”方案：

```text
Isaac Sim / IsaacLab
-> 环境适配层
-> 本地 bridge 协议
-> Python 3.12 policy worker
-> SmolVLA 预处理 / 推理 / 后处理
-> bridge 返回 action
-> 环境执行
```

这样做的原因是：

- 先把最短闭环跑通。
- 先解决观测和动作接口问题。
- 暂时不引入 ROS2 或远程 policy service 的额外复杂度。
- 避免把当前 `Python 3.12` 的 `lerobot` 主线强行塞进 Isaac Sim 的 `Python 3.11` 进程。

## 当前基线状态

截至 `2026-05-18`，当前工程已经完成一次可复现的最小 bridge rollout，并成功生成视频。

- 运行入口：`smolvla_isaac_embed/scripts/run_eval_bridge.py`
- checkpoint：`smolvla_isaac_embed/models/smolvla-arena-gr1-microwave`
- 运行形态：`Python 3.11 Arena + Python 3.12 policy worker`
- 已验证链路：
  - 环境 reset
  - 观测适配
  - SmolVLA policy 推理
  - 36 维动作回传
  - 视频录制
- 当前观测到的 GPU 指标：
  - 显存峰值：`7.05 GB`
  - 功率：`70 W / 170 W`

这意味着当前工作重点可以从“把链路跑通”切换到“验证动作语义、记录 rollout 指标、加最小 safety wrapper”。

## 协作原则

- 优先保留 `lerobot` 上游结构，不复制整段源码到这里。
- 任何长期有效的结论，都沉淀到 Markdown 文档中。
- 临时脚本如果有复用价值，应整理进 `scripts/`。
- 与 AI 协作时，尽量给出：
  - 当前目标
  - 当前卡点
  - 已知环境版本
  - 你希望 AI 修改或分析的目录范围

## 下一步建议

1. 固化动作维度与动作顺序检查，确认 36 维输出与环境执行顺序完全一致。
2. 为 rollout 增加最小指标记录，至少保存 episode steps、reward、terminated/truncated 和视频路径。
3. 在 `wrappers/` 中落地第一个 safety wrapper，优先做动作限幅或急停保护。
