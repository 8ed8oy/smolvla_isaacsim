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
- `dry_run_policy.py`：单步验证模型前向与动作输出。
- `run_eval.py`：最短闭环运行脚本，当前可先跑 `1 episode / 1 env / 少量 rollout steps`。
- `benchmark_latency.py`：记录推理与环境步进延迟。

### `adapters/`

放“观测和动作接口适配层”。

这部分是整个工程最关键、最容易出错的地方，建议与主脚本分离。

当前已实现的最小路径：

- `env_adapter.py`：只支持 `policy.robot_joint_pos -> observation.state`
- `env_adapter.py`：只支持 `camera_obs.robot_pov_cam_rgb -> observation.images.robot_pov_cam`
- `action_adapter.py`：只支持 `shape=(*, 36)` 的 GR1 Pink 动作直通校验

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

### `tests/`

放最小验证，不需要一开始写很多，但要覆盖关键接口。

建议优先写：

- 观测字段映射测试。
- 动作维度与顺序测试。
- 安全包装层输入输出测试。

## 当前开发策略

当前采用“嵌入式直连”方案：

```text
Isaac Sim / IsaacLab
-> 环境适配层
-> SmolVLA 预处理
-> SmolVLA 推理
-> 动作后处理
-> 环境执行
```

这样做的原因是：

- 先把最短闭环跑通。
- 先解决观测和动作接口问题。
- 暂时不引入 ROS2 或独立 policy service 的额外复杂度。

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

1. 把现有 `lerobot` 仓库放到 `smolvla_workspace/lerobot/`。
2. 在 `lerobot` 环境中先复现官方 IsaacLab Arena 的 `SmolVLA` 示例。
3. 再开始在本工程中写 `inspect_obs.py` 与 `run_eval.py`。
