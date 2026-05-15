# 接口约定

本文件用于描述 `Isaac Sim / IsaacLab` 与 `SmolVLA` 之间最关键的输入输出接口。

## 1. 总体链路

```text
Isaac 环境观测
-> 观测适配层
-> LeRobot / SmolVLA 输入
-> SmolVLA 输出动作
-> 动作适配层
-> Isaac 环境执行
```

## 2. 环境观测原始结构

记录 Isaac 环境实际返回的 observation 结构。

本节结果来自一次历史最小检查记录；截至 `2026-05-13`，这些值已被整理进本文，但尚未在当前 Codex 代理终端里再次稳定复现。

补充说明：

- `2026-05-15` 在当前 Codex 终端中重新执行同一条 `arena_smoke_check.py --num_steps 0` 命令时，Isaac Sim 在 GPU / Vulkan 初始化阶段失败
- 当前终端可读到 `/proc/driver/nvidia/gpus/0000:0b:00.0/information`，但没有可用的 `/dev/nvidia*` 设备节点
- 这意味着“历史观测记录仍有效”，但“当前代理终端能否直接启动 Isaac App”不能再假定成立
- 同一天，用户本机终端已成功跑通同一条 smoke check，说明这里的问题是“Codex 运行上下文的 GPU 透传/会话差异”，不是环境记录本身失效

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

记录口径：

- 环境：`gr1_open_microwave`
- embodiment：`gr1_pink`
- object：`mustard_bottle`
- 观测采样时 `num_envs=1`
- 设备：历史记录为 `cuda:0`

### 2.1 顶层字段

- `policy`
- `camera_obs`
- 其他：无，当前顶层 keys 为 `["camera_obs", "policy"]`

### 2.2 `policy` 子字段

- `actions`：`shape=(1, 36)`，`dtype=torch.float32`
- `robot_joint_pos`：`shape=(1, 54)`，`dtype=torch.float32`
- `robot_root_pos`：`shape=(1, 3)`，`dtype=torch.float32`
- `robot_root_rot`：`shape=(1, 4)`，`dtype=torch.float32`
- `robot_links_state`：`shape=(1, 55, 13)`，`dtype=torch.float32`
- `left_eef_pos`：`shape=(1, 3)`，`dtype=torch.float32`
- `left_eef_quat`：`shape=(1, 4)`，`dtype=torch.float32`
- `right_eef_pos`：`shape=(1, 3)`，`dtype=torch.float32`
- `right_eef_quat`：`shape=(1, 4)`，`dtype=torch.float32`
- `hand_joint_state`：`shape=(1, 22)`，`dtype=torch.float32`
- `head_joint_state`：`shape=(1, 3)`，`dtype=torch.float32`

### 2.3 `camera_obs` 子字段

- `robot_pov_cam_rgb`：`shape=(1, 512, 512, 3)`，`dtype=torch.uint8`
- 其他：无

### 2.4 当前终端复核状态

`2026-05-15` 在当前 Codex 终端中复核该 smoke check 时，未能完成 Isaac App 启动，关键报错包括：

- `NVML_ERROR_DRIVER_NOT_LOADED`
- `No device could be created`
- `Failed to create any GPU devices`

排查结果：

- `nvidia-smi` 在当前终端中不可用
- `/dev/nvidia*` 在当前终端中不存在
- `/proc/driver/nvidia/gpus/0000:0b:00.0/information` 仍可读到 `NVIDIA GeForce RTX 3060`

结论：

- 这份接口记录仍可作为“历史已观测 schema”的来源
- 但在当前终端里，Isaac Sim 启动前提没有满足，因此不应把这里写成“已在本终端最新复现成功”
- 但用户本机终端已经成功复现，因此当前 schema 记录可以视为“已验证有效”，只是“Codex 当前会话”暂时不能复现

## 3. LeRobot / SmolVLA 期望输入

### 3.1 状态输入

- 目标字段名：`observation.state`
- 来源字段：待最终适配时确定
- 拼接顺序：待最终适配时确定
- 最终维度：待最终适配时确定

当前已知的环境侧可用 `policy` 键为：

- `actions`
- `robot_joint_pos`
- `robot_root_pos`
- `robot_root_rot`
- `robot_links_state`
- `left_eef_pos`
- `left_eef_quat`
- `right_eef_pos`
- `right_eef_quat`
- `hand_joint_state`
- `head_joint_state`

### 3.2 图像输入

- 目标字段名：待最终适配时确定
- 来源字段：`camera_obs.robot_pov_cam_rgb`
- 图像尺寸：`512 x 512`
- 通道顺序：`HWC`
- 是否归一化：环境原始输出未归一化，`dtype=torch.uint8`

### 3.3 任务文本

- 输入字段名：`task`
- 文本来源：待最终适配时确定
- 当前示例任务：可描述为 `open microwave`

## 4. `state_keys`

### 4.1 环境侧可见 `policy` 键

```text
actions
robot_joint_pos
robot_root_pos
robot_root_rot
robot_links_state
left_eef_pos
left_eef_quat
right_eef_pos
right_eef_quat
hand_joint_state
head_joint_state
```

说明：

- 上面这一组是环境 `policy` 观测里当前可见的键
- 这里不能直接等同于最终送入 `observation.state` 的 `state_keys`

### 4.2 当前推荐 `state_keys`

```text
robot_joint_pos
```

说明：

- 该推荐值与 `lerobot/docs/source/envhub_isaaclab_arena.mdx` 中的官方示例保持一致
- 这是当前最保守、最接近 checkpoint 预期的状态输入方案
- 是否需要扩展到更多状态字段，仍应以实际 checkpoint 特征与单帧推理结果为准
- 当前已固化到配置文件：`smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml`

## 5. `camera_keys`

当前配置：

```text
robot_pov_cam_rgb
```

说明：

- 当前已固化到配置文件：`smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml`

## 6. `rename_map`

当前推荐配置：

```json
{
  "observation.images.robot_pov_cam_rgb": "observation.images.robot_pov_cam"
}
```

说明：

- 左侧是环境当前名字
- 右侧是策略期望名字
- 该映射与 `lerobot/docs/source/envhub_isaaclab_arena.mdx` 中的 `SmolVLA` 评测命令保持一致
- 这表示“推荐适配方案”，不等于当前本工程已经有适配层实现
- 当前已固化到配置文件：`smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml`

## 6.1 当前基线配置文件

当前推荐把以下值统一从配置文件读取，而不是继续依赖命令行记忆：

- Arena 环境名：`gr1_open_microwave`
- LeRobot 文档环境名：`gr1_microwave`
- checkpoint：`nvidia/smolvla-arena-gr1-microwave`
- `state_keys`：`robot_joint_pos`
- `camera_keys`：`robot_pov_cam_rgb`
- `rename_map`：`observation.images.robot_pov_cam_rgb -> observation.images.robot_pov_cam`

## 7. 动作接口

### 7.1 模型输出

- 动作张量形状：环境 `action_space.shape == (1, 36)`（本次检查使用 `num_envs=1`）
- `action_dim`：`36`
- 是否 chunked action：否，当前是单步连续动作张量

补充说明：

- `gr1_pink` 使用 `PinkInverseKinematicsAction`
- 动作维度可由 `2 * (3 + 4) + 22 = 36` 理解：
  - 左手末端目标位姿：`7`
  - 右手末端目标位姿：`7`
  - 双手手指关节：`22`

### 7.2 动作语义

记录每一维动作对应的控制含义。

| 索引 | 名称 | 含义 | 备注 |
|---|---|---|---|
| 0 | `left_eef_pos_x` | 左手末端目标位置 x | `gr1_pink` PINK IK 动作 |
| 1 | `left_eef_pos_y` | 左手末端目标位置 y | `gr1_pink` PINK IK 动作 |
| 2 | `left_eef_pos_z` | 左手末端目标位置 z | `gr1_pink` PINK IK 动作 |
| 3 | `left_eef_quat_w` | 左手末端目标姿态四元数 w | `gr1_pink` PINK IK 动作 |
| 4 | `left_eef_quat_x` | 左手末端目标姿态四元数 x | `gr1_pink` PINK IK 动作 |
| 5 | `left_eef_quat_y` | 左手末端目标姿态四元数 y | `gr1_pink` PINK IK 动作 |
| 6 | `left_eef_quat_z` | 左手末端目标姿态四元数 z | `gr1_pink` PINK IK 动作 |
| 7 | `right_eef_pos_x` | 右手末端目标位置 x | `gr1_pink` PINK IK 动作 |
| 8 | `right_eef_pos_y` | 右手末端目标位置 y | `gr1_pink` PINK IK 动作 |
| 9 | `right_eef_pos_z` | 右手末端目标位置 z | `gr1_pink` PINK IK 动作 |
| 10 | `right_eef_quat_w` | 右手末端目标姿态四元数 w | `gr1_pink` PINK IK 动作 |
| 11 | `right_eef_quat_x` | 右手末端目标姿态四元数 x | `gr1_pink` PINK IK 动作 |
| 12 | `right_eef_quat_y` | 右手末端目标姿态四元数 y | `gr1_pink` PINK IK 动作 |
| 13 | `right_eef_quat_z` | 右手末端目标姿态四元数 z | `gr1_pink` PINK IK 动作 |
| 14 | `L_index_proximal_joint` | 左手食指近端关节 | 手指关节位置目标 |
| 15 | `L_middle_proximal_joint` | 左手中指近端关节 | 手指关节位置目标 |
| 16 | `L_pinky_proximal_joint` | 左手小指近端关节 | 手指关节位置目标 |
| 17 | `L_ring_proximal_joint` | 左手无名指近端关节 | 手指关节位置目标 |
| 18 | `L_thumb_proximal_yaw_joint` | 左手拇指近端偏航关节 | 手指关节位置目标 |
| 19 | `R_index_proximal_joint` | 右手食指近端关节 | 手指关节位置目标 |
| 20 | `R_middle_proximal_joint` | 右手中指近端关节 | 手指关节位置目标 |
| 21 | `R_pinky_proximal_joint` | 右手小指近端关节 | 手指关节位置目标 |
| 22 | `R_ring_proximal_joint` | 右手无名指近端关节 | 手指关节位置目标 |
| 23 | `R_thumb_proximal_yaw_joint` | 右手拇指近端偏航关节 | 手指关节位置目标 |
| 24 | `L_index_intermediate_joint` | 左手食指中间关节 | 手指关节位置目标 |
| 25 | `L_middle_intermediate_joint` | 左手中指中间关节 | 手指关节位置目标 |
| 26 | `L_pinky_intermediate_joint` | 左手小指中间关节 | 手指关节位置目标 |
| 27 | `L_ring_intermediate_joint` | 左手无名指中间关节 | 手指关节位置目标 |
| 28 | `L_thumb_proximal_pitch_joint` | 左手拇指近端俯仰关节 | 手指关节位置目标 |
| 29 | `R_index_intermediate_joint` | 右手食指中间关节 | 手指关节位置目标 |
| 30 | `R_middle_intermediate_joint` | 右手中指中间关节 | 手指关节位置目标 |
| 31 | `R_pinky_intermediate_joint` | 右手小指中间关节 | 手指关节位置目标 |
| 32 | `R_ring_intermediate_joint` | 右手无名指中间关节 | 手指关节位置目标 |
| 33 | `R_thumb_proximal_pitch_joint` | 右手拇指近端俯仰关节 | 手指关节位置目标 |
| 34 | `L_thumb_distal_joint` | 左手拇指远端关节 | 手指关节位置目标 |
| 35 | `R_thumb_distal_joint` | 右手拇指远端关节 | 手指关节位置目标 |

### 7.3 环境执行顺序

动作在送回环境前是否需要：

- 重排：当前环境动作顺序已由 `PinkInverseKinematicsAction` 定义，适配层若沿用该顺序则不需要重排
- 裁剪：待确认
- 限幅：待确认
- 类型转换：需要保证为 `torch.float32`

## 8. 当前最大风险点

优先记录最容易出错的接口问题。

- `policy` 中既包含机器人状态，也包含 `actions` 历史项；若直接拼状态，容易误把动作历史混入 `observation.state`
- `robot_links_state` 为三维张量 `shape=(1, 55, 13)`，不能在不了解语义时直接平铺
- 图像当前为 `uint8`、`HWC`，若策略期望 `CHW` 或浮点归一化，适配层必须显式转换
- 当前文档记录的是 `num_envs=1` 下的最小检查结果；并行环境时 batch 维会变化
- task 文本来源仍未在当前最小链路里固定
- `gr1_open_microwave` 是 Arena 原生示例环境名，而 LeRobot 文档里常出现 `gr1_microwave`；后续配置文件必须显式区分两套命名语境
