# SmolVLA + IsaacLab Arena 白话实操版

## 这份文件是干什么的

这份计划的目标不是让你一口气看懂整个 lerobot，而是先把现成的 SmolVLA + IsaacLab Arena 闭环跑通，并且知道每一步为什么这么做。

你看完以后，应该能回答这 4 个问题：

1. 现成任务到底怎么启动。
2. 环境观测是怎么变成模型能吃的输入。
3. 模型输出的动作是怎么回到环境里的。
4. 哪些地方一旦名字不对就会直接失败。

## 先记住几个结论

- 当前做的事情是评估和闭环推理，不是训练。
- 当前目标环境是 IsaacLab Arena。
- 当前示例任务是 gr1_microwave。
- 当前示例 checkpoint 是 nvidia/smolvla-arena-gr1-microwave。
- 当前相机键和状态键必须和 checkpoint 对上，不对就要改名。
- 当前动作不是一次性全发出去，而是模型先生成一段，再一拍一拍执行。

## 实操顺序

### 1. 先看最上层的现成命令

先读 [envhub_isaaclab_arena.mdx](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/docs/source/envhub_isaaclab_arena.mdx:1)。

这一步只看三件事：

- 它到底是用什么命令启动的。
- 它用的是哪个 checkpoint。
- 它对环境名、相机名、状态名是怎么写的。

如果这一页你都看不懂，先不要往下翻代码，因为后面的代码只是把这一页写成程序。

### 2. 再看命令入口是怎么把配置装进去的

先读 [lerobot_eval.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/scripts/lerobot_eval.py:16)，再读 [configs/eval.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/configs/eval.py:29)。

你重点搞清楚三件事：

- 传进来的 checkpoint 路径是怎么找到它自己的 config 的。
- rename_map 和 trust_remote_code 最后被放到哪里了。
- 评估结果会输出到哪里。

这一层的作用很简单，就是把你命令行里写的东西整理成程序能用的一份总配置。

### 3. 看闭环主循环

继续看 [lerobot_eval.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/scripts/lerobot_eval.py:98) 和 [eval_main](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/scripts/lerobot_eval.py:520)。

你只需要跟住这一条线：

- 先创建环境。
- 再创建模型。
- 再把环境观测整理成模型输入。
- 模型根据输入算出动作。
- 动作再送回环境。
- 环境往前走一步。
- 重复这个过程。

如果你能用自己的话把这条线讲出来，就说明你已经知道闭环怎么跑了。

### 4. 搞清楚环境原始观测长什么样

看 [IsaaclabArenaEnv](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/envs/configs.py:641)。

这一步主要确认：

- 环境默认叫什么。
- 任务描述是什么。
- 状态维度是多少。
- 动作维度是多少。
- 哪些键是状态，哪些键是相机。

你可以把它理解成：环境一开始吐出来的是原始数据，不是模型想要的格式，所以这里先确认“原件”长什么样。

### 5. 明白为什么要开远程代码信任

看 [envs/factory.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/envs/factory.py:58)。

这里要确认一件事：

- 这个环境不是本地随手注册的一个 gym 环境，而是要从 Hub 里拉代码来创建。

所以如果不允许远程代码执行，环境就起不来。

### 6. 看观测是怎么改成模型输入的

先看 [preprocess_observation](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/envs/utils.py:68)，再看 [IsaaclabArenaProcessorStep](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/processor/env_processor.py:156)。

这一步是全流程里最关键的一步，因为它决定了环境原始数据怎么变成模型认识的格式。

你要记住：

- 原始环境观测里有 policy 和 camera_obs 两大块。
- 状态键决定从 policy 里拿哪些数。
- 相机键决定从 camera_obs 里拿哪路图像。
- 图像会被整理成模型更喜欢的格式。

如果这一步的名字对不上，后面模型再强也没用，因为它根本收不到正确输入。

### 7. 看模型怎么把环境特征和自身配置对齐

看 [make_policy](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/policies/factory.py:480) 附近，再看 [make_pre_post_processors](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/policies/factory.py:236)。

这一步主要确认：

- checkpoint 保存的输入输出特征，会不会和环境特征一一对上。
- 如果名字不一致，rename_map 会不会把它修正掉。
- 预训练 checkpoint 不是临时凭空猜一套处理器，而是尽量沿用它自己保存的那套。

### 8. 看 SmolVLA 自己到底要什么输入

先看 [SmolVLAConfig](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py:24)，再看 [processor_smolvla.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/policies/smolvla/processor_smolvla.py:39)。

这一步主要记住：

- 文本任务会参与处理，不是摆设。
- 状态和动作会按模型规则做归一化。
- 图像不会随便乱改。
- 默认配置里的维度上限，不一定就是当前 checkpoint 的真实可用维度。

### 9. 最后看动作是怎么生成并执行的

看 [modeling_smolvla.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:230)。

你只要搞清楚一件事：

- 模型不是每一帧都重新生成整段动作，而是在需要时生成一段动作，再按顺序一条条取出来执行。

这就是为什么从环境视角看，它仍然是一步一步往前走的。

### 10. 用示例和测试确认容易出错的地方

最后读 [using_smolvla_example.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/examples/tutorial/smolvla/using_smolvla_example.py:14) 和 [test_visual_validation.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/tests/training/test_visual_validation.py:111)。

这一步主要看两件事：

- 相机名字是不是和训练时一致。
- 特征对齐有没有问题。

如果这里不对，程序通常不会帮你自动修好，而是直接报错。

## 你读完后要能写下来的结果

- 当前任务是 eval 或 rollout，不是训练。
- 当前环境是 isaaclab_arena。
- 当前示例任务是 gr1_microwave。
- 当前 checkpoint 是 nvidia/smolvla-arena-gr1-microwave。
- 当前状态键是 robot_joint_pos。
- 当前相机键是 robot_pov_cam_rgb。
- 当前需要的改名是把 robot_pov_cam_rgb 对应到 robot_pov_cam。
- 当前环境默认状态维度是 54。
- 当前环境默认动作维度是 36。
- 当前动作执行方式是：模型先生成一段，环境再逐步执行。

## 怎么判断你已经看懂了

如果你能回答下面 3 个问题，就算真的看懂了：

1. 为什么这个流程必须开 trust_remote_code。
2. 环境原始的 policy 和 camera_obs 是在哪一步变成 observation.state 和 observation.images.* 的。
3. 如果你现在要自己写一个 inspect_obs.py 或 dry_run_policy.py，最小闭环应该抄哪几步。

## 这份计划默认假设

- 现在的目标是先跑通现成闭环，不是先做自定义适配层。
- 现在优先看 lerobot-eval 这条路，不先去扩散看别的 rollout、训练或异步推理脚本。



# 大致要做的工作
最少需要做这几类工作：

1. **先确认环境真实长什么样**
   - 启动 `IsaacLab Arena` 里的目标任务。
   - 记录原始观测里有哪些键，尤其是 `policy` 和 `camera_obs`。
   - 确认当前任务类型是 `eval / rollout`，不是训练。
   - 目标是把“环境到底吐什么数据”先弄清楚。

2. **把观测对齐到 SmolVLA 需要的输入格式**
   - 明确 `state_keys`。
   - 明确 `camera_keys`。
   - 配好 `rename_map`，把环境名映射到模型期望的名字。
   - 明确状态拼接顺序、图像输入格式、任务文本 `task`。
   - 目标是让环境数据能被模型正确读到。

3. **确认 checkpoint 和环境是同一套语义**
   - 加载对应的 SmolVLA checkpoint。
   - 检查 checkpoint 期望的状态名、相机名、动作维度是否和环境一致。
   - 如果不一致，就通过适配层修正，而不是改环境本身。
   - 目标是避免“模型和环境说的不是同一种语言”。

4. **把模型输出动作回写到 Isaac Sim**
   - 明确动作张量维度和动作顺序。
   - 做必要的重排、裁剪、限幅、类型转换。
   - 把动作送回环境执行。
   - 目标是完成“模型出动作，环境真的动起来”。

5. **做一个最小闭环脚本**
   - 先跑一帧推理，确认输入输出通路正确。
   - 再跑短回合 rollout，确认动作顺序没错。
   - 目标是形成一个最小可复现 demo，而不是一开始就追求长 rollout。

6. **把结果固化成配置和文档**
   - 把环境名、checkpoint、键名、维度、动作顺序写进 config 或文档。
   - 保持可复现。
   - 目标是后续换任务时不需要重新摸索。

7. **最后再加安全和扰动层**
   - 基线闭环跑通后，再考虑 safety wrapper、扰动测试、失败案例记录。
   - 目标是先让系统能跑，再让系统更稳。

结合你现在这个仓库的现状，最实际的执行顺序是：

1. `IsaacLab Arena` smoke test。
2. 记录真实的 `state_keys`、`camera_keys`、`action_dim`、动作顺序。
3. 在 `smolvla_isaac_embed/` 里做最小适配。
4. 先做 one-frame inference，再做短 rollout。
5. 成功后再补安全层。