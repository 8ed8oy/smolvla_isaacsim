# SmolVLA + IsaacLab Arena 闭环阅读顺序清单

## Summary

按“先命令入口，再闭环主线，再环境适配，再策略输入输出，最后看校验与示例”的顺序读。目标不是理解整个 `lerobot`，而是尽快回答这 6 个问题：

1. 现成闭环到底用什么命令跑。
2. `lerobot-eval` 在哪里创建环境、加载 checkpoint、做推理、把动作送回环境。
3. IsaacLab Arena 的原始观测长什么样，在哪里被改成 LeRobot / SmolVLA 需要的格式。
4. SmolVLA checkpoint 期望哪些输入键、相机键、状态维度、动作维度。
5. `rename_map` 为什么必须配，什么时候会炸。
6. 你后面自己写 `smolvla_isaac_embed/` 适配层时，应该复刻哪一段最小链路。

## Reading Order

1. 先读 Arena 文档基线命令  
   读 [envhub_isaaclab_arena.mdx](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/docs/source/envhub_isaaclab_arena.mdx:1)。  
   重点只抓现成闭环的事实：
   - 命令入口是 `lerobot-eval`
   - 现成 SmolVLA checkpoint 是 `nvidia/smolvla-arena-gr1-microwave`
   - 环境类型是 `isaaclab_arena`
   - 当前示例的 `rename_map`、`state_keys`、`camera_keys`、`eval.batch_size`、`trust_remote_code` 怎么写

2. 再读 `lerobot-eval` 的 CLI 入口和配置装载  
   先看 [lerobot_eval.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/scripts/lerobot_eval.py:16)，再看 [configs/eval.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/configs/eval.py:29)。  
   你要确认：
   - `--policy.path` 如何自动反解 checkpoint 自带的 `config.json`
   - `rename_map` 和 `trust_remote_code` 在总配置里处于什么位置
   - 输出目录和 job name 怎么生成

3. 把闭环主循环读透  
   回到 [lerobot_eval.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/scripts/lerobot_eval.py:98) 和 [eval_main](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/scripts/lerobot_eval.py:520)。  
   按这条链看：
   - `make_env`
   - `make_policy`
   - `make_pre_post_processors`
   - `make_env_pre_post_processors`
   - `preprocess_observation`
   - `env_preprocessor`
   - `preprocessor`
   - `policy.select_action`
   - `postprocessor`
   - `env.step`
   
   读完这里，你应该能口头复述整个闭环。

4. 读 IsaacLab Arena 环境配置，不要先读别的 env  
   看 [IsaaclabArenaEnv](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/envs/configs.py:641)。  
   重点记住：
   - 默认 `environment="gr1_microwave"`
   - 默认 `task="Reach out to the microwave and open it."`
   - 默认 `state_dim=54`
   - 默认 `action_dim=36`
   - `state_keys` / `camera_keys` 是如何驱动特征声明与 env processor 的

5. 读环境创建逻辑，明确为什么需要 `trust_remote_code=True`  
   看 [envs/factory.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/envs/factory.py:58)。  
   你要弄清：
   - `isaaclab_arena` 不是本地 gym id，而是 Hub env
   - `hub_path=nvidia/isaaclab-arena-envs`
   - `make_env` 会下载并执行 Hub 里的 `env.py`
   - 所以没有 `trust_remote_code=True` 就不会过

6. 读观测适配链，重点是 Arena 这一段  
   先看 [preprocess_observation](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/envs/utils.py:68)，再看 [IsaaclabArenaProcessorStep](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/processor/env_processor.py:156)。  
   这里是你后续自己写 `smolvla_isaac_embed/adapters/` 的直接蓝本。要抓住：
   - 原始 Arena 观测顶层键是 `policy` 和 `camera_obs`
   - `state_keys` 决定从 `obs["policy"]` 取哪些张量并按顺序拼到 `observation.state`
   - `camera_keys` 决定从 `obs["camera_obs"]` 取哪些图像并映射成 `observation.images.<cam>`
   - 图像会从 `BHWC uint8` 变成 `BCHW float32 [0,1]`

7. 读策略实例化和特征对齐逻辑  
   看 [make_policy` 附近逻辑](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/policies/factory.py:480) 和 [make_pre_post_processors](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/policies/factory.py:236)。  
   你要确认：
   - checkpoint 的 `input_features` / `output_features` 会和 env 特征做对齐
   - `rename_map` 存在时，视觉特征一致性校验会被绕开
   - 预训练 checkpoint 会直接加载保存好的 preprocessor / postprocessor，而不是临时重建一套“猜的”

8. 读 SmolVLA 自己需要什么输入  
   先看 [SmolVLAConfig](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py:24)，再看 [processor_smolvla.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/policies/smolvla/processor_smolvla.py:39)。  
   重点理解：
   - 预处理顺序是 `rename -> add batch -> newline task -> tokenize -> device -> normalize`
   - `task` 文本不是装饰品，而是真进 tokenizer
   - `STATE` 和 `ACTION` 默认是 `MEAN_STD` 归一化
   - `VISUAL` 默认不做额外归一化
   - base config 里 `max_state_dim=32`、`max_action_dim=32` 是模型内部上限；真正运行时以 checkpoint 和 env feature 为准，不要只看默认值误判

9. 最后读 SmolVLA 推理动作怎么吐出来  
   看 [modeling_smolvla.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:230)。  
   你要明确：
   - `select_action()` 不是每一步都全量生成，它在队列空时才生成一整个 action chunk
   - 真正送回环境的是 `popleft()` 后的一步动作
   - 所以从环境角度看仍然是一拍一动作的闭环

10. 最后用测试和示例补“容易错的点”  
   读 [using_smolvla_example.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/examples/tutorial/smolvla/using_smolvla_example.py:14) 和 [test_visual_validation.py](/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/lerobot/tests/training/test_visual_validation.py:111)。  
   这里主要补两件事：
   - 相机 key 必须匹配训练时命名，否则就靠 `rename_map`
   - 没有正确的 camera naming / feature alignment，闭环会在前处理或特征校验阶段失败，不会“自动适配”

## 读完后你必须写下来的事实

- 当前现成任务类型：`rollout/eval`，不是训练，不是 safety test。
- 当前现成环境：`env.type=isaaclab_arena`，示例环境 `gr1_microwave`。
- 当前现成 checkpoint：`nvidia/smolvla-arena-gr1-microwave`。
- 当前现成 `state_keys`：`robot_joint_pos`。
- 当前现成 `camera_keys`：`robot_pov_cam_rgb`。
- 当前现成 `rename_map`：`observation.images.robot_pov_cam_rgb -> observation.images.robot_pov_cam`。
- 当前 Arena env 默认维度：`state_dim=54`，`action_dim=36`。
- 当前动作执行模式：SmolVLA 内部 chunk 生成，环境侧逐步执行单步 action。

## Acceptance

如果你按这个顺序读完，你应该已经能自己回答这 3 个问题：

1. 为什么现成命令必须带 `trust_remote_code=True` 和 `rename_map`。
2. Arena 的 `policy` / `camera_obs` 是在哪一层被改成 `observation.state` / `observation.images.*`。
3. 如果你下一步自己写 `inspect_obs.py` 和 `dry_run_policy.py`，应该复刻 `lerobot-eval` 里的哪一段最小闭环。

## Assumptions

- 目标是“跑通现成 Arena 闭环”，不是先做自定义适配层，也不是先训练新 checkpoint。
- 优先理解 `lerobot-eval` 路径；`lerobot-rollout`、异步推理、RTC、训练脚本都暂时不在第一轮阅读范围内。
