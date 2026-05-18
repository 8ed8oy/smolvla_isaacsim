# 实验笔记

本文件用于记录每天的推进情况、关键命令、观察到的现象与临时结论。

## 2026-05-18 bridge rollout 成功生成视频

- 日期：`2026-05-18`
- 目标：完成 `run_eval_bridge.py` 的最小闭环 rollout，并确认视频录制链路可用
- 使用的脚本：
  - `smolvla_isaac_embed/scripts/run_eval_bridge.py`
- 使用的 checkpoint：
  - `smolvla_isaac_embed/models/smolvla-arena-gr1-microwave`
- 是否跑通 rollout：
  - 成功
- 是否生成视频：
  - 成功
- GPU 观测：
  - 显存峰值：`7.05 GB`
  - 功率：`70 W / 170 W`
- 当前结论：
  - `Python 3.11 Arena + Python 3.12 policy worker` 的 bridge 基线路径已经可以完整跑通
  - 本地 checkpoint、bridge 协议、最小观测适配、36 维动作直通和视频录制链路均已通过一次端到端验证
  - 当前结果已经满足“最小闭环 + 视频产出”的阶段目标，可以把注意力从环境兼容问题转向动作语义验证和安全包装层
- 下一步：
  - 固化一次动作维度与动作顺序检查
  - 补一条单 episode 的 rollout 指标记录
  - 开始接入最小 safety wrapper

## 2026-05-15 补跑最小单元测试

- 日期：`2026-05-15`
- 目标：验证 `smolvla_isaac_embed/tests/` 下的观测适配器与动作适配器测试，并把可复现命令写入文档
- 使用的环境配置：
  - Arena 环境：`./.conda/lerobot-arena`
  - Python：`3.11.15`
  - 说明：`./.venv` 与 `./.venv-lerobot` 里暂未安装 `pytest`
- 执行命令：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python \
  -m pytest -q \
  smolvla_isaac_embed/tests/test_env_adapter.py \
  smolvla_isaac_embed/tests/test_action_adapter.py
```

- 是否跑通：
  - 成功
- 关键输出：
  - `3 passed in 4.23s`
- 额外复核：
  - `OMNI_KIT_ACCEPT_EULA=YES ./.conda/lerobot-arena/bin/python -m pytest -q smolvla_isaac_embed/tests`
  - 结果同样为 `3 passed in 4.22s`
- 当前结论：
  - 这两组测试可以作为 `MinimalIsaacEnvAdapter` 与 `MinimalIsaacActionAdapter` 的最小回归检查
  - 由于测试依赖 `torch` 与 `pytest`，当前更适合放在 Arena 环境里执行
- 下一步：
  - 把测试命令继续保存在 `docs/commands.md`
  - 如果后续需要在 `.venv` 中执行，再单独补最小测试依赖

## 2026-05-15 本机终端成功复核 Arena smoke check

- 日期：`2026-05-15`
- 目标：在本机 GPU / 显示上下文正常的终端里，重新执行 `arena_smoke_check.py`，确认 Arena 最小链路可启动并完成一次 reset
- 使用的环境配置：
  - Arena 环境：`./.conda/lerobot-arena`
  - Python：`3.11`
  - 环境：`gr1_open_microwave`
  - embodiment：`gr1_pink`
  - object：`mustard_bottle`
  - 运行参数：`--headless --enable_cameras --num_steps 0`
- 执行命令：

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

- 是否跑通：
  - 成功
- 关键输出：
  - `reset_ok`
  - `device=cuda:0`
  - `action_space_shape=(1, 36)`
  - `observation_top_level_keys=['camera_obs', 'policy']`
  - `smoke_test_ok`
- 当前结论：
  - Arena / Isaac Sim / IsaacLab 主链路在本机正确 GPU 会话中可正常启动
  - 当前最小环境动作维度为 `36`
  - 当前最小观测顶层 keys 为 `camera_obs` 与 `policy`
  - 这份环境 schema 可以继续作为 `run_eval_single_frame.py`、`adapters/` 和 `run_eval.py` 的基线
- 当前问题：
  - 相同命令在当前 Codex 代理终端中仍无法复现，问题更接近 GPU 设备透传 / 会话上下文差异，而不是 Arena 环境本身损坏
- 下一步：
  - 在 `run_eval_single_frame.py` 中完成 checkpoint 单帧前向
  - 用最小适配器串起 `run_eval.py`
  - 补齐最小测试与阶段里程碑记录

## 2026-05-13 文档同步核查

- 日期：`2026-05-13`
- 目标：核对 `docs/` 与 `experiments/` 下记录文档是否与当前工作区事实一致
- 使用的环境配置：
  - Arena 环境：`./.conda/lerobot-arena`，Python `3.11.15`
  - LeRobot 辅助环境：`./.venv-lerobot`，Python `3.12.13`
- 是否跑通：
  - `setup_isaaclab_arena_env.sh` 跑通
  - `isaac_app_probe.py` 与 `arena_smoke_check.py` 在当前 Codex 代理终端中未跑通
- 当前问题：
  - `commands.md` 原先仍把 smoke test 写成“待执行”，与本轮核查事实不同步
  - `interfaces.md` 原先把环境 `policy` 可见键与推荐 `state_keys` 混在一起，容易误读
  - `interfaces.md` 原先的 `rename_map` 还是空对象，与官方 `SmolVLA` Arena 示例不一致
  - `milestones.md`、`failures.md`、`notes.md` 还没有沉淀这轮核查结论
- 下一步：
  - 在用户本机可用 GPU 终端里重新执行 `arena_smoke_check.py`
  - 补一条真正的 SmolVLA checkpoint 加载记录
  - 开始落地 `adapters/` 与 `run_eval_single_frame.py`

## 记录模板：rollout / video 复核

建议后续每次完整 rollout 都补一条记录，字段如下：

- 日期
- 目标
- 使用的脚本
- 使用的 checkpoint
- 是否跑通 rollout
- 是否生成视频
- 输出路径
- 当前阻塞点
- 下一步

可直接复制的模板：

```md
## YYYY-MM-DD rollout 复核

- 日期：`YYYY-MM-DD`
- 目标：
- 使用的脚本：
- 使用的 checkpoint：
- 是否跑通 rollout：
- 是否生成视频：
- 输出路径：
- 当前阻塞点：
- 下一步：
```

## 记录模板

建议每次实验至少记录：

- 日期
- 目标
- 使用的环境配置
- 是否跑通
- 当前问题
- 下一步
