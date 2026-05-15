# 实验笔记

本文件用于记录每天的推进情况、关键命令、观察到的现象与临时结论。

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
  - 这份环境 schema 可以继续作为 `dry_run_policy.py`、`adapters/` 和 `run_eval.py` 的基线
- 当前问题：
  - 相同命令在当前 Codex 代理终端中仍无法复现，问题更接近 GPU 设备透传 / 会话上下文差异，而不是 Arena 环境本身损坏
- 下一步：
  - 在 `dry_run_policy.py` 中完成 checkpoint 单帧前向
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
  - 开始落地 `adapters/` 与 `dry_run_policy.py`

## 记录模板

建议每次实验至少记录：

- 日期
- 目标
- 使用的环境配置
- 是否跑通
- 当前问题
- 下一步
