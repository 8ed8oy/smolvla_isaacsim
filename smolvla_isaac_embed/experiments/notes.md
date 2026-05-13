# 实验笔记

本文件用于记录每天的推进情况、关键命令、观察到的现象与临时结论。

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
