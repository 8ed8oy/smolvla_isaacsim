# 失败案例记录

本文件用于记录失败案例与排查结果，避免重复踩坑。

## 2026-05-13 Codex 代理终端中 Isaac App 无法完成 GPU 初始化

- 失败现象：`isaac_app_probe.py` 与 `arena_smoke_check.py` 都未进入 `reset_ok`，启动阶段即报错退出
- 触发条件：在 Codex 代理终端中执行 headless + cameras 的 Isaac Sim / Arena 启动命令
- 初步原因：当前代理终端上下文未稳定继承可用的 NVIDIA 驱动 / Vulkan / CUDA 设备访问条件
- 是否已解决：否
- 关键报错：`NVML_ERROR_DRIVER_NOT_LOADED`、`No device could be created`、`Failed to create primary CUDA context`
- 参考命令或相关文件：
  - `smolvla_isaac_embed/scripts/isaac_app_probe.py`
  - `smolvla_isaac_embed/scripts/arena_smoke_check.py`
  - `smolvla_isaac_embed/docs/commands.md`
  - `smolvla_isaac_embed/docs/environment.md`

## 2026-05-13 当前工作区 `lerobot` 无法直接安装到 Arena Python 3.11 环境

- 失败现象：在 Arena 环境中对 `lerobot/src/lerobot/motors/motors_bus.py` 做编译检查失败
- 触发条件：使用 `./.conda/lerobot-arena/bin/python -m py_compile ...`
- 初步原因：当前工作区 `lerobot` 已使用 Python 3.12 语法，而 Arena 主环境为 Python 3.11
- 是否已解决：已绕开，尚未根治
- 关键结论：不要把当前工作区这份 `lerobot` 直接装入 Isaac Sim 主环境
- 参考命令或相关文件：
  - `smolvla_isaac_embed/docs/commands.md`
  - `smolvla_isaac_embed/docs/environment.md`
  - `lerobot/pyproject.toml`

## 记录模板

建议每次记录包括：

- 失败现象
- 触发条件
- 初步原因
- 是否已解决
- 参考命令或相关文件
