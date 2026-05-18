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

## 2026-05-18 `run_eval.py` 单进程路径无法承载当前 SmolVLA + Arena 组合

- 失败现象：`run_eval.py` 在 Arena Python 3.11 中导入 `lerobot.configs` 时先报 `ModuleNotFoundError: No module named 'draccus'`
- 触发条件：用 `./.conda/lerobot-arena/bin/python` 直接运行单进程 `run_eval.py`
- 初步原因：表面是缺依赖，根因是当前工作区 `lerobot` 要求 Python `>=3.12`，而 Isaac / Arena 主环境为 Python `3.11`
- 是否已解决：已绕开
- 处理结果：新增 `run_eval_bridge.py` 与 `policy_worker.py`，使用 Python 3.11 Arena 进程 + Python 3.12 policy 进程
- 参考命令或相关文件：
  - `smolvla_isaac_embed/scripts/run_eval_bridge.py`
  - `smolvla_isaac_embed/scripts/policy_worker.py`
  - `smolvla_isaac_embed/bridge_protocol.py`
  - `smolvla_isaac_embed/docs/environment.md`

## 2026-05-18 Arena parser 与 shell 命令拼接问题

- 失败现象：`run_eval_bridge.py` 初期只打印 `AppLauncher` warning 后长时间无输出；后续误拼命令时报 `invalid choice: 'source'`
- 触发条件：
  - 早期 bridge 代码读取 Arena parser 的内部 subparser 结构
  - 或者把 `source .../activate` 接在 `run_eval_bridge.py` 命令末尾
- 初步原因：
  - Arena CLI parser 构造函数内部会执行一次 `parse_known_args()`，不是纯 parser 构造逻辑
  - shell 续行反斜杠后有空格或把环境激活命令接到末尾，会让 `source` 进入 Python argv
- 是否已解决：已修正
- 处理结果：
  - `_inject_example_environment()` 改为简单追加配置中的 `example_environment`
  - bridge 增加 `stage=` / `parse_debug=` 日志
  - bridge 对误拼的 `source` / `activate` 增加更明确的错误提示
- 参考命令或相关文件：
  - `smolvla_isaac_embed/scripts/run_eval_bridge.py`
  - `smolvla_isaac_embed/README.md`

## 2026-05-18 policy worker 环境与 checkpoint 下载问题

- 失败现象：
  - worker 初期报 `ModuleNotFoundError: No module named 'torch'`
  - 修正环境后，在线拉取 `nvidia/smolvla-arena-gr1-microwave` 时出现 `Temporary failure in name resolution`
  - Hugging Face retry 链路中还出现 `RuntimeError: Cannot send a request, as the client has been closed.`
- 触发条件：`policy_worker.py` 启动后执行 `PreTrainedConfig.from_pretrained(args.checkpoint)`
- 初步原因：
  - 早期 bridge 对 `./.venv-lerobot/bin/python` 使用 `Path.resolve()`，导致子进程绕过 venv，落到底层 uv CPython
  - 当前环境访问 Hugging Face 的 DNS / 网络不稳定，且本地没有 checkpoint snapshot 时，worker 必须在线下载
- 是否已解决：部分解决
- 处理结果：
  - 已修正 policy Python 路径规范化逻辑，保留 venv 入口
  - 已补齐 `./.venv-lerobot` 的 `lerobot[smolvla]` 依赖
  - 当前推荐使用本地 checkpoint 路径 `smolvla_isaac_embed/models/smolvla-arena-gr1-microwave`
- 参考命令或相关文件：
  - `smolvla_isaac_embed/scripts/run_eval_bridge.py`
  - `smolvla_isaac_embed/scripts/policy_worker.py`
  - `smolvla_isaac_embed/docs/commands.md`

## 记录模板

建议每次记录包括：

- 失败现象
- 触发条件
- 初步原因
- 是否已解决
- 参考命令或相关文件
