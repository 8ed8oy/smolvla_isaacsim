# Tests

这里放 `smolvla_isaac_embed` 的最小回归测试。

## 推荐运行环境

建议使用当前已经验证过的 Arena 环境：

- 环境路径：`./.conda/lerobot-arena`
- Python：`3.11.15`
- 依赖情况：已确认同时具备 `torch` 与 `pytest`

不建议默认使用：

- `./.venv`，因为当前没有安装 `pytest`
- `./.venv-lerobot`，因为当前也没有安装 `pytest`

## 标准命令

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ./.conda/lerobot-arena/bin/python \
  -m pytest -q \
  smolvla_isaac_embed/tests/test_bridge_protocol.py \
  smolvla_isaac_embed/tests/test_env_adapter.py \
  smolvla_isaac_embed/tests/test_action_adapter.py
```

## 覆盖范围

当前这组测试主要覆盖纯 adapter / 参数层逻辑，不需要启动 Isaac App：

- `test_bridge_protocol.py`
  - 验证 bridge 消息 framing 的读写 round-trip
  - 验证 observation/action 的 tensor <-> numpy 树转换
  - 验证调试摘要里的 `shape` / `dtype`
- `test_env_adapter.py`
  - 验证 `policy.robot_joint_pos -> observation.state`
  - 验证 `camera_obs.robot_pov_cam_rgb -> observation.images.robot_pov_cam`
- `test_action_adapter.py`
  - 验证 `36` 维动作 identity 适配器
  - 验证 1D 输入升 batch
  - 验证 2D 输入直通
  - 验证错误维度、错误 rank、非法类型报错
  - 验证 `describe()` 与 `ACTION_ORDER` 一致
  - 验证 `preview_action_with_order()` 与 `ACTION_ORDER` 对齐

## 已验证结果

- 历史上，旧版最小测试集合曾在 Arena 环境中通过
- 当前 `test_action_adapter.py` 已扩展，建议按上面的标准命令重新执行一次完整回归
