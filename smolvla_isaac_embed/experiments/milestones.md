# 里程碑

建议按阶段更新：

## 阶段一：环境与最小闭环

- [x] 确认 Isaac Sim / IsaacLab 环境版本
- [x] 在 `./.venv-lerobot` 中确认当前工作区 `lerobot` 源码可导入 / 挂载
- [x] 跑通官方 `SmolVLA` Isaac 示例
- [x] 在本机可复现终端中重新完成环境观测采样
- [x] 通过 `run_eval_single_frame.py` / `run_eval.py` 落地最小动作输出路径
- [x] 通过 `run_eval_bridge.py` 完成最小 rollout 并成功生成视频

当前状态补充：

- `2026-05-15` 已在本机终端成功复核 `arena_smoke_check.py`
- 关键结果：`reset_ok`、`device=cuda:0`、`action_space_shape=(1, 36)`、`smoke_test_ok`
- `2026-05-18` 已成功运行 bridge rollout 并生成视频
- 当前已知运行基线：
  - checkpoint：`smolvla_isaac_embed/models/smolvla-arena-gr1-microwave`
  - GPU 显存峰值：`7.05 GB`
  - GPU 功率：`70 W / 170 W`
- 当前 Codex 代理终端仍不能稳定继承同样的 GPU 设备访问条件，因此“本机复现成功”与“代理终端可复现”需要分开记录

## 阶段二：适配层稳定化

- [x] 在代码中实现最小 `rename_map` 路径
- [ ] 完成状态维度与动作维度的 checkpoint 级对齐验证
- [x] 完成最小 `run_eval.py`
- [x] 补充最小测试文件

当前状态补充：

- 当前基线配置已固化到 `smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml`
- 最小观测适配器已支持 `robot_joint_pos -> observation.state`
- 最小图像适配器已支持 `robot_pov_cam_rgb -> observation.images.robot_pov_cam`
- 最小动作适配器已支持 `shape=(*, 36)` 的 GR1 Pink 动作直通校验
- bridge 协议已兼容 Python 3.11 / 3.12 双环境与不同 `numpy` 版本之间的数组传输
- 测试文件已落地到 `smolvla_isaac_embed/tests/`
- `2026-05-15` 已在 `./.conda/lerobot-arena` 中补跑 `pytest`，并确认 `smolvla_isaac_embed/tests/test_env_adapter.py` 与 `smolvla_isaac_embed/tests/test_action_adapter.py` 可以稳定通过

## 阶段三：安全测试最小功能

- [ ] 动作限幅
- [ ] 图像遮挡或噪声注入
- [ ] 延迟注入
- [ ] 保存对比日志
