# 里程碑

建议按阶段更新：

## 阶段一：环境与最小闭环

- [x] 确认 Isaac Sim / IsaacLab 环境版本
- [x] 在 `./.venv-lerobot` 中确认当前工作区 `lerobot` 源码可导入 / 挂载
- [ ] 跑通官方 `SmolVLA` Isaac 示例
- [x] 在本机可复现终端中重新完成环境观测采样
- [x] 通过 `dry_run_policy.py` / `run_eval.py` 落地最小动作输出路径

当前状态补充：

- `2026-05-15` 已在本机终端成功复核 `arena_smoke_check.py`
- 关键结果：`reset_ok`、`device=cuda:0`、`action_space_shape=(1, 36)`、`smoke_test_ok`
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
- 测试文件已落地到 `smolvla_isaac_embed/tests/`
- 当前工作区暂未补齐 `pytest`，因此测试执行记录仍需在具备 `pytest` 的环境中补做

## 阶段三：安全测试最小功能

- [ ] 动作限幅
- [ ] 图像遮挡或噪声注入
- [ ] 延迟注入
- [ ] 保存对比日志
