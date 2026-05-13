# 里程碑

建议按阶段更新：

## 阶段一：环境与最小闭环

- [x] 确认 Isaac Sim / IsaacLab 环境版本
- [x] 在 `./.venv-lerobot` 中确认当前工作区 `lerobot` 源码可导入 / 挂载
- [ ] 跑通官方 `SmolVLA` Isaac 示例
- [ ] 在当前可复现终端中重新完成环境观测采样
- [ ] 成功输出动作

当前状态补充：

- 已有一份历史最小观测记录整理进 `docs/interfaces.md`
- 该记录尚未在当前 Codex 代理终端中稳定复现

## 阶段二：适配层稳定化

- [ ] 在代码中实现 `rename_map`
- [ ] 完成状态维度与动作维度的 checkpoint 级对齐验证
- [ ] 完成最小 `run_eval.py`
- [ ] 补充最小测试

当前状态补充：

- 推荐 `rename_map` 已写入 `docs/interfaces.md`
- 候选状态维度与动作维度已写入 `docs/interfaces.md`

## 阶段三：安全测试最小功能

- [ ] 动作限幅
- [ ] 图像遮挡或噪声注入
- [ ] 延迟注入
- [ ] 保存对比日志
