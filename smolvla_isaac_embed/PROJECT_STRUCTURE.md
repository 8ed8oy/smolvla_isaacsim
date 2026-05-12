# 项目结构说明

本文件用于说明 `smolvla_isaac_embed/` 的目录职责。

它既给人读，也给 AI 读。目的不是替代 `README.md`，而是提供更直接的“去哪找什么”的索引。

## 1. 目录总览

```text
smolvla_isaac_embed/
├── README.md
├── AGENTS.md
├── PROJECT_STRUCTURE.md
├── .gitignore
├── configs/
├── scripts/
├── adapters/
├── wrappers/
├── docs/
├── experiments/
├── outputs/
└── tests/
```

## 2. 各目录职责

### `README.md`

面向人类读者的总说明。

适合放：

- 项目目标
- 当前方案
- 目录介绍
- 开发边界

### `AGENTS.md`

面向 AI 助手的协作规则与项目边界说明。

适合放：

- AI 应优先修改哪里
- 哪些目录尽量不要动
- 当前阶段优先级
- 文档语言约定

### `PROJECT_STRUCTURE.md`

面向人类和 AI 的“目录索引文件”。

适合放：

- 每个目录的职责
- 常见文件应放在哪里
- 交接时快速定位入口

### `configs/`

放配置文件。

适合放：

- 运行参数
- `rename_map`
- 环境配置
- 低资源调试配置

### `scripts/`

放可运行脚本。

适合放：

- observation 检查脚本
- 模型 dry run 脚本
- 闭环运行脚本
- 延迟或显存 benchmark 脚本

### `adapters/`

放输入输出适配逻辑。

适合放：

- 环境 observation 转换
- 相机名字映射
- 动作重排与动作格式转换

### `wrappers/`

放安全测试包装逻辑。

适合放：

- 图像扰动
- 状态扰动
- 动作限幅
- 急停机制

### `docs/`

放长期有效的交接文档模板和项目事实。

当前约定：

- `environment.md`：环境与版本
- `interfaces.md`：输入输出接口
- `commands.md`：已验证命令

### `experiments/`

放实验过程文档。

当前约定：

- `notes.md`：日常记录
- `failures.md`：失败案例
- `milestones.md`：阶段进度

### `outputs/`

放程序生成物，不放手写文档。

适合放：

- 日志
- 视频
- 调试转储
- 指标输出

### `tests/`

放最小测试。

优先测试：

- observation 映射
- action 维度和顺序
- safety wrapper 输入输出

## 3. 新文件放置规则

如果你或 AI 要新增文件，优先按下面规则放：

- 想记录项目事实：放 `docs/`
- 想记录实验过程：放 `experiments/`
- 想写运行入口：放 `scripts/`
- 想写输入输出适配：放 `adapters/`
- 想写安全逻辑：放 `wrappers/`
- 想写自动验证：放 `tests/`

## 4. 当前最重要的文件

如果是新对话或新协作者，优先阅读：

1. `README.md`
2. `AGENTS.md`
3. `PROJECT_STRUCTURE.md`
4. `docs/environment.md`
5. `docs/interfaces.md`
6. `docs/commands.md`
7. `experiments/notes.md`
8. `experiments/failures.md`
9. `experiments/milestones.md`
