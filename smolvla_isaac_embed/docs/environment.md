# 环境说明

本文件用于记录本项目真正验证过的运行环境，而不是“理论上应该可以”的环境。

## 1. 主机信息

- 操作系统：Ubuntu 22.04.5 LTS
- 内核版本：6.8.0-111-generic
- 机器类型：待补充
- GPU 型号：待补充
- GPU 显存：待补充
- CUDA 驱动版本：待补充
- 当前 GPU 状态：`nvidia-smi` 失败，驱动尚未正常响应

## 2. Python 与虚拟环境

- 系统 Python：`3.13.12`
- 工作区 `.venv`：`3.12.13`
- 当前推荐环境管理方式：`conda`
- 当前环境路径：`/media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/.conda/lerobot-arena`
- 当前环境 Python：`3.11.15`
- conda 版本：`26.1.1`

## 3. 核心软件版本

- Isaac Sim：`5.1.0` 安装中
- IsaacLab：未安装
- IsaacLab-Arena / LeIsaac：未安装
- lerobot：工作区源码已存在，尚未装入新环境
- torch：待 `isaacsim` 安装完成后确认
- torchvision：待 `isaacsim` 安装完成后确认
- numpy：目标版本 `1.26.0`

## 4. 关键安装命令

按实际执行顺序记录，确保下一个对话或下一个人可以复现。

```bash
mkdir -p .cache .conda/pkgs

XDG_CACHE_HOME=$PWD/.cache \
CONDA_PKGS_DIRS=$PWD/.conda/pkgs \
conda create -y -p $PWD/.conda/lerobot-arena python=3.11 ffmpeg=7.1.1 \
  --solver classic \
  --override-channels \
  -c https://conda.anaconda.org/conda-forge

PIP_CACHE_DIR=$PWD/.cache/pip \
  ./.conda/lerobot-arena/bin/pip install \
  "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com
```

## 5. 已验证结果

- [ ] 可成功导入 `lerobot`
- [ ] 可成功导入 Isaac 相关环境
- [ ] 可创建环境实例
- [ ] 可加载 SmolVLA checkpoint
- [ ] 可单步前向输出动作
- [ ] 可闭环驱动环境
- [x] 可访问 `conda.anaconda.org`
- [x] 可访问 `pypi.nvidia.com`
- [x] 已创建 Python 3.11 + ffmpeg 的工作区本地 conda 环境

## 6. 已知兼容性问题

记录版本冲突、显存问题、依赖回退等。

- 问题：ZJU 镜像源在 `conda` 安装阶段 TLS 握手不稳定
  - 现象：`terms.json`、`notices.json` 与 `current_repodata.json.zst` 出现 `SSLEOFError`
  - 原因：镜像侧 SSL / 超时不稳定，不是工作区命令本身错误
  - 解决方式：优先尝试现有镜像；失败后切到官方 `conda-forge`

- 问题：GPU 驱动当前未被 `nvidia-smi` 正常识别
  - 现象：`NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`
  - 原因：驱动未正常加载，或当前会话无法访问驱动
  - 解决方式：在继续做 Isaac Sim 启动验证前，先修复驱动侧问题

## 7. 当前推荐环境

这一节只写“目前项目最建议使用的环境组合”。

- 推荐 Python：`3.11`
- 推荐 Isaac：`Isaac Sim 5.1.0` + `IsaacLab 2.3.0` + `IsaacLab-Arena release/0.1.1`
- 推荐 lerobot：使用当前工作区 `lerobot/`，以 editable 模式安装 `.[evaluation,smolvla]`
- 备注：镜像可先试 ZJU，若出现 TLS EOF，直接切换到官方 `conda-forge`
