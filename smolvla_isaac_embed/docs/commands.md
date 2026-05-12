# 已验证命令

本文件只记录“真正执行过并有结果”的命令。

不要把尚未验证的命令写成既成事实。

## 1. 环境检查

```bash
python3 --version
./.venv/bin/python --version
conda --version
nvidia-smi
curl -I https://conda.anaconda.org
curl -I https://pypi.nvidia.com
```

结果：

- 是否成功：部分成功
- 备注：`python3` 为 `3.13.12`，工作区 `.venv` 为 `3.12.13`，`conda` 为 `26.1.1`；`conda.anaconda.org` 与 `pypi.nvidia.com` 可访问；`nvidia-smi` 当前失败，说明 GPU 驱动仍需修复

## 2. Isaac 环境创建

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

结果：

- 是否成功：部分成功
- 备注：`./.conda/lerobot-arena` 已成功创建，环境 Python 为 `3.11.15`；ZJU 镜像源在 TLS 上不稳定，最终改用官方 `conda-forge` 成功建环境；`isaacsim 5.1.0` 安装已启动并完成多个大 wheel 下载，但本记录更新时尚未做完全部安装校验

## 3. SmolVLA 加载

```bash
# 待填写
```

结果：

- 是否成功：
- 备注：

## 4. 最小闭环运行

```bash
# 待填写
```

结果：

- 是否成功：
- 输出位置：
- 备注：

## 5. 调试命令

用于排查 observation、动作维度、显存等。

```bash
du -sh ./.conda/lerobot-arena
du -sh /tmp/pip-unpack-* 2>/dev/null | sort -h | tail
find /tmp -maxdepth 2 -type d -name 'pip-*' | sed -n '1,20p'
conda activate /media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/.conda/lerobot-arena
```

## 6. 当前推荐命令

这一节只保留“当前最推荐使用的那几条命令”。

### 6.1 推荐的环境检查命令

```bash
conda activate /media/bed8oy/3T/01_workspace/vla_models/smolvla_workspace/.conda/lerobot-arena
python --version
pip show isaacsim
nvidia-smi
```

### 6.2 推荐的最小运行命令

```bash
./smolvla_isaac_embed/scripts/setup_isaaclab_arena_env.sh
```
