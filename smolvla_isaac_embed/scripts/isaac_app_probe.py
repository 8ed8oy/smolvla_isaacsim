#!/usr/bin/env python3

"""用于 IsaacLab 环境的最小 Isaac Sim 启动探针。

这个脚本刻意不加载任何 Arena 任务。它只启动 IsaacLab 的 AppLauncher，
这样我们就能区分“Isaac 根本无法启动”和“Arena 任务代码/导入在后面失败”。

用法：
- 当你怀疑 Isaac 启动本身有问题时，先运行这个脚本。
- 典型示例：``python smolvla_isaac_embed/scripts/isaac_app_probe.py --headless --enable_cameras``。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Isaac Sim launch probe")
    parser.add_argument("--headless", action="store_true", help="Launch Isaac Sim headless.")
    parser.add_argument(
        "--enable_cameras",
        action="store_true",
        help="Enable camera rendering so the rendering experience file is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=args.enable_cameras)
    try:
        print("app_started")
    finally:
        app_launcher.app.close()
        print("app_closed")


if __name__ == "__main__":
    main()
