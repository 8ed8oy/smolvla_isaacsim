#!/usr/bin/env python3

"""Minimal Isaac Sim launch probe for IsaacLab environments.

This script intentionally avoids loading any Arena task. It only launches the
IsaacLab AppLauncher so we can distinguish "Isaac cannot start at all" from
"Arena task code/imports are failing later".
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
