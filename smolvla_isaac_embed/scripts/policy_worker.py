#!/usr/bin/env python3

"""Python 3.12 SmolVLA 桥接中的策略端 worker。

这个进程应由 Arena 侧的桥接运行器拉起。它绝不能直接把协议数据
打印到 stdout；stdout 仅用于传输有帧边界的消息。人类可读日志应写到
stderr。
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, BinaryIO

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = WORKSPACE_ROOT / "lerobot" / "src"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from smolvla_isaac_embed.bridge_protocol import (  # noqa: E402
    numpy_to_torch_tree,
    read_message,
    summarize_array_tree,
    tensors_to_numpy_tree,
    write_message,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SmolVLA policy inference in the Python 3.12 lerobot env.")
    parser.add_argument("--checkpoint", required=True, help="Policy checkpoint path or Hugging Face repo id.")
    parser.add_argument("--task", required=True, help="Task text passed to the policy.")
    parser.add_argument("--state_dim", type=int, required=True, help="Flattened state dimension from Arena env.")
    parser.add_argument("--action_dim", type=int, required=True, help="Action dimension from Arena env.")
    parser.add_argument("--camera_height", type=int, required=True, help="Camera image height after adaptation.")
    parser.add_argument("--camera_width", type=int, required=True, help="Camera image width after adaptation.")
    parser.add_argument("--policy_device", default=None, help="Override policy device, e.g. cuda:0 or cpu.")
    parser.add_argument("--embodiment", default=None, help="LeRobot embodiment name.")
    parser.add_argument("--object", dest="object_name", default=None, help="LeRobot object name.")
    parser.add_argument("--lerobot_environment", default=None, help="LeRobot environment/task family name.")
    parser.add_argument("--headless", action="store_true", help="Mirror Arena headless setting in env config.")
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[policy_worker] %(message)s",
        stream=sys.stderr,
    )


def _install_protocol_stdout_guard() -> BinaryIO:
    """Reserve the original stdout buffer for the bridge protocol only.

    Third-party policy loading code sometimes uses plain ``print(...)`` during
    startup. The parent process reads framed binary messages from stdout, so any
    text output there can corrupt the protocol. We keep a handle to the original
    stdout buffer for protocol writes and redirect normal stdout text to stderr.
    """

    protocol_stdout = sys.stdout.buffer
    sys.stdout = sys.stderr
    return protocol_stdout


def _build_env_cfg(args: argparse.Namespace, policy_device: str, isaaclab_arena_env_cls: Any) -> Any:
    return isaaclab_arena_env_cls(
        embodiment=args.embodiment,
        object=args.object_name,
        environment=args.lerobot_environment,
        task=args.task,
        enable_cameras=True,
        headless=bool(args.headless),
        device=policy_device,
        state_keys="robot_joint_pos",
        camera_keys="robot_pov_cam",
        state_dim=int(args.state_dim),
        action_dim=int(args.action_dim),
        camera_height=int(args.camera_height),
        camera_width=int(args.camera_width),
    )


def main() -> None:
    args = parse_args()
    protocol_stdout = _install_protocol_stdout_guard()
    _configure_logging()
    logger = logging.getLogger("policy_worker")

    try:
        from lerobot.configs import PreTrainedConfig
        from lerobot.envs.configs import IsaaclabArenaEnv
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        logger.info("loading checkpoint=%s", args.checkpoint)
        policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
        policy_cfg.pretrained_path = args.checkpoint
        if args.policy_device is not None:
            policy_cfg.device = args.policy_device

        env_cfg = _build_env_cfg(args, str(policy_cfg.device), IsaaclabArenaEnv)
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
        policy.eval()

        device_override = {"device": str(policy.config.device)}
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": device_override},
        )

        autocast_context = (
            torch.autocast(device_type=str(policy.config.device).split(":")[0])
            if policy.config.use_amp and str(policy.config.device).startswith("cuda")
            else nullcontext()
        )

        write_message(
            protocol_stdout,
            {
                "type": "ready",
                "policy_type": policy_cfg.type,
                "policy_device": str(policy.config.device),
                "use_amp": bool(policy.config.use_amp),
                "action_dim": int(args.action_dim),
            },
        )
        logger.info("ready policy_type=%s device=%s", policy_cfg.type, policy.config.device)

        while True:
            request = read_message(sys.stdin.buffer)
            request_type = request.get("type")

            if request_type == "reset":
                if hasattr(policy, "reset"):
                    policy.reset()
                write_message(protocol_stdout, {"type": "reset_ack"})
                continue

            if request_type == "infer":
                observation = numpy_to_torch_tree(request["observation"])
                with torch.inference_mode(), autocast_context:
                    policy_input = preprocessor(observation)
                    action = policy.select_action(policy_input)
                    action = postprocessor(action)

                action_payload = tensors_to_numpy_tree(action)
                write_message(
                    protocol_stdout,
                    {
                        "type": "action",
                        "action": action_payload,
                        "action_summary": summarize_array_tree(action_payload),
                    },
                )
                continue

            if request_type == "stop":
                write_message(protocol_stdout, {"type": "stop_ack"})
                return

            raise ValueError(f"Unsupported request type: {request_type!r}")

    except EOFError:
        logger.info("stdin closed; exiting")
    except Exception as exc:  # noqa: BLE001
        error_message = {
            "type": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            write_message(protocol_stdout, error_message)
        except Exception:  # noqa: BLE001
            pass
        logger.error("%s: %s", type(exc).__name__, exc)
        logger.error("%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
