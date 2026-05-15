#!/usr/bin/env python3

"""运行一个最小的 SmolVLA -> IsaacLab-Arena rollout。

这个脚本的范围刻意保持得很窄：

- 只用一个环境
- 只用一个策略 checkpoint
- 只跑少量 rollout 步数
- 只保留最小观测路径：
  - policy.robot_joint_pos
  - camera_obs.robot_pov_cam_rgb

用法：
- 当你想测试把策略动作回写到环境里的最短端到端控制回路时，运行这个脚本。
- 典型示例：``python smolvla_isaac_embed/scripts/run_eval.py --config smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml --max_steps 5 --num_episodes 1``。
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = WORKSPACE_ROOT / "lerobot" / "src"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from isaaclab_arena.examples.example_environments.cli import (  # noqa: E402
    get_arena_builder_from_cli,
    get_isaaclab_arena_example_environment_cli_parser,
)
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext  # noqa: E402
from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.envs.configs import IsaaclabArenaEnv  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402

from smolvla_isaac_embed.adapters import (  # noqa: E402
    MinimalIsaacActionAdapter,
    MinimalIsaacEnvAdapter,
    TARGET_IMAGE_KEY,
)

DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "smolvla_isaac_embed" / "configs" / "gr1_open_microwave_smolvla.toml"


def _load_runtime_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    policy_cfg = raw_config.get("policy", {})
    env_cfg = raw_config.get("env", {})
    if not isinstance(policy_cfg, dict) or not isinstance(env_cfg, dict):
        raise ValueError("Config sections [policy] and [env] must be TOML tables.")

    return {
        "checkpoint": policy_cfg.get("checkpoint"),
        "task": policy_cfg.get("task"),
        "example_environment": env_cfg.get("example_environment"),
        "lerobot_environment": env_cfg.get("lerobot_environment"),
        "embodiment": env_cfg.get("embodiment"),
        "object": env_cfg.get("object"),
        "headless": env_cfg.get("headless"),
        "enable_cameras": env_cfg.get("enable_cameras"),
    }


def _parser_default_config_path() -> str:
    return str(DEFAULT_CONFIG_PATH.relative_to(WORKSPACE_ROOT))


def _parse_known_config(argv: list[str]) -> tuple[Path, dict[str, Any]]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=_parser_default_config_path())
    parsed, _ = pre_parser.parse_known_args(argv)
    config_path = Path(parsed.config)
    if not config_path.is_absolute():
        config_path = (WORKSPACE_ROOT / config_path).resolve()
    return config_path, _load_runtime_config(config_path)


def _consume_option_tokens(argv: list[str], index: int, option_action: argparse.Action) -> int:
    token = argv[index]
    if "=" in token:
        return index + 1

    nargs = option_action.nargs
    if nargs in (0, None):
        return index + (0 if nargs == 0 else 2)
    if nargs == "?":
        if index + 1 < len(argv) and not argv[index + 1].startswith("-"):
            return index + 2
        return index + 1
    if nargs in ("*", "+"):
        next_index = index + 1
        while next_index < len(argv) and not argv[next_index].startswith("-"):
            next_index += 1
        return next_index
    if isinstance(nargs, int):
        return index + 1 + nargs
    return index + 1


def _inject_example_environment(argv: list[str], parser: argparse.ArgumentParser, example_environment: str | None) -> list[str]:
    if not example_environment:
        return argv

    subparser_action = next(
        (action for action in parser._actions if isinstance(action, argparse._SubParsersAction)),
        None,
    )
    if subparser_action is None:
        return argv
    if example_environment not in subparser_action.choices:
        raise ValueError(f"Unsupported example environment in config: {example_environment}")

    choices = set(subparser_action.choices)
    option_actions = parser._option_string_actions
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in choices:
            return argv
        if token == "--":
            return argv[:index] + [example_environment] + argv[index:]
        if token.startswith("-"):
            option_action = option_actions.get(token.split("=", 1)[0])
            if option_action is None:
                return argv[:index] + [example_environment] + argv[index:]
            index = _consume_option_tokens(argv, index, option_action)
            continue
        return argv[:index] + [example_environment] + argv[index:]
    return argv + [example_environment]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]

    config_path, config = _parse_known_config(argv)
    parser = get_isaaclab_arena_example_environment_cli_parser()
    parser.description = "Run a minimal rollout on IsaacLab-Arena using the embedded adapters."
    parser.add_argument(
        "--config",
        default=str(config_path),
        help="TOML config path for checkpoint and environment defaults.",
    )
    parser.add_argument(
        "--checkpoint",
        default=config.get("checkpoint"),
        help="Policy checkpoint path or Hugging Face repo id.",
    )
    parser.add_argument(
        "--task",
        default=config.get("task") or "open microwave",
        help="Task text passed to the policy.",
    )
    parser.add_argument(
        "--policy_device",
        default=None,
        help="Override policy device, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5,
        help="Maximum rollout steps per episode.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="How many episodes to run.",
    )
    parser.add_argument(
        "--preview_dims",
        type=int,
        default=8,
        help="How many leading action values to print per step.",
    )
    parser.set_defaults(
        embodiment=config.get("embodiment"),
        object=config.get("object"),
        headless=config.get("headless"),
        enable_cameras=config.get("enable_cameras"),
    )
    argv = _inject_example_environment(argv, parser, config.get("example_environment"))
    args = parser.parse_args(argv)
    if not args.checkpoint:
        parser.error("checkpoint is required either via --checkpoint or [policy].checkpoint in --config.")
    if args.num_episodes < 1:
        parser.error("--num_episodes must be >= 1.")
    if args.max_steps < 1:
        parser.error("--max_steps must be >= 1.")
    args.lerobot_environment = config.get("lerobot_environment") or config.get("example_environment")
    return args


def _preview_tensor(tensor: torch.Tensor, n: int) -> list[float]:
    flat = tensor.detach().to("cpu").reshape(-1)
    return [float(value) for value in flat[: max(n, 0)].tolist()]


def _as_python_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().to("cpu").any().item())
    return bool(value)


def _as_python_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().to("cpu").reshape(-1)[0].item())
    return float(value)


def _build_env_cfg(args: argparse.Namespace, env: Any, first_observation: dict[str, Any]) -> IsaaclabArenaEnv:
    env_adapter = MinimalIsaacEnvAdapter(task=args.task)
    adapted = env_adapter.adapt(first_observation)
    state_dim = int(adapted["observation.state"].shape[-1])
    camera_height = int(adapted[TARGET_IMAGE_KEY].shape[-2])
    camera_width = int(adapted[TARGET_IMAGE_KEY].shape[-1])
    action_dim = int(env.action_space.shape[-1])

    return IsaaclabArenaEnv(
        embodiment=getattr(args, "embodiment", None),
        object=getattr(args, "object", None),
        environment=getattr(args, "lerobot_environment", None),
        task=args.task,
        enable_cameras=True,
        headless=bool(getattr(args, "headless", False)),
        device=str(env.device),
        state_keys="robot_joint_pos",
        camera_keys="robot_pov_cam",
        state_dim=state_dim,
        action_dim=action_dim,
        camera_height=camera_height,
        camera_width=camera_width,
    )


def main() -> None:
    args = parse_args()

    with SimulationAppContext(args):
        env = get_arena_builder_from_cli(args).make_registered()
        try:
            if args.seed is not None:
                env.seed(args.seed)
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)

            first_obs, info = env.reset()
            del info

            env_cfg = _build_env_cfg(args, env, first_obs)
            env_adapter = MinimalIsaacEnvAdapter(task=args.task)
            action_adapter = MinimalIsaacActionAdapter(expected_action_dim=int(env.action_space.shape[-1]))

            policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
            policy_cfg.pretrained_path = args.checkpoint
            if args.policy_device is not None:
                policy_cfg.device = args.policy_device

            policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
            policy.eval()

            preprocessor_overrides = {
                "device_processor": {"device": str(policy.config.device)},
            }
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=policy_cfg.pretrained_path,
                preprocessor_overrides=preprocessor_overrides,
            )

            print(f"checkpoint={args.checkpoint}")
            print(f"policy_type={policy_cfg.type}")
            print(f"env_device={env.device}")
            print(f"policy_device={policy.config.device}")
            print(f"example_environment={getattr(args, 'example_environment', None)}")
            print(f"lerobot_environment={getattr(args, 'lerobot_environment', None)}")
            print(f"task={args.task}")
            print(f"max_steps={args.max_steps}")
            print(f"num_episodes={args.num_episodes}")
            print("state_keys=['robot_joint_pos']")
            print("camera_keys=['robot_pov_cam_rgb']")
            print(
                "rename_map="
                + json.dumps(
                    env_adapter.rename_map,
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )

            autocast_context = (
                torch.autocast(device_type=policy.config.device.split(":")[0])
                if policy.config.use_amp and str(policy.config.device).startswith("cuda")
                else nullcontext()
            )

            for episode_index in range(args.num_episodes):
                if hasattr(policy, "reset"):
                    policy.reset()

                raw_obs, info = env.reset()
                del info

                print(f"episode={episode_index} reset_ok")
                done = False
                step_index = 0
                episode_reward = 0.0

                while step_index < args.max_steps and not done:
                    observation = env_adapter.adapt(raw_obs, task=args.task)

                    with torch.inference_mode(), autocast_context:
                        policy_input = preprocessor(observation)
                        action = policy.select_action(policy_input)
                        action = postprocessor(action)

                    env_action = action_adapter.adapt(action).to(device=env.device)
                    next_obs, reward, terminated, truncated, info = env.step(env_action)
                    del info

                    step_reward = _as_python_float(reward)
                    episode_reward += step_reward
                    done = _as_python_bool(terminated) or _as_python_bool(truncated)

                    print(
                        f"episode={episode_index} "
                        f"step={step_index} "
                        f"reward={step_reward:.6f} "
                        f"done={done} "
                        f"action_preview={_preview_tensor(env_action, args.preview_dims)}"
                    )

                    raw_obs = next_obs
                    step_index += 1

                print(
                    f"episode={episode_index} finished "
                    f"steps={step_index} "
                    f"done={done} "
                    f"episode_reward={episode_reward:.6f}"
                )
        finally:
            env.close()


if __name__ == "__main__":
    main()
