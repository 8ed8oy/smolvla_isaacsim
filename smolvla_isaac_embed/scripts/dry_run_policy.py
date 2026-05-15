#!/usr/bin/env python3

"""运行一次单帧的 IsaacLab-Arena -> policy dry run。

这个脚本只做一次环境 reset，把第一帧观测转换为 LeRobot 策略输入，
执行一次策略推理，并打印得到的 action 形状以及前几个数值。

它的范围刻意很窄：
- 不做 rollout 循环
- 不把动作执行回环境
- 不加安全包装器

用法：
- 当你想在完整 rollout 之前确认单帧策略推理是否正常时，运行这个脚本。
- 典型示例：``python smolvla_isaac_embed/scripts/dry_run_policy.py --config smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml``。
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = WORKSPACE_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from isaaclab_arena.examples.example_environments.cli import (  # noqa: E402
    get_arena_builder_from_cli,
    get_isaaclab_arena_example_environment_cli_parser,
)
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext  # noqa: E402
from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.envs.configs import IsaaclabArenaEnv  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402

DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "smolvla_isaac_embed" / "configs" / "gr1_open_microwave_smolvla.toml"


def _load_runtime_config(config_path: Path) -> dict:
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    policy_cfg = raw_config.get("policy", {})
    env_cfg = raw_config.get("env", {})
    obs_cfg = raw_config.get("observations", {})
    rename_map = obs_cfg.get("rename_map", {})

    if not isinstance(policy_cfg, dict) or not isinstance(env_cfg, dict) or not isinstance(obs_cfg, dict):
        raise ValueError("Config sections [policy], [env], and [observations] must be TOML tables.")

    return {
        "checkpoint": policy_cfg.get("checkpoint"),
        "task": policy_cfg.get("task"),
        "example_environment": env_cfg.get("example_environment"),
        "embodiment": env_cfg.get("embodiment"),
        "object": env_cfg.get("object"),
        "headless": env_cfg.get("headless"),
        "enable_cameras": env_cfg.get("enable_cameras"),
        "state_keys": obs_cfg.get("state_keys"),
        "camera_keys": obs_cfg.get("camera_keys"),
        "rename_map": rename_map,
    }


def _coerce_str_list(raw_value: object, field_name: str) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if not isinstance(raw_value, list) or any(not isinstance(item, str) for item in raw_value):
        raise ValueError(f"Config field {field_name} must be an array of strings.")
    return tuple(item.strip() for item in raw_value if item.strip())


def _validate_rename_map(raw_value: object) -> dict[str, str]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError("Config field observations.rename_map must be a table of string-to-string mappings.")
    rename_map: dict[str, str] = {}
    for key, value in raw_value.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Config field observations.rename_map must only contain string keys and values.")
        rename_map[key] = value
    return rename_map


def _parser_default_config_path() -> str:
    return str(DEFAULT_CONFIG_PATH.relative_to(WORKSPACE_ROOT))


def _parse_known_config(argv: list[str]) -> tuple[Path, dict]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=_parser_default_config_path())
    parsed, _ = pre_parser.parse_known_args(argv)
    config_path = Path(parsed.config)
    if not config_path.is_absolute():
        config_path = (WORKSPACE_ROOT / config_path).resolve()
    config = _load_runtime_config(config_path)
    return config_path, config


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
    parser.description = "Run a single-frame policy dry run on an IsaacLab-Arena environment."
    parser.add_argument(
        "--config",
        default=str(config_path),
        help="TOML config path for checkpoint, environment, and observation wiring.",
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
        "--state_keys",
        default=",".join(_coerce_str_list(config.get("state_keys"), "observations.state_keys")) or "robot_joint_pos",
        help="Comma-separated state keys to concatenate into observation.state.",
    )
    parser.add_argument(
        "--camera_keys",
        default=",".join(_coerce_str_list(config.get("camera_keys"), "observations.camera_keys"))
        or "robot_pov_cam_rgb",
        help="Comma-separated camera keys to expose as observation.images.*.",
    )
    parser.add_argument(
        "--rename_map",
        default=json.dumps(
            _validate_rename_map(config.get("rename_map"))
            or {"observation.images.robot_pov_cam_rgb": "observation.images.robot_pov_cam"},
            ensure_ascii=True,
            sort_keys=True,
        ),
        help="JSON object for observation key renaming before policy preprocessing.",
    )
    parser.add_argument(
        "--policy_device",
        default=None,
        help="Override policy device, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--preview_dims",
        type=int,
        default=8,
        help="How many leading action values to print.",
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
    return args


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_rename_map(raw_value: str) -> dict[str, str]:
    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError("--rename_map must parse to a JSON object.")
    rename_map: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("--rename_map keys and values must be strings.")
        rename_map[key] = value
    return rename_map


def _flatten_dim(tensor: torch.Tensor) -> int:
    if tensor.dim() <= 1:
        return int(tensor.numel())
    return int(np.prod(tensor.shape[1:], dtype=np.int64))


def _infer_state_dim(policy_obs: dict[str, torch.Tensor], state_keys: tuple[str, ...]) -> int:
    total_dim = 0
    for key in state_keys:
        if key not in policy_obs:
            raise KeyError(f"Missing state key in env observation['policy']: {key}")
        total_dim += _flatten_dim(policy_obs[key])
    return total_dim


def _infer_camera_shape(camera_obs: dict[str, torch.Tensor], camera_keys: tuple[str, ...]) -> tuple[int, int]:
    for key in camera_keys:
        if key not in camera_obs:
            raise KeyError(f"Missing camera key in env observation['camera_obs']: {key}")
        image = camera_obs[key]
        if image.dim() != 4:
            raise ValueError(f"Expected camera tensor with shape (B, H, W, C), got {tuple(image.shape)} for {key}")
        return int(image.shape[1]), int(image.shape[2])
    raise ValueError("At least one camera key is required to infer image shape.")


def _preview_tensor(tensor: torch.Tensor, n: int) -> list[float]:
    flat = tensor.detach().to("cpu").reshape(-1)
    return [float(value) for value in flat[: max(n, 0)].tolist()]


def main() -> None:
    args = parse_args()
    state_keys = _split_csv(args.state_keys)
    camera_keys = _split_csv(args.camera_keys)
    rename_map = _parse_rename_map(args.rename_map)

    with SimulationAppContext(args):
        env = get_arena_builder_from_cli(args).make_registered()
        try:
            if args.seed is not None:
                env.seed(args.seed)
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)

            raw_obs, info = env.reset()
            del info

            if not isinstance(raw_obs, dict):
                raise TypeError(f"Expected env.reset() to return a mapping, got {type(raw_obs).__name__}")
            if "policy" not in raw_obs or "camera_obs" not in raw_obs:
                raise KeyError(
                    "Expected IsaacLab-Arena observation to contain both 'policy' and 'camera_obs' top-level keys."
                )

            state_dim = _infer_state_dim(raw_obs["policy"], state_keys)
            camera_height, camera_width = _infer_camera_shape(raw_obs["camera_obs"], camera_keys)
            action_dim = int(env.action_space.shape[-1])

            env_cfg = IsaaclabArenaEnv(
                embodiment=getattr(args, "embodiment", None),
                object=getattr(args, "object", None),
                environment=getattr(args, "example_environment", None),
                task=args.task,
                enable_cameras=bool(camera_keys),
                headless=bool(getattr(args, "headless", False)),
                device=str(env.device),
                state_keys=",".join(state_keys),
                camera_keys=",".join(camera_keys) if camera_keys else None,
                state_dim=state_dim,
                action_dim=action_dim,
                camera_height=camera_height,
                camera_width=camera_width,
            )

            policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
            policy_cfg.pretrained_path = args.checkpoint
            if args.policy_device is not None:
                policy_cfg.device = args.policy_device

            policy = make_policy(
                cfg=policy_cfg,
                env_cfg=env_cfg,
                rename_map=rename_map,
            )
            policy.eval()

            preprocessor_overrides = {
                "device_processor": {"device": str(policy.config.device)},
                "rename_observations_processor": {"rename_map": rename_map},
            }
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=policy_cfg.pretrained_path,
                preprocessor_overrides=preprocessor_overrides,
            )
            env_preprocessor, _ = env_cfg.get_env_processors()

            observation = preprocess_observation(raw_obs)
            observation = env_preprocessor(observation)
            observation["task"] = args.task

            autocast_context = (
                torch.autocast(device_type=policy.config.device.split(":")[0])
                if policy.config.use_amp and str(policy.config.device).startswith("cuda")
                else nullcontext()
            )
            with torch.inference_mode(), autocast_context:
                policy_input = preprocessor(observation)
                action = policy.select_action(policy_input)
                action = postprocessor(action)

            print(f"checkpoint={args.checkpoint}")
            print(f"policy_type={policy_cfg.type}")
            print(f"env_device={env.device}")
            print(f"policy_device={policy.config.device}")
            print(f"task={args.task}")
            print(f"state_keys={list(state_keys)}")
            print(f"camera_keys={list(camera_keys)}")
            print(f"rename_map={json.dumps(rename_map, ensure_ascii=True, sort_keys=True)}")
            print(f"observation.state.shape={tuple(observation['observation.state'].shape)}")
            for camera_key in camera_keys:
                obs_key = f"observation.images.{camera_key}"
                if obs_key in observation:
                    print(f"{obs_key}.shape={tuple(observation[obs_key].shape)}")
            print(f"action.shape={tuple(action.shape)}")
            print(f"action.preview={_preview_tensor(action, args.preview_dims)}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
