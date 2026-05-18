#!/usr/bin/env python3

"""运行一个完整的 SmolVLA -> IsaacLab-Arena rollout。

这个脚本保留当前最小接入路径，不扩展 schema 表面：

- 只支持 `policy.robot_joint_pos`
- 只支持 `camera_obs.robot_pov_cam_rgb`
- 动作仍走 36 维 GR1 Pink 最小路径

同时把执行流整理成明确的 rollout 主循环：

- 运行时配置解析
- 环境创建与视频校验
- policy 初始化
- episode 级循环
- episode 内 step 级循环
- 统一打印 step / episode 摘要

用法示例：
- 最小 rollout：
  `python smolvla_isaac_embed/scripts/run_eval.py --config smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml --max_steps 5 --num_episodes 1`
- 打开视频录制：
  `python smolvla_isaac_embed/scripts/run_eval.py --config smolvla_isaac_embed/configs/gr1_open_microwave_smolvla.toml --video --video_length 100 --video_interval 200`
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tomllib
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = WORKSPACE_ROOT / "lerobot" / "src"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from smolvla_isaac_embed.adapters import (  # noqa: E402
    ACTION_ORDER,
    EXPECTED_ACTION_DIM,
    MinimalIsaacActionAdapter,
    MinimalIsaacEnvAdapter,
    SOURCE_CAMERA_KEY,
    SOURCE_STATE_KEY,
    TARGET_IMAGE_KEY,
)

DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "smolvla_isaac_embed" / "configs" / "gr1_open_microwave_smolvla.toml"
DEFAULT_VIDEO_DIR = WORKSPACE_ROOT / "smolvla_isaac_embed" / "outputs" / "videos" / "run_eval"
LEROBOT_PYPROJECT = WORKSPACE_ROOT / "lerobot" / "pyproject.toml"


def _read_lerobot_requires_python() -> str | None:
    if not LEROBOT_PYPROJECT.is_file():
        return None

    with LEROBOT_PYPROJECT.open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    project = pyproject.get("project")
    if not isinstance(project, dict):
        return None

    required = project.get("requires-python")
    return required if isinstance(required, str) else None


def _validate_lerobot_runtime_prereqs() -> None:
    required_python = _read_lerobot_requires_python() or ">=3.12"
    if sys.version_info < (3, 12):
        current_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise RuntimeError(
            "run_eval.py embeds the workspace lerobot runtime, but the current Arena interpreter is too old.\n"
            f"- current python: {current_python}\n"
            f"- lerobot requires: {required_python}\n"
            f"- interpreter: {sys.executable}\n"
            "This is a workspace-level compatibility issue, not just a missing pip package.\n"
            "Recommended next steps:\n"
            "1. Use smolvla_isaac_embed/scripts/arena_smoke_check.py in the Python 3.11 Arena env to keep validating "
            "Isaac observation/action wiring.\n"
            "2. Keep lerobot policy/config inspection in the separate Python 3.12 env (./.venv-lerobot).\n"
            "3. If we want full rollout in one command, we need a bridge strategy instead of importing lerobot "
            "directly inside the Python 3.11 Arena process."
        )

    missing_modules = [name for name in ("draccus", "huggingface_hub") if importlib.util.find_spec(name) is None]
    if missing_modules:
        missing_display = ", ".join(missing_modules)
        raise ModuleNotFoundError(
            "run_eval.py cannot import the embedded lerobot runtime because required packages are missing.\n"
            f"- missing modules: {missing_display}\n"
            f"- interpreter: {sys.executable}\n"
            "Install the missing runtime dependencies into the interpreter that will execute run_eval.py, "
            "or switch to the environment that already contains them."
        )


def _import_arena_cli():
    from isaaclab_arena.examples.example_environments.cli import (
        get_arena_builder_from_cli,
        get_isaaclab_arena_example_environment_cli_parser,
    )
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    return get_arena_builder_from_cli, get_isaaclab_arena_example_environment_cli_parser, SimulationAppContext


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
        "video": env_cfg.get("video"),
        "video_length": env_cfg.get("video_length"),
        "video_interval": env_cfg.get("video_interval"),
        "video_dir": env_cfg.get("video_dir"),
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


def _default_video_dir(configured_dir: str | None) -> str:
    if configured_dir:
        candidate = Path(configured_dir)
        if not candidate.is_absolute():
            candidate = (WORKSPACE_ROOT / candidate).resolve()
        return str(candidate)
    return str(DEFAULT_VIDEO_DIR)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]

    config_path, config = _parse_known_config(argv)
    _validate_lerobot_runtime_prereqs()
    _, get_isaaclab_arena_example_environment_cli_parser, _ = _import_arena_cli()
    parser = get_isaaclab_arena_example_environment_cli_parser()
    parser.description = "Run a complete SmolVLA rollout on IsaacLab-Arena using the embedded adapters."
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
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Enable Gymnasium RecordVideo capture for rollout episodes.",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=int(config.get("video_length") or 100),
        help="Maximum recorded steps per triggered video clip.",
    )
    parser.add_argument(
        "--video_interval",
        type=int,
        default=int(config.get("video_interval") or 200),
        help="Record a new video every N environment steps.",
    )
    parser.add_argument(
        "--video_dir",
        default=_default_video_dir(config.get("video_dir")),
        help="Directory where RecordVideo stores rollout videos.",
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
    if args.preview_dims < 0:
        parser.error("--preview_dims must be >= 0.")
    if args.video_length < 1:
        parser.error("--video_length must be >= 1.")
    if args.video_interval < 1:
        parser.error("--video_interval must be >= 1.")
    if getattr(args, "num_envs", 1) != 1:
        parser.error("run_eval.py only supports a single environment. Set --num_envs 1.")
    args.lerobot_environment = config.get("lerobot_environment") or config.get("example_environment")
    args.video_dir = str(Path(args.video_dir).expanduser().resolve())
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


def _build_env_cfg(args: argparse.Namespace, env: Any, first_observation: dict[str, Any]) -> Any:
    from lerobot.envs.configs import IsaaclabArenaEnv

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
        state_keys=SOURCE_STATE_KEY,
        camera_keys="robot_pov_cam",
        state_dim=state_dim,
        action_dim=action_dim,
        camera_height=camera_height,
        camera_width=camera_width,
        video=bool(args.video),
        video_length=int(args.video_length),
        video_interval=int(args.video_interval),
    )


def _get_base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _validate_runtime_support(args: argparse.Namespace) -> None:
    if not getattr(args, "enable_cameras", False):
        raise ValueError(
            "run_eval.py requires camera observations for the minimal path. "
            "Set --enable_cameras when running this script."
        )
    if args.video and not getattr(args, "enable_cameras", False):
        raise ValueError(
            "Video recording requires --enable_cameras because frames come from "
            f"camera_obs.{SOURCE_CAMERA_KEY}."
        )


def _extract_video_frame(raw_observation: dict[str, Any]) -> np.ndarray:
    camera_obs = raw_observation.get("camera_obs")
    if not isinstance(camera_obs, dict):
        raise KeyError("Observation is missing camera_obs; cannot render video frames.")
    if SOURCE_CAMERA_KEY not in camera_obs:
        raise KeyError(
            f"Observation is missing camera_obs.{SOURCE_CAMERA_KEY}; cannot render video frames for --video."
        )

    frame = camera_obs[SOURCE_CAMERA_KEY]
    if not isinstance(frame, torch.Tensor):
        raise TypeError(
            f"Expected camera_obs.{SOURCE_CAMERA_KEY} to be a torch.Tensor, got {type(frame).__name__}."
        )

    if frame.dim() == 4:
        frame = frame[0]
    if frame.dim() != 3 or frame.shape[-1] != 3:
        raise ValueError(
            f"Expected camera_obs.{SOURCE_CAMERA_KEY} frame with shape (H, W, 3), got {tuple(frame.shape)}."
        )

    frame = frame.detach().to("cpu")
    if frame.dtype != torch.uint8:
        frame = frame.clamp(0.0, 255.0).to(torch.uint8)
    return frame.contiguous().numpy()


class IsaacLabRolloutRecorderEnv(gym.Env):
    """Tiny Gym adapter so Gymnasium RecordVideo can capture IsaacLab rollout frames."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, env: Any, initial_observation: dict[str, Any]):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.render_mode = "rgb_array"
        self._last_observation = initial_observation

    @property
    def unwrapped(self) -> Any:
        return self.env

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._last_observation = observation
        return observation, info

    def step(self, action: Any) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._last_observation = observation
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return _extract_video_frame(self._last_observation)

    def close(self) -> None:
        self.env.close()


@dataclass(slots=True)
class EpisodeSummary:
    episode_index: int
    steps: int
    episode_reward: float
    terminated: bool
    truncated: bool
    hit_max_steps: bool


def _print_runtime_summary(
    args: argparse.Namespace,
    env: Any,
    env_cfg: IsaaclabArenaEnv,
    env_adapter: MinimalIsaacEnvAdapter,
    action_adapter: MinimalIsaacActionAdapter,
) -> None:
    runtime_summary = {
        "checkpoint": args.checkpoint,
        "example_environment": getattr(args, "example_environment", None),
        "lerobot_environment": getattr(args, "lerobot_environment", None),
        "task": args.task,
        "env_device": str(env.device),
        "policy_device": str(getattr(args, "policy_device", None) or "from-checkpoint"),
        "max_steps": int(args.max_steps),
        "num_episodes": int(args.num_episodes),
        "headless": bool(getattr(args, "headless", False)),
        "enable_cameras": bool(getattr(args, "enable_cameras", False)),
        "state_keys": [SOURCE_STATE_KEY],
        "camera_keys": [SOURCE_CAMERA_KEY],
        "rename_map": env_adapter.rename_map,
        "action_dim": int(action_adapter.expected_action_dim),
        "expected_action_dim": int(EXPECTED_ACTION_DIM),
        "video": bool(args.video),
        "video_length": int(args.video_length),
        "video_interval": int(args.video_interval),
        "video_dir": args.video_dir if args.video else None,
        "env_cfg_video": bool(env_cfg.video),
        "env_cfg_video_length": int(env_cfg.video_length),
        "env_cfg_video_interval": int(env_cfg.video_interval),
    }
    print("runtime_config=" + json.dumps(runtime_summary, ensure_ascii=True, sort_keys=True))
    print("action_order=" + json.dumps(list(ACTION_ORDER), ensure_ascii=True))


def _make_policy_runtime(args: argparse.Namespace, env_cfg: Any) -> tuple[Any, Any, Any, Any]:
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

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
    return policy, preprocessor, postprocessor, policy_cfg


def _record_video_step_trigger(video_interval: int):
    return lambda step_id: step_id % video_interval == 0


def _create_rollout_env(
    args: argparse.Namespace,
    env: Any,
    initial_observation: dict[str, Any],
) -> tuple[Any, str | None]:
    if not args.video:
        return env, None

    Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    rollout_env = IsaacLabRolloutRecorderEnv(env=env, initial_observation=initial_observation)
    wrapped_env = RecordVideo(
        rollout_env,
        video_folder=args.video_dir,
        step_trigger=_record_video_step_trigger(int(args.video_interval)),
        video_length=int(args.video_length),
        disable_logger=True,
    )
    return wrapped_env, args.video_dir


def _run_episode(
    *,
    episode_index: int,
    args: argparse.Namespace,
    env: Any,
    raw_obs: dict[str, Any],
    env_adapter: MinimalIsaacEnvAdapter,
    action_adapter: MinimalIsaacActionAdapter,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    autocast_context: Any,
) -> EpisodeSummary:
    terminated = False
    truncated = False
    step_index = 0
    episode_reward = 0.0

    if hasattr(policy, "reset"):
        policy.reset()

    print(f"episode={episode_index} reset_ok")

    while step_index < args.max_steps and not (terminated or truncated):
        observation = env_adapter.adapt(raw_obs, task=args.task)

        with torch.inference_mode(), autocast_context:
            policy_input = preprocessor(observation)
            action = policy.select_action(policy_input)
            action = postprocessor(action)

        env_action = action_adapter.adapt(action).to(device=_get_base_env(env).device)
        next_obs, reward, step_terminated, step_truncated, info = env.step(env_action)
        del info

        step_reward = _as_python_float(reward)
        episode_reward += step_reward
        terminated = _as_python_bool(step_terminated)
        truncated = _as_python_bool(step_truncated)

        print(
            f"episode={episode_index} "
            f"step={step_index} "
            f"reward={step_reward:.6f} "
            f"terminated={terminated} "
            f"truncated={truncated} "
            f"action_preview={_preview_tensor(env_action, args.preview_dims)}"
        )

        raw_obs = next_obs
        step_index += 1

    hit_max_steps = step_index >= args.max_steps and not (terminated or truncated)
    print(
        f"episode={episode_index} finished "
        f"steps={step_index} "
        f"terminated={terminated} "
        f"truncated={truncated} "
        f"hit_max_steps={hit_max_steps} "
        f"episode_reward={episode_reward:.6f}"
    )

    return EpisodeSummary(
        episode_index=episode_index,
        steps=step_index,
        episode_reward=episode_reward,
        terminated=terminated,
        truncated=truncated,
        hit_max_steps=hit_max_steps,
    )


def main() -> None:
    args = parse_args()
    _validate_runtime_support(args)
    get_arena_builder_from_cli, _, SimulationAppContext = _import_arena_cli()

    with SimulationAppContext(args):
        env = get_arena_builder_from_cli(args).make_registered()
        rollout_env: Any = env
        try:
            if args.seed is not None:
                env.seed(args.seed)
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)

            first_obs, info = env.reset()
            del info

            if args.video:
                _extract_video_frame(first_obs)

            env_cfg = _build_env_cfg(args, env, first_obs)
            env_adapter = MinimalIsaacEnvAdapter(task=args.task)
            action_adapter = MinimalIsaacActionAdapter(expected_action_dim=int(env.action_space.shape[-1]))
            if action_adapter.expected_action_dim != EXPECTED_ACTION_DIM:
                raise ValueError(
                    f"run_eval.py expects the GR1 Pink minimal action path with {EXPECTED_ACTION_DIM} dims, "
                    f"but env.action_space reports {action_adapter.expected_action_dim}."
                )

            policy, preprocessor, postprocessor, policy_cfg = _make_policy_runtime(args, env_cfg)
            _print_runtime_summary(args, env, env_cfg, env_adapter, action_adapter)
            print(f"policy_type={policy_cfg.type}")
            print(f"policy_device={policy.config.device}")

            rollout_env, video_output_dir = _create_rollout_env(args, env, first_obs)
            if video_output_dir is not None:
                print(f"video_output_dir={video_output_dir}")

            autocast_context = (
                torch.autocast(device_type=policy.config.device.split(":")[0])
                if policy.config.use_amp and str(policy.config.device).startswith("cuda")
                else nullcontext()
            )

            for episode_index in range(args.num_episodes):
                raw_obs, info = rollout_env.reset()
                del info
                _run_episode(
                    episode_index=episode_index,
                    args=args,
                    env=rollout_env,
                    raw_obs=raw_obs,
                    env_adapter=env_adapter,
                    action_adapter=action_adapter,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    autocast_context=autocast_context,
                )
        finally:
            rollout_env.close()


if __name__ == "__main__":
    main()
