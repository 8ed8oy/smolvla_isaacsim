#!/usr/bin/env python3

"""通过连接 Python 3.11 的 Arena 和 Python 3.12 的 lerobot 来运行 SmolVLA rollout。

这个脚本会把 Isaac Sim / IsaacLab-Arena 保持在已验证的 Python 3.11
环境中，同时把 SmolVLA 策略推理委托给独立的 Python 3.12 worker 进程。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from smolvla_isaac_embed.adapters import (  # noqa: E402
    ACTION_ORDER,
    EXPECTED_ACTION_DIM,
    MinimalIsaacActionAdapter,
    MinimalIsaacEnvAdapter,
    SOURCE_CAMERA_KEY,
    TARGET_IMAGE_KEY,
    validate_action_order,
)
from smolvla_isaac_embed.bridge_protocol import (  # noqa: E402
    read_message,
    summarize_array_tree,
    tensors_to_numpy_tree,
    write_message,
)

DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "smolvla_isaac_embed" / "configs" / "gr1_open_microwave_smolvla.toml"
DEFAULT_VIDEO_DIR = WORKSPACE_ROOT / "smolvla_isaac_embed" / "outputs" / "videos" / "run_eval_bridge"
DEFAULT_POLICY_PYTHON = WORKSPACE_ROOT / ".venv-lerobot" / "bin" / "python"
DEFAULT_POLICY_WORKER = WORKSPACE_ROOT / "smolvla_isaac_embed" / "scripts" / "policy_worker.py"


def _print_stage(stage: str, **details: Any) -> None:
    if details:
        payload = json.dumps(details, ensure_ascii=True, sort_keys=True)
        print(f"stage={stage} details={payload}", flush=True)
    else:
        print(f"stage={stage}", flush=True)


def _debug_parse(message: str, **details: Any) -> None:
    if details:
        payload = json.dumps(details, ensure_ascii=True, sort_keys=True)
        print(f"parse_debug={message} details={payload}", flush=True)
    else:
        print(f"parse_debug={message}", flush=True)


def _normalize_cli_path(raw_path: str | Path) -> Path:
    expanded = os.path.expanduser(str(raw_path))
    if os.path.isabs(expanded):
        return Path(os.path.abspath(expanded))
    return Path(os.path.abspath(str(WORKSPACE_ROOT / expanded)))


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
    del parser

    if not example_environment:
        return argv
    if example_environment in argv:
        return argv
    return [*argv, example_environment]


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

    _debug_parse("parse_args_enter", argv=argv)
    config_path, config = _parse_known_config(argv)
    _debug_parse("known_config_loaded", config_path=str(config_path), example_environment=config.get("example_environment"))
    _debug_parse("arena_cli_import_begin")
    _, get_isaaclab_arena_example_environment_cli_parser, _ = _import_arena_cli()
    _debug_parse("arena_cli_import_done")
    _debug_parse("arena_parser_build_begin")
    parser = get_isaaclab_arena_example_environment_cli_parser()
    _debug_parse("arena_parser_build_done", parser_type=type(parser).__name__)
    parser.description = "Run a SmolVLA rollout through a Python 3.11 Arena <-> Python 3.12 policy bridge."
    parser.add_argument("--config", default=str(config_path), help="TOML config path.")
    parser.add_argument("--checkpoint", default=config.get("checkpoint"), help="Policy checkpoint path or repo id.")
    parser.add_argument("--task", default=config.get("task") or "open microwave", help="Task text for the policy.")
    parser.add_argument("--policy_device", default=None, help="Override policy device for the Python 3.12 worker.")
    parser.add_argument("--policy_python", default=str(DEFAULT_POLICY_PYTHON), help="Python executable for the policy worker.")
    parser.add_argument("--policy_worker_script", default=str(DEFAULT_POLICY_WORKER), help="Worker script path.")
    parser.add_argument("--max_steps", type=int, default=5, help="Maximum rollout steps per episode.")
    parser.add_argument("--num_episodes", type=int, default=1, help="How many episodes to run.")
    parser.add_argument("--preview_dims", type=int, default=8, help="How many leading action values to print.")
    parser.add_argument("--video", action="store_true", default=False, help="Enable Gymnasium RecordVideo capture.")
    parser.add_argument("--video_length", type=int, default=int(config.get("video_length") or 100), help="Clip length.")
    parser.add_argument("--video_interval", type=int, default=int(config.get("video_interval") or 200), help="Video trigger interval.")
    parser.add_argument("--video_dir", default=_default_video_dir(config.get("video_dir")), help="RecordVideo output directory.")
    parser.set_defaults(
        embodiment=config.get("embodiment"),
        object=config.get("object"),
        headless=config.get("headless"),
        enable_cameras=config.get("enable_cameras"),
    )
    _debug_parse("custom_args_added")
    argv = _inject_example_environment(argv, parser, config.get("example_environment"))
    _debug_parse("example_environment_injected", argv=argv)
    _debug_parse("argparse_parse_begin")
    args = parser.parse_args(argv)
    _debug_parse("argparse_parse_done")
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
        parser.error("run_eval_bridge.py only supports a single environment. Set --num_envs 1.")
    if not getattr(args, "enable_cameras", False):
        parser.error("run_eval_bridge.py requires --enable_cameras for the current minimal image path.")
    args.lerobot_environment = config.get("lerobot_environment") or config.get("example_environment")
    args.video_dir = str(_normalize_cli_path(args.video_dir))
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


def _record_video_step_trigger(video_interval: int):
    return lambda step_id: step_id % video_interval == 0


def _create_rollout_env(args: argparse.Namespace, env: Any, initial_observation: dict[str, Any]) -> tuple[Any, str | None]:
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


def _get_base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _build_worker_command(args: argparse.Namespace, env: Any, first_observation: dict[str, Any]) -> list[str]:
    env_adapter = MinimalIsaacEnvAdapter(task=args.task)
    adapted = env_adapter.adapt(first_observation)
    state_dim = int(adapted["observation.state"].shape[-1])
    camera_height = int(adapted[TARGET_IMAGE_KEY].shape[-2])
    camera_width = int(adapted[TARGET_IMAGE_KEY].shape[-1])
    action_dim = int(env.action_space.shape[-1])

    command = [
        str(_normalize_cli_path(args.policy_python)),
        "-u",
        str(_normalize_cli_path(args.policy_worker_script)),
        "--checkpoint",
        args.checkpoint,
        "--task",
        args.task,
        "--state_dim",
        str(state_dim),
        "--action_dim",
        str(action_dim),
        "--camera_height",
        str(camera_height),
        "--camera_width",
        str(camera_width),
    ]
    if getattr(args, "lerobot_environment", None):
        command.extend(["--lerobot_environment", str(args.lerobot_environment)])
    if args.policy_device is not None:
        command.extend(["--policy_device", str(args.policy_device)])
    if getattr(args, "embodiment", None):
        command.extend(["--embodiment", str(args.embodiment)])
    if getattr(args, "object", None):
        command.extend(["--object", str(args.object)])
    if getattr(args, "headless", False):
        command.append("--headless")
    return command


def _assert_worker_paths(args: argparse.Namespace) -> None:
    policy_python = _normalize_cli_path(args.policy_python)
    worker_script = _normalize_cli_path(args.policy_worker_script)
    if not policy_python.is_file():
        raise FileNotFoundError(f"Policy Python executable not found: {policy_python}")
    if not worker_script.is_file():
        raise FileNotFoundError(f"Policy worker script not found: {worker_script}")


def _probe_policy_python(args: argparse.Namespace) -> None:
    policy_python = str(_normalize_cli_path(args.policy_python))
    probe_code = (
        "import importlib.util, json, sys; "
        "mods=('torch','huggingface_hub','lerobot'); "
        "status={name: importlib.util.find_spec(name) is not None for name in mods}; "
        "print(json.dumps({'python': sys.executable, 'status': status}, ensure_ascii=True))"
    )
    result = subprocess.run(
        [policy_python, "-c", probe_code],
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to probe the policy Python environment before launching Isaac.\n"
            f"- policy_python: {policy_python}\n"
            f"- returncode: {result.returncode}\n"
            f"- stderr: {stderr or '<empty>'}"
        )
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Policy Python probe returned non-JSON output.\n"
            f"- policy_python: {policy_python}\n"
            f"- stdout: {stdout or '<empty>'}\n"
            f"- stderr: {stderr or '<empty>'}"
        ) from exc

    status = payload.get("status", {})
    missing = [name for name, is_available in status.items() if not is_available]
    _print_stage("policy_python_probe", payload=payload)
    if missing:
        missing_display = ", ".join(missing)
        raise RuntimeError(
            "The selected policy Python environment is missing required packages.\n"
            f"- policy_python: {policy_python}\n"
            f"- missing: {missing_display}\n"
            "Install the missing packages into that Python 3.12 environment, or pass "
            "--policy_python to a different environment that already has them."
        )


@dataclass
class PolicyWorkerClient:
    process: subprocess.Popen[bytes]
    ready_payload: dict[str, Any]

    @classmethod
    def start(cls, command: list[str]) -> "PolicyWorkerClient":
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            cwd=str(WORKSPACE_ROOT),
        )
        if process.stdin is None or process.stdout is None:
            process.kill()
            raise RuntimeError("Failed to create stdio pipes for policy worker.")

        try:
            ready_payload = read_message(process.stdout)
        except EOFError as exc:
            process.wait(timeout=5)
            raise RuntimeError(
                "Policy worker exited before sending a ready message. "
                "Check the worker stderr output above for missing Python 3.12 dependencies "
                f"or checkpoint-loading errors. Command: {command!r}"
            ) from exc
        if ready_payload.get("type") == "error":
            process.wait(timeout=5)
            raise RuntimeError(
                "Policy worker failed during startup:\n"
                f"{ready_payload.get('error_type')}: {ready_payload.get('message')}\n"
                f"{ready_payload.get('traceback')}"
            )
        if ready_payload.get("type") != "ready":
            process.kill()
            raise RuntimeError(f"Expected worker ready message, got {ready_payload!r}")
        return cls(process=process, ready_payload=ready_payload)

    def _ensure_running(self) -> None:
        return_code = self.process.poll()
        if return_code is not None:
            raise RuntimeError(f"Policy worker exited unexpectedly with return code {return_code}.")

    @property
    def stdin(self):
        if self.process.stdin is None:
            raise RuntimeError("Policy worker stdin is unavailable.")
        return self.process.stdin

    @property
    def stdout(self):
        if self.process.stdout is None:
            raise RuntimeError("Policy worker stdout is unavailable.")
        return self.process.stdout

    def reset(self) -> None:
        self._ensure_running()
        write_message(self.stdin, {"type": "reset"})
        try:
            response = read_message(self.stdout)
        except EOFError as exc:
            raise RuntimeError("Policy worker closed the stream while waiting for reset_ack.") from exc
        self._raise_for_error(response)
        if response.get("type") != "reset_ack":
            raise RuntimeError(f"Expected reset_ack from worker, got {response!r}")

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        self._ensure_running()
        wire_observation = tensors_to_numpy_tree(observation)
        write_message(self.stdin, {"type": "infer", "observation": wire_observation})
        try:
            response = read_message(self.stdout)
        except EOFError as exc:
            raise RuntimeError("Policy worker closed the stream while waiting for an action response.") from exc
        self._raise_for_error(response)
        if response.get("type") != "action":
            raise RuntimeError(f"Expected action message from worker, got {response!r}")

        action = response.get("action")
        if not isinstance(action, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray action from worker, got {type(action).__name__}.")
        return action

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                write_message(self.stdin, {"type": "stop"})
                response = read_message(self.stdout)
                self._raise_for_error(response)
            except Exception:  # noqa: BLE001
                self.process.kill()
            else:
                self.process.wait(timeout=5)

    @staticmethod
    def _raise_for_error(response: dict[str, Any]) -> None:
        if response.get("type") != "error":
            return
        raise RuntimeError(
            "Policy worker returned an error:\n"
            f"{response.get('error_type')}: {response.get('message')}\n"
            f"{response.get('traceback')}"
        )


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
    worker: PolicyWorkerClient,
    env_adapter: MinimalIsaacEnvAdapter,
    action_adapter: MinimalIsaacActionAdapter,
    first_observation: dict[str, Any],
) -> None:
    adapted = env_adapter.adapt(first_observation, task=args.task)
    runtime_summary = {
        "checkpoint": args.checkpoint,
        "example_environment": getattr(args, "example_environment", None),
        "lerobot_environment": getattr(args, "lerobot_environment", None),
        "task": args.task,
        "arena_python": sys.executable,
        "policy_python": str(_normalize_cli_path(args.policy_python)),
        "policy_worker_script": str(_normalize_cli_path(args.policy_worker_script)),
        "env_device": str(env.device),
        "worker_policy_device": worker.ready_payload.get("policy_device"),
        "policy_type": worker.ready_payload.get("policy_type"),
        "max_steps": int(args.max_steps),
        "num_episodes": int(args.num_episodes),
        "state_shape": tuple(int(dim) for dim in adapted["observation.state"].shape),
        "image_shape": tuple(int(dim) for dim in adapted[TARGET_IMAGE_KEY].shape),
        "camera_keys": [SOURCE_CAMERA_KEY],
        "rename_map": env_adapter.rename_map,
        "action_dim": int(action_adapter.expected_action_dim),
        "expected_action_dim": int(EXPECTED_ACTION_DIM),
        "video": bool(args.video),
        "video_dir": args.video_dir if args.video else None,
    }
    print("runtime_config=" + json.dumps(runtime_summary, ensure_ascii=True, sort_keys=True))
    print("action_order=" + json.dumps(list(ACTION_ORDER), ensure_ascii=True))
    print(
        "action_order_validation="
        + json.dumps(
            validate_action_order(action_adapter.action_order, expected_action_dim=action_adapter.expected_action_dim),
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    print("worker_ready=" + json.dumps(worker.ready_payload, ensure_ascii=True, sort_keys=True))
    print("observation_wire_summary=" + json.dumps(summarize_array_tree(tensors_to_numpy_tree(adapted)), ensure_ascii=True, sort_keys=True))


def _run_episode(
    *,
    episode_index: int,
    args: argparse.Namespace,
    env: Any,
    raw_obs: dict[str, Any],
    env_adapter: MinimalIsaacEnvAdapter,
    action_adapter: MinimalIsaacActionAdapter,
    worker: PolicyWorkerClient,
) -> EpisodeSummary:
    terminated = False
    truncated = False
    step_index = 0
    episode_reward = 0.0

    worker.reset()
    print(f"episode={episode_index} reset_ok")

    while step_index < args.max_steps and not (terminated or truncated):
        observation = env_adapter.adapt(raw_obs, task=args.task)
        worker_action = worker.infer(observation)
        env_action = action_adapter.adapt(worker_action).to(device=_get_base_env(env).device)
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
    _print_stage("startup", arena_python=sys.executable)
    args = parse_args()
    _print_stage(
        "args_parsed",
        checkpoint=args.checkpoint,
        example_environment=getattr(args, "example_environment", None),
        policy_python=str(_normalize_cli_path(args.policy_python)),
        task=args.task,
    )
    _assert_worker_paths(args)
    _print_stage(
        "worker_paths_ok",
        policy_python=str(_normalize_cli_path(args.policy_python)),
        worker_script=str(_normalize_cli_path(args.policy_worker_script)),
    )
    _probe_policy_python(args)
    get_arena_builder_from_cli, _, SimulationAppContext = _import_arena_cli()
    _print_stage("arena_cli_imported")

    _print_stage("simulation_app_enter")
    with SimulationAppContext(args):
        _print_stage("simulation_app_ready")
        env = get_arena_builder_from_cli(args).make_registered()
        _print_stage("env_created", env_device=str(env.device), action_shape=tuple(env.action_space.shape))
        rollout_env: Any = env
        worker: PolicyWorkerClient | None = None
        try:
            if args.seed is not None:
                env.seed(args.seed)
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                _print_stage("seed_applied", seed=int(args.seed))

            _print_stage("first_reset_begin")
            first_obs, info = env.reset()
            del info
            _print_stage("first_reset_ok", observation_keys=sorted(first_obs.keys()))

            if args.video:
                _extract_video_frame(first_obs)
                _print_stage("video_frame_probe_ok")

            env_adapter = MinimalIsaacEnvAdapter(task=args.task)
            action_adapter = MinimalIsaacActionAdapter(expected_action_dim=int(env.action_space.shape[-1]))
            if action_adapter.expected_action_dim != EXPECTED_ACTION_DIM:
                raise ValueError(
                    f"run_eval_bridge.py expects the GR1 Pink minimal action path with {EXPECTED_ACTION_DIM} dims, "
                    f"but env.action_space reports {action_adapter.expected_action_dim}."
                )

            worker_command = _build_worker_command(args, env, first_obs)
            _print_stage("worker_spawn_begin", command=worker_command)
            worker = PolicyWorkerClient.start(worker_command)
            _print_stage("worker_ready", payload=worker.ready_payload)
            _print_runtime_summary(args, env, worker, env_adapter, action_adapter, first_obs)

            rollout_env, video_output_dir = _create_rollout_env(args, env, first_obs)
            if video_output_dir is not None:
                _print_stage("video_wrapper_ready", video_output_dir=video_output_dir)
                print(f"video_output_dir={video_output_dir}")

            for episode_index in range(args.num_episodes):
                _print_stage("episode_begin", episode_index=int(episode_index))
                raw_obs, info = rollout_env.reset()
                del info
                _print_stage("episode_reset_ok", episode_index=int(episode_index), observation_keys=sorted(raw_obs.keys()))
                _run_episode(
                    episode_index=episode_index,
                    args=args,
                    env=rollout_env,
                    raw_obs=raw_obs,
                    env_adapter=env_adapter,
                    action_adapter=action_adapter,
                    worker=worker,
                )
        finally:
            if worker is not None:
                _print_stage("worker_close_begin")
                worker.close()
                _print_stage("worker_close_done")
            _print_stage("env_close_begin")
            rollout_env.close()
            _print_stage("env_close_done")


if __name__ == "__main__":
    main()
