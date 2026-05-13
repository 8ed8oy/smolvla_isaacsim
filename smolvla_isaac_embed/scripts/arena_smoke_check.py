#!/usr/bin/env python3

"""Run a minimal IsaacLab-Arena smoke test without depending on lerobot installation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping

import numpy as np
import torch

from isaaclab_arena.examples.example_environments.cli import (
    get_arena_builder_from_cli,
    get_isaaclab_arena_example_environment_cli_parser,
)
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def _format_shape(value: object) -> str:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None:
        return type(value).__name__
    return f"shape={tuple(shape)} dtype={dtype}"


def _print_mapping(prefix: str, mapping: Mapping[str, object]) -> None:
    for key, value in mapping.items():
        print(f"{prefix}{key}: {_format_shape(value)}")


def main() -> None:
    parser = get_isaaclab_arena_example_environment_cli_parser()
    parser.add_argument("--num_steps", type=int, default=2, help="How many zero-action env steps to run.")
    args = parser.parse_args()

    with SimulationAppContext(args):
        env = get_arena_builder_from_cli(args).make_registered()
        try:
            if args.seed is not None:
                env.seed(args.seed)
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)

            obs, info = env.reset()
            del info

            print("reset_ok")
            print(f"device={env.device}")
            print(f"action_space_shape={env.action_space.shape}")

            if isinstance(obs, Mapping):
                print(f"observation_top_level_keys={sorted(obs.keys())}")
                if "policy" in obs and isinstance(obs["policy"], Mapping):
                    print("policy_keys:")
                    _print_mapping("  ", obs["policy"])
                if "camera_obs" in obs and isinstance(obs["camera_obs"], Mapping):
                    print("camera_obs_keys:")
                    _print_mapping("  ", obs["camera_obs"])

            action_shape = tuple(env.action_space.shape)
            zero_actions = torch.zeros(action_shape, device=env.device)

            for step_idx in range(args.num_steps):
                with torch.inference_mode():
                    obs, reward, terminated, truncated, info = env.step(zero_actions)
                    del info
                print(
                    f"step={step_idx} reward={_format_shape(reward)} "
                    f"terminated={_format_shape(terminated)} truncated={_format_shape(truncated)}"
                )

            print("smoke_test_ok")
        finally:
            env.close()


if __name__ == "__main__":
    main()
