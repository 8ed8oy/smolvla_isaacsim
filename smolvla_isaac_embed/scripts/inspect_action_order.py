#!/usr/bin/env python3

"""Print the current 36-dim GR1 Pink action order in a review-friendly format."""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from smolvla_isaac_embed.adapters import ACTION_ORDER, describe_action_order, validate_action_order  # noqa: E402


def main() -> None:
    validation = validate_action_order()
    print("action_order_validation=" + json.dumps(validation, ensure_ascii=True, sort_keys=True))

    for entry in describe_action_order(ACTION_ORDER):
        print(
            f"index={entry['index']:02d} "
            f"group={entry['group']} "
            f"name={entry['name']}"
        )


if __name__ == "__main__":
    main()
