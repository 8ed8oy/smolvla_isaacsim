# AGENTS.md

This workspace is designed for AI-assisted development.

## Scope

The main goal is to integrate SmolVLA with Isaac Sim / IsaacLab using an embedded direct integration approach first, then grow toward a safety testing platform.

## Project Boundaries

- `lerobot/` is the upstream dependency and should remain mostly untouched.
- `smolvla_isaac_embed/` is the main write area for new work.
- Prefer writing integration code outside `lerobot/`.
- If `lerobot/` must be changed, keep edits minimal and document why.

## Preferred Workflow

1. Inspect environment observations before writing adapters.
2. Verify one-frame policy inference before closing the control loop.
3. Validate action ordering before any long rollout.
4. Add safety wrappers only after the baseline loop works.
5. Keep experiments reproducible via config files and notes.

## Required Context For Any AI Agent

Before suggesting code changes, the agent should identify:

- Which Isaac environment is being used
- Which checkpoint is being loaded
- Which camera keys and state keys are available
- The exact expected action dimension and ordering
- Whether the current task is inspection, dry-run, rollout, or safety testing

## File Ownership Guidance

- `smolvla_isaac_embed/adapters/`: observation/action schema logic
- `smolvla_isaac_embed/wrappers/`: safety and perturbation logic
- `smolvla_isaac_embed/scripts/`: runnable orchestration scripts
- `smolvla_isaac_embed/configs/`: run parameters and experiment configs
- `smolvla_isaac_embed/tests/`: focused correctness tests
- `smolvla_isaac_embed/experiments/`: human notes and run records

## Collaboration Rules

- Keep code paths short and explicit.
- Avoid hidden assumptions in notebook cells or shell history.
- When adding a new dependency, explain why it is needed.
- When changing interfaces, update README or config comments in the same turn.
- Prefer small scripts over one giant control file.

## What AI Should Not Do By Default

- Do not refactor `lerobot/` broadly.
- Do not introduce ROS as the default integration path here.
- Do not optimize for scale before a single-environment baseline works.
- Do not assume camera names or action ordering without inspection.

## Useful Deliverables For This Workspace

- Observation schema documentation
- Action mapping tests
- Minimal end-to-end demo script
- Safety perturbation wrappers
- Failure case writeups with reproduction steps
