# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `src/openpi/`, split by concern: `models/`, `models_pytorch/`, `policies/`, `training/`, `serving/`, `shared/`, and `utils/`. CLI and operational entry points live in `scripts/` (for example, `scripts/train.py` and `scripts/serve_policy.py`). End-to-end usage examples and robot-specific workflows live under `examples/`. The workspace package `packages/openpi-client/` contains the client runtime used for remote inference. Treat `third_party/` as vendored code, not a place for routine edits.

## Build, Test, and Development Commands
Use `uv` for environment management.

- `GIT_LFS_SKIP_SMUDGE=1 uv sync` installs project and dev dependencies.
- `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .` installs the root package in editable mode.
- `uv run pytest` runs the default test suite across `src`, `scripts`, and `packages`.
- `uv run pytest -m "not manual"` skips tests marked `manual`.
- `uv run ruff check .` runs lint checks.
- `uv run ruff format .` applies formatting.
- `pre-commit run --all-files` runs the same repo hooks used in CI, including `uv-lock` and Ruff.

## Coding Style & Naming Conventions
Target Python is 3.11 with a 120-character line limit. Use 4-space indentation and keep imports Ruff/isort-compatible; this repo enforces single-line imports in many cases. Follow existing naming patterns: `snake_case` for modules/functions, `PascalCase` for classes, and `*_test.py` for tests. Avoid editing generated or vendor-style code in `src/openpi/models_pytorch/transformers_replace/` unless the change is intentional.

## Testing Guidelines
Pytest is the test runner. Keep unit tests close to the code they cover, such as `src/openpi/models/model_test.py` or `packages/openpi-client/src/openpi_client/image_tools_test.py`. Use the `manual` marker only for tests that cannot run in normal automation. For training or inference changes, add at least one focused regression test and run the nearest relevant test module before opening a PR.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects; keep them specific, and include a PR number when squashing through GitHub if applicable. PRs should have a clear title, a concise description of behavior changes, linked issues or discussion threads when relevant, and screenshots/log snippets for UI, robot, or training-output changes. Before submitting, run `ruff check`, `ruff format`, `pre-commit`, and the affected `pytest` targets.

## Repository Hygiene
Do not commit local runtime artifacts such as `checkpoints/`, `wandb/`, or ad hoc debug files unless the change explicitly requires curated fixtures or documentation assets.
