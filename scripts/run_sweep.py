"""Run a multi-condition Inspect sweep from a TOML config.

Each `[[conditions]]` block in the TOML file is instantiated by calling
the `@task` function named in `task = "module.path:func_name"` with the
remaining kwargs, then all tasks are dispatched together via
`inspect_ai.eval_set()` which gives durable per-task completion
tracking, retries, and a shared `log_dir` for joint analysis.

Usage:

    uv run python scripts/run_sweep.py configs/main_comparison.toml
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tomllib
from pathlib import Path

# Make the repo root importable when running this script directly so the
# `tasks.*` module paths in the TOML resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_ai import eval_set  # noqa: E402


_RESERVED = {"name", "task"}


def _load_task(spec: str, kwargs: dict):
    """Import `module.path:func_name` and call it with kwargs."""
    module_path, func_name = spec.split(":")
    return getattr(importlib.import_module(module_path), func_name)(**kwargs)


def main(config_path: Path) -> None:
    cfg = tomllib.loads(config_path.read_text())

    if "conditions" not in cfg or not cfg["conditions"]:
        raise ValueError(f"No [[conditions]] in {config_path}")
    if "model" not in cfg:
        raise ValueError(f"Missing top-level `model = ...` in {config_path}")
    if "log_dir" not in cfg:
        raise ValueError(f"Missing top-level `log_dir = ...` in {config_path}")

    tasks = []
    for c in cfg["conditions"]:
        task_kwargs = {k: v for k, v in c.items() if k not in _RESERVED}
        tasks.append(_load_task(c["task"], task_kwargs))

    print(f"Dispatching {len(tasks)} conditions to {cfg['log_dir']}")
    eval_set(tasks, model=cfg["model"], log_dir=cfg["log_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to the sweep TOML")
    args = parser.parse_args()
    main(args.config)
