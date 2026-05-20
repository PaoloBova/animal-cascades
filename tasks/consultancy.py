"""AI Organization consultancy task.

Run from the repo root, e.g.:

    uv run inspect eval tasks/consultancy.py --model anthropic/claude-opus-4-1 \\
        -T T=5 -T seed=42 -T grader=paper

See `configs/main_comparison.toml` + `scripts/run_sweep.py` for multi-condition
sweeps via `inspect_ai.eval_set()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset

from animal_cascades.org import (
    BLOCK_PROBS_PRESETS,
    ROLE_COUNT_PRESETS,
    agents_from_counts,
    build_org_data,
    org_consultancy,
)


def _scorers(grader: str, grader_model: str | None):
    """Switch between the paper-style and per-component graders."""
    if grader == "paper":
        from animal_cascades.graders import ethics_scorer, utility_scorer
        return [utility_scorer(model=grader_model), ethics_scorer(model=grader_model)]
    elif grader == "alt":
        from animal_cascades.graders_alt import ethics_grader, utility_grader
        return [utility_grader(model=grader_model), ethics_grader(model=grader_model)]
    else:
        raise ValueError(f"Unknown grader: {grader!r}. Choose 'paper' or 'alt'.")


@task
def animal_cascade_scenarios(
    T: int = 5,
    seed: int = 42,
    grader: Literal["paper", "alt"] = "paper",
    grader_model: str | None = None,
    role_counts_preset: str = "default",
    block_probs_preset: str = "default",
) -> Task:
    """AI Organization consultancy: multi-agent over SBM-structured email graph."""
    project_root = Path(__file__).parent.parent
    dataset_path = str(project_root / "datasets" / "animal_cascade.jsonl")

    agent_data = agents_from_counts(ROLE_COUNT_PRESETS[role_counts_preset])
    block_probs = BLOCK_PROBS_PRESETS[block_probs_preset]
    G, org_data = build_org_data(
        agent_data=agent_data, seed=seed, block_probs=block_probs
    )

    return Task(
        dataset=json_dataset(dataset_path),
        solver=org_consultancy(G=G, org_data=org_data, T=T),
        scorer=_scorers(grader, grader_model),
    )
