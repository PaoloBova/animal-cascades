"""Single-agent baseline task.

One Claude react loop given the same role list as the AI Organization
(role specification controlled for as a confounder per the paper). Runs
the same dataset and graders as `tasks/consultancy.py` for direct
comparison.

Run from the repo root, e.g.:

    uv run inspect eval tasks/single_agent.py --model anthropic/claude-opus-4-1 \\
        -T seed=42 -T grader=paper
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset

from animal_cascades.org import (
    ROLE_COUNT_PRESETS,
    agents_from_counts,
    build_org_data,
)
from animal_cascades.single_agent import consultancy_partner


def _scorers(grader: str, grader_model: str | None):
    if grader == "paper":
        from animal_cascades.graders import ethics_scorer, utility_scorer
        return [utility_scorer(model=grader_model), ethics_scorer(model=grader_model)]
    elif grader == "alt":
        from animal_cascades.graders_alt import ethics_grader, utility_grader
        return [utility_grader(model=grader_model), ethics_grader(model=grader_model)]
    else:
        raise ValueError(f"Unknown grader: {grader!r}. Choose 'paper' or 'alt'.")


@task
def animal_cascade_baseline(
    seed: int = 42,
    grader: Literal["paper", "alt"] = "paper",
    grader_model: str | None = None,
    role_counts_preset: str = "default",
) -> Task:
    """Single-agent baseline. T and block_probs_preset are unused but accepted
    in the kwarg surface for parity with the org task (so sweep configs can
    share kwargs across conditions)."""
    project_root = Path(__file__).parent.parent
    dataset_path = str(project_root / "datasets" / "animal_cascade.jsonl")

    agent_data = agents_from_counts(ROLE_COUNT_PRESETS[role_counts_preset])
    _, org_data = build_org_data(agent_data=agent_data, seed=seed)

    return Task(
        dataset=json_dataset(dataset_path),
        solver=consultancy_partner(org_data=org_data),
        scorer=_scorers(grader, grader_model),
    )
