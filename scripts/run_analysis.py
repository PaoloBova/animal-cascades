"""Read-only analysis of Inspect eval logs: load logs and produce figures.

Reads `*.eval` logs from `--log-dir` via `inspect_ai.analysis.evals_df`
and `samples_df`, joins them, and saves a canonical figure set to
`--out-dir` using `animal_cascades.plots`.

Run after one or more eval runs have completed:

    uv run inspect eval tasks/consultancy.py --model anthropic/claude-opus-4-1
    uv run python scripts/run_analysis.py --log-dir logs

Column names are derived from the loaded DataFrames so this script
works with either the paper-shape or the per-component grader without
edits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from animal_cascades import plots  # noqa: E402


# Heuristics for identifying which columns hold score values and which
# scorer is which. `samples_df` puts one column per scorer; names follow
# the @scorer function names from `animal_cascades/graders*.py`.
_SCORE_PREFIXES = ("score_",)
_UTILITY_HINTS = ("utility",)
_ETHICS_HINTS = ("ethics",)


def _identify_score_columns(samples: pd.DataFrame) -> dict[str, str]:
    """Return a dict mapping {'utility': col_name, 'ethics': col_name}.

    Looks for columns whose names contain "utility" / "ethics" (case
    insensitive). Falls back to the first two `score_*` columns if no
    name match. Returns an empty dict if nothing usable is found.
    """
    score_cols = [
        c for c in samples.columns
        if c.startswith(_SCORE_PREFIXES)
        or any(h in c.lower() for h in _UTILITY_HINTS + _ETHICS_HINTS)
    ]
    util = next((c for c in score_cols if any(h in c.lower() for h in _UTILITY_HINTS)), None)
    eth = next((c for c in score_cols if any(h in c.lower() for h in _ETHICS_HINTS)), None)
    if util is None or eth is None:
        # Fallback: first two score_* columns
        plain = [c for c in score_cols if c.startswith("score_")]
        if len(plain) >= 2:
            util, eth = util or plain[0], eth or plain[1]
    out = {}
    if util:
        out["utility"] = util
    if eth:
        out["ethics"] = eth
    return out


def _condition_column(samples: pd.DataFrame) -> str | None:
    """Identify a sensible condition-grouping column.

    `task_name` is the natural condition proxy (org vs single-agent task
    name); we return it if present.
    """
    return "task_name" if "task_name" in samples.columns else None


def main(log_dir: Path, out_dir: Path) -> None:
    from inspect_ai.analysis import evals_df, samples_df

    if not log_dir.exists():
        raise FileNotFoundError(f"Log dir not found: {log_dir}")

    log_files = list(log_dir.rglob("*.eval")) + list(log_dir.rglob("*.json"))
    if not log_files:
        print(f"No eval logs in {log_dir} — run `inspect eval ...` first.")
        return

    evals = evals_df(str(log_dir))
    samples = samples_df(str(log_dir))

    print(f"Loaded {len(evals)} eval(s) and {len(samples)} sample row(s) "
          f"from {log_dir}")
    if not len(samples):
        print("No samples in logs.")
        return

    score_cols = _identify_score_columns(samples)
    condition_col = _condition_column(samples)
    print(f"Score columns: {score_cols}")
    print(f"Condition column: {condition_col!r}")

    plots.set_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    # 1. Utility × ethics scatter, coloured by condition.
    if "utility" in score_cols and "ethics" in score_cols:
        fig, _ = plots.scatter(
            samples,
            x=score_cols["utility"], y=score_cols["ethics"],
            hue=condition_col, identity_line=True, alpha=0.7,
        )
        saved.append(plots.save_fig(fig, out_dir / "scatter_utility_vs_ethics.png"))

    # 2. Paired dumbbell: same sample under each condition.
    if (
        "utility" in score_cols
        and condition_col is not None
        and "id" in samples.columns
        and samples[condition_col].nunique() == 2
    ):
        # Aggregate over any seed replicates so each sample has one value
        # per condition.
        agg = (
            samples.groupby(["id", condition_col])[score_cols["utility"]]
            .mean().reset_index()
        )
        fig, _ = plots.paired_dumbbell(
            agg, id_col="id", condition_col=condition_col,
            value_col=score_cols["utility"], sort_by="delta",
        )
        saved.append(plots.save_fig(fig, out_dir / "dumbbell_utility.png"))

    # 3. Distribution box-plot of each score by condition.
    for metric, col in score_cols.items():
        fig, _ = plots.distribution(
            samples, value=col, hue=condition_col, kind="box",
        )
        saved.append(plots.save_fig(fig, out_dir / f"distribution_{metric}.png"))

    # 4. Bar-with-CI summary per scorer × condition.
    if condition_col is not None:
        for metric, col in score_cols.items():
            fig, _ = plots.bar_with_ci(
                samples, group=condition_col, value=col, n_boot=1000,
            )
            saved.append(plots.save_fig(fig, out_dir / f"bar_{metric}_by_condition.png"))

    if saved:
        print(f"\nSaved {len(saved)} figure(s):")
        for p in saved:
            print(f"  {p}")
    else:
        print("\nNo figures saved — score / condition columns not identifiable.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    main(args.log_dir, args.out_dir)
