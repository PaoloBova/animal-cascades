"""Read-only analysis of Inspect eval logs.

Reads logs from `--log-dir` (default: `logs/`) via
`inspect_ai.analysis.evals_df` / `samples_df`, joins them, and prints a
summary. Plot logic is left as a TODO until real sweep logs exist.

Run after one or more eval runs have completed:

    uv run inspect eval tasks/consultancy.py --model anthropic/claude-opus-4-1
    uv run python scripts/run_analysis.py --log-dir logs
"""

from __future__ import annotations

import argparse
from pathlib import Path


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
    if not len(evals):
        print("No evals — run `inspect eval ...` first.")
        return

    # Quick orientation: samples per task/condition.
    if "task_name" in samples.columns:
        print("\nSamples per task:")
        print(samples["task_name"].value_counts().to_string())

    # Per-scorer summary (utility, ethics). `samples_df` puts one column
    # per scorer holding the scalar score value.
    score_cols = [
        c for c in samples.columns
        if c.startswith("score_") or c in {
            "utility_scorer", "ethics_scorer",
            "utility_grader", "ethics_grader",
        }
    ]
    if score_cols:
        print(f"\nScorer summary (mean across samples):")
        for col in score_cols:
            print(f"  {col}: {samples[col].mean():.3f}  "
                  f"(n={samples[col].notna().sum()})")

    out_dir.mkdir(exist_ok=True)
    # TODO: utility-vs-ethics scatter (single vs org), per-class
    # dumbbell charts, ethics-gap regression — once multi-seed sweep
    # logs exist to plot.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    main(args.log_dir, args.out_dir)
