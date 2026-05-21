"""Generic plotting utilities for the animal-welfare consultancy experiment.

Design principles (apply to every plot function in this module):

1. First positional arg is `df: pandas.DataFrame`. Column references are
   passed by name (keyword args). No column names are hardcoded.
2. Functions accept an optional `ax: matplotlib.axes.Axes | None`. If
   `None`, a new figure is created; otherwise the function draws onto
   the provided axes so callers can compose multi-panel layouts.
3. Functions return `(Figure, Axes)`. For faceted plots the second
   element is a 2-D numpy array of axes (as returned by
   `plt.subplots(nrows, ncols)`).
4. `**kwargs` is forwarded to the underlying matplotlib call (e.g.
   `ax.scatter(..., **kwargs)`) so callers can override marker shape,
   alpha, edge colour, etc. without us enumerating every option.
5. No side effects — never `plt.show()`, never `savefig()`. Saving is
   the caller's responsibility (see `save_fig`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# Default palette consulted only when the caller doesn't override the
# colour kwarg. Override per-call to swap palettes; no required
# dependency on these values.
CONDITION_COLORS: dict[str, str] = {
    "org": "#d62728",
    "single": "#1f77b4",
}

CLASS_MARKERS: dict[str, str] = {"A": "o", "B": "s", "C": "^"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_style(theme: str = "default") -> None:
    """Configure matplotlib rcParams for sensible defaults.

    `theme="default"` applies a light, paper-friendly style. Extend with
    additional themes as needed.
    """
    if theme == "default":
        plt.rcParams.update({
            "figure.figsize": (6.0, 4.0),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "legend.frameon": False,
        })
    else:
        raise ValueError(f"Unknown theme: {theme!r}")


def save_fig(
    fig: Figure,
    path: str | Path,
    dpi: int = 150,
    bbox_inches: str = "tight",
    **savefig_kwargs: Any,
) -> Path:
    """Save `fig` to `path`, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches=bbox_inches, **savefig_kwargs)
    return p


def bootstrap_ci(
    values: Iterable[float],
    ci: float = 0.95,
    n: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`.

    Returns `(low, high)`. Values are converted to a 1-D numpy array;
    NaN entries are dropped before bootstrapping. With fewer than 2
    finite values, returns `(nan, nan)`.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan"), float("nan")
    rng = rng if rng is not None else np.random.default_rng()
    samples = rng.choice(arr, size=(n, arr.size), replace=True).mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def add_gap_column(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    name: str = "gap",
) -> pd.DataFrame:
    """Return a copy of `df` with `name` column = df[x_col] - df[y_col]."""
    out = df.copy()
    out[name] = out[x_col] - out[y_col]
    return out


def to_long(
    df: pd.DataFrame,
    id_cols: Sequence[str],
    value_cols: Sequence[str],
    var_name: str = "metric",
    value_name: str = "value",
) -> pd.DataFrame:
    """Wide-to-long melt. Thin wrapper around `pd.melt` with named args."""
    return df.melt(
        id_vars=list(id_cols),
        value_vars=list(value_cols),
        var_name=var_name,
        value_name=value_name,
    )


def _resolve_color(label: str, default_palette: dict[str, str]) -> str:
    """Look up `label` in `default_palette`; fall back to mpl default cycle."""
    if label in default_palette:
        return default_palette[label]
    # Hash the label into one of the default Tableau colours so unseen
    # labels get stable colours across calls.
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return cycle[hash(label) % len(cycle)]


def _ensure_axes(
    ax: Axes | None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Return `(fig, ax)`: use provided `ax` if given, else create a new one."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    facet: str | None = None,
    ax: Axes | None = None,
    identity_line: bool = False,
    palette: dict[str, str] | None = None,
    legend: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes | np.ndarray]:
    """Scatter plot of `df[x]` vs `df[y]` with optional colour grouping and faceting.

    Args:
        df: Source DataFrame.
        x, y: Column names for the two axes.
        hue: Optional column to colour-group points by.
        facet: Optional column to split into one subplot per unique value.
            Mutually exclusive with `ax`.
        ax: Existing axes to draw onto. Ignored if `facet` is set.
        identity_line: If True, draw a y=x reference line.
        palette: Mapping from `hue` values to colours. Defaults to
            `CONDITION_COLORS`. Unmapped labels fall back to the
            matplotlib default cycle.
        legend: Whether to draw the legend when `hue` is set.
        **kwargs: Forwarded to `Axes.scatter`.

    Returns:
        `(fig, ax)`. When `facet` is set, `ax` is a 2-D ndarray of axes
        with shape `(1, n)`.
    """
    palette = palette if palette is not None else CONDITION_COLORS

    if facet is not None and ax is not None:
        raise ValueError("Cannot pass `ax` with `facet`; facet creates its own figure.")

    if facet is not None:
        facet_values = sorted(df[facet].dropna().unique())
        ncols = max(1, len(facet_values))
        fig, axes = plt.subplots(
            1, ncols, squeeze=False, sharex=True, sharey=True,
            figsize=(4 * ncols, 4),
        )
        for ax_i, val in zip(axes[0], facet_values):
            sub = df[df[facet] == val]
            _scatter_to_ax(sub, x, y, hue, palette, ax_i, identity_line, legend, **kwargs)
            ax_i.set_title(f"{facet}={val}")
        return fig, axes

    fig, ax = _ensure_axes(ax)
    _scatter_to_ax(df, x, y, hue, palette, ax, identity_line, legend, **kwargs)
    return fig, ax


def _scatter_to_ax(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None,
    palette: dict[str, str],
    ax: Axes,
    identity_line: bool,
    legend: bool,
    **kwargs: Any,
) -> None:
    if hue is None:
        ax.scatter(df[x], df[y], **kwargs)
    else:
        # Drop 'color' from kwargs since we set it per-group below.
        kwargs = {k: v for k, v in kwargs.items() if k != "color"}
        for label, group in df.groupby(hue):
            color = _resolve_color(str(label), palette)
            ax.scatter(group[x], group[y], color=color, label=str(label), **kwargs)
        if legend:
            ax.legend(title=hue)
    if identity_line:
        lo = float(min(df[x].min(), df[y].min()))
        hi = float(max(df[x].max(), df[y].max()))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def paired_dumbbell(
    df: pd.DataFrame,
    id_col: str,
    condition_col: str,
    value_col: str,
    conditions: tuple[str, str] | None = None,
    ax: Axes | None = None,
    palette: dict[str, str] | None = None,
    sort_by: Literal["id", "delta", "first", "second"] = "id",
    line_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Horizontal dumbbell: one line per `id_col`, endpoints at the two conditions.

    Args:
        df: Long-format DataFrame with at least `id_col`, `condition_col`,
            `value_col`. Each id should appear once per condition.
        id_col: Column identifying each pair (e.g. scenario id).
        condition_col: Column distinguishing the two conditions.
        value_col: Column with the numeric value to plot.
        conditions: 2-tuple giving the order of conditions (left endpoint,
            right endpoint). Defaults to sorted unique values.
        ax: Optional axes; new one created if `None`.
        palette: Mapping from condition value to colour. Defaults to
            `CONDITION_COLORS`.
        sort_by: How to order pairs along the y-axis. `"id"` preserves
            id order; `"delta"` sorts by (second - first); `"first"` /
            `"second"` sort by the respective condition's value.
        line_kwargs: Passed to the connecting line; defaults to a light
            grey thin line.
        **kwargs: Forwarded to `Axes.scatter` for the endpoints.

    Returns:
        `(fig, ax)`.
    """
    palette = palette if palette is not None else CONDITION_COLORS

    if conditions is None:
        unique = list(df[condition_col].dropna().unique())
        if len(unique) != 2:
            raise ValueError(
                f"paired_dumbbell needs exactly 2 conditions; got {unique}. "
                "Pass `conditions=(a, b)` to disambiguate."
            )
        conditions = tuple(sorted(unique))
    if len(conditions) != 2:
        raise ValueError(
            f"`conditions` must be a 2-tuple; got {conditions!r}."
        )
    cond_a, cond_b = conditions

    wide = df.pivot(index=id_col, columns=condition_col, values=value_col)
    missing_cond = [c for c in (cond_a, cond_b) if c not in wide.columns]
    if missing_cond:
        raise ValueError(f"Conditions not present in data: {missing_cond}")
    wide = wide[[cond_a, cond_b]].dropna()

    if sort_by == "id":
        wide = wide.sort_index()
    elif sort_by == "delta":
        wide = (
            wide.assign(_d=wide[cond_b] - wide[cond_a])
            .sort_values("_d")
            .drop(columns="_d")
        )
    elif sort_by == "first":
        wide = wide.sort_values(cond_a)
    elif sort_by == "second":
        wide = wide.sort_values(cond_b)
    else:
        raise ValueError(f"Unknown sort_by: {sort_by!r}")

    fig, ax = _ensure_axes(ax)
    y_positions = np.arange(len(wide))
    color_a = _resolve_color(str(cond_a), palette)
    color_b = _resolve_color(str(cond_b), palette)
    line_kwargs = {"color": "0.6", "linewidth": 1, **(line_kwargs or {})}

    for y, (val_a, val_b) in zip(y_positions, wide[[cond_a, cond_b]].values):
        ax.plot([val_a, val_b], [y, y], **line_kwargs)

    scatter_kwargs = {k: v for k, v in kwargs.items() if k not in {"color", "label"}}
    ax.scatter(
        wide[cond_a], y_positions,
        color=color_a, label=str(cond_a), **scatter_kwargs,
    )
    ax.scatter(
        wide[cond_b], y_positions,
        color=color_b, label=str(cond_b), **scatter_kwargs,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(wide.index)
    ax.set_xlabel(value_col)
    ax.set_ylabel(id_col)
    ax.legend(title=condition_col)
    return fig, ax


def distribution(
    df: pd.DataFrame,
    value: str,
    hue: str | None = None,
    facet: str | None = None,
    kind: Literal["box", "violin", "strip"] = "box",
    ax: Axes | None = None,
    palette: dict[str, str] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes | np.ndarray]:
    """Distribution of `value`, optionally grouped by `hue` and/or faceted.

    Args:
        df: Long-format DataFrame.
        value: Column whose values are plotted.
        hue: Optional grouping column; each group becomes one
            box/violin/strip on the x-axis.
        facet: Optional column to split into one subplot per unique value.
        kind: One of `"box"`, `"violin"`, or `"strip"`.
        ax: Optional axes; new one created if `None`. Mutually exclusive
            with `facet`.
        palette: Mapping from group label to colour. Used by `violin`
            and `strip`; `box` matplotlib doesn't accept per-box colour
            cleanly.
        **kwargs: Forwarded to the underlying matplotlib call (`boxplot`,
            `violinplot`, or `scatter`).

    Returns:
        `(fig, ax)`; when `facet` is set, `ax` is a 2-D ndarray.
    """
    palette = palette if palette is not None else CONDITION_COLORS

    if facet is not None and ax is not None:
        raise ValueError("Cannot pass `ax` with `facet`; facet creates its own figure.")

    if facet is not None:
        facet_values = sorted(df[facet].dropna().unique())
        ncols = max(1, len(facet_values))
        fig, axes = plt.subplots(
            1, ncols, squeeze=False, sharey=True, figsize=(4 * ncols, 4)
        )
        for ax_i, val in zip(axes[0], facet_values):
            sub = df[df[facet] == val]
            _distribution_to_ax(sub, value, hue, kind, palette, ax_i, **kwargs)
            ax_i.set_title(f"{facet}={val}")
        return fig, axes

    fig, ax = _ensure_axes(ax)
    _distribution_to_ax(df, value, hue, kind, palette, ax, **kwargs)
    return fig, ax


def _distribution_to_ax(
    df: pd.DataFrame,
    value: str,
    hue: str | None,
    kind: str,
    palette: dict[str, str],
    ax: Axes,
    **kwargs: Any,
) -> None:
    if hue is None:
        groups = [df[value].dropna().values]
        labels: list[str] = [value]
    else:
        labels = [str(label) for label in sorted(df[hue].dropna().unique())]
        groups = [df[df[hue].astype(str) == label][value].dropna().values for label in labels]

    positions = np.arange(len(groups))

    if kind == "box":
        ax.boxplot(groups, positions=positions, **kwargs)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
    elif kind == "violin":
        parts = ax.violinplot(groups, positions=positions, showmeans=True, **kwargs)
        bodies = parts.get("bodies", [])
        for body, label in zip(bodies, labels):
            body.set_facecolor(_resolve_color(label, palette))
            body.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
    elif kind == "strip":
        rng = np.random.default_rng(0)
        scatter_kwargs = {k: v for k, v in kwargs.items() if k != "color"}
        for pos, group, label in zip(positions, groups, labels):
            if len(group) == 0:
                continue
            jitter = rng.uniform(-0.15, 0.15, size=len(group))
            color = _resolve_color(label, palette)
            ax.scatter(
                np.full(len(group), float(pos)) + jitter,
                group, color=color, **scatter_kwargs,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
    else:
        raise ValueError(f"Unknown kind: {kind!r}. Use 'box', 'violin', or 'strip'.")

    ax.set_xlabel(hue if hue else "")
    ax.set_ylabel(value)


def bar_with_ci(
    df: pd.DataFrame,
    group: str,
    value: str,
    hue: str | None = None,
    ci: float = 0.95,
    n_boot: int = 1000,
    ax: Axes | None = None,
    palette: dict[str, str] | None = None,
    bar_width: float = 0.8,
    rng_seed: int | None = 0,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Bars of group means with bootstrap CI error bars.

    Args:
        df: Source DataFrame.
        group: Column for the x-axis grouping.
        value: Numeric column whose mean is plotted.
        hue: Optional sub-grouping; produces dodged bars per `group`.
        ci: Confidence-interval level (e.g. 0.95).
        n_boot: Bootstrap resamples per group.
        ax: Optional axes; new one created if `None`.
        palette: Mapping from `hue` value to colour. Defaults to
            `CONDITION_COLORS`.
        bar_width: Total width allotted to each `group` cluster.
        rng_seed: Seed for the bootstrap RNG; `None` for nondeterministic.
        **kwargs: Forwarded to `Axes.bar` (e.g. `edgecolor`, `linewidth`).

    Returns:
        `(fig, ax)`.
    """
    palette = palette if palette is not None else CONDITION_COLORS
    rng = np.random.default_rng(rng_seed)

    fig, ax = _ensure_axes(ax)
    group_labels = sorted(df[group].dropna().unique())
    group_positions = np.arange(len(group_labels))

    def _stats(subset_values: np.ndarray) -> tuple[float, float, float]:
        if subset_values.size == 0:
            return float("nan"), 0.0, 0.0
        mean = float(subset_values.mean())
        lo, hi = bootstrap_ci(subset_values, ci=ci, n=n_boot, rng=rng)
        err_lo = mean - lo if not np.isnan(lo) else 0.0
        err_hi = hi - mean if not np.isnan(hi) else 0.0
        return mean, max(err_lo, 0.0), max(err_hi, 0.0)

    if hue is None:
        means, err_lo, err_hi = [], [], []
        for g in group_labels:
            vals = df[df[group] == g][value].dropna().values
            m, lo, hi = _stats(np.asarray(vals, dtype=float))
            means.append(m); err_lo.append(lo); err_hi.append(hi)
        ax.bar(
            group_positions, means, width=bar_width,
            yerr=[err_lo, err_hi], capsize=4, **kwargs,
        )
        ax.set_xticks(group_positions)
        ax.set_xticklabels([str(g) for g in group_labels])
    else:
        hue_labels = sorted(df[hue].dropna().unique())
        n_hue = len(hue_labels)
        sub_width = bar_width / n_hue
        rest_kwargs = {k: v for k, v in kwargs.items() if k not in {"color", "label"}}
        for i, h in enumerate(hue_labels):
            means, err_lo, err_hi = [], [], []
            for g in group_labels:
                mask = (df[group] == g) & (df[hue] == h)
                vals = df[mask][value].dropna().values
                m, lo, hi = _stats(np.asarray(vals, dtype=float))
                means.append(m); err_lo.append(lo); err_hi.append(hi)
            offset = (i - (n_hue - 1) / 2) * sub_width
            color = _resolve_color(str(h), palette)
            ax.bar(
                group_positions + offset, means, width=sub_width,
                yerr=[err_lo, err_hi], capsize=3, color=color, label=str(h),
                **rest_kwargs,
            )
        ax.set_xticks(group_positions)
        ax.set_xticklabels([str(g) for g in group_labels])
        ax.legend(title=hue)

    ax.set_xlabel(group)
    ax.set_ylabel(value)
    return fig, ax


# Default bloc → colour palette for `org_graph`. Independent of the
# CONDITION_COLORS palette because blocs are distinct from conditions.
BLOC_COLORS: dict[str, str] = {
    "manager": "#d62728",
    "specialist": "#2ca02c",
    "intern": "#1f77b4",
}


def org_graph(
    G: nx.Graph,
    agent_data: Sequence[Any] | None = None,
    ax: Axes | None = None,
    layout: Literal["spring", "circular", "shell"] = "spring",
    bloc_colors: dict[str, str] | None = None,
    show_labels: bool = True,
    node_size: int = 600,
    seed: int = 0,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Visualise the org graph; colour nodes by bloc when `agent_data` is given.

    Args:
        G: A networkx graph (typically the directed SBM email graph).
        agent_data: Optional sequence of agent specs aligned with the
            graph's node ordering. Each item should expose a `.bloc`
            attribute (for colour) and a `.name` or `.description`
            attribute (for the label). Missing attributes fall back to
            sensible defaults.
        ax: Optional axes; new one created if `None`.
        layout: One of `"spring"`, `"circular"`, `"kamada_kawai"`,
            `"shell"`.
        bloc_colors: Mapping from bloc value to colour. Defaults to
            `BLOC_COLORS`; unknown blocs fall back to neutral grey.
        show_labels: Whether to draw node labels.
        node_size: Node size passed through to networkx.
        seed: RNG seed for the spring layout (deterministic plots).
        **kwargs: Forwarded to `networkx.draw_networkx`.

    Returns:
        `(fig, ax)`.
    """
    fig, ax = _ensure_axes(ax)

    layouts = {
        "spring": lambda g: nx.spring_layout(g, seed=seed),
        "circular": nx.circular_layout,
        "shell": nx.shell_layout,
    }
    if layout not in layouts:
        raise ValueError(
            f"Unknown layout: {layout!r}. Use one of {sorted(layouts)}."
        )
    pos = layouts[layout](G)

    bloc_colors = bloc_colors if bloc_colors is not None else BLOC_COLORS

    node_colors: list[str] | str = "lightgray"
    labels: dict[Any, str] | None = None
    blocs_seen: set[str] = set()

    if agent_data is not None:
        if len(agent_data) != G.number_of_nodes():
            raise ValueError(
                f"agent_data has {len(agent_data)} entries but graph has "
                f"{G.number_of_nodes()} nodes."
            )
        nodes = list(G.nodes())
        node_colors = []
        labels = {}
        for node, a in zip(nodes, agent_data):
            bloc = getattr(a, "bloc", None)
            if bloc is not None:
                blocs_seen.add(str(bloc))
            node_colors.append(bloc_colors.get(str(bloc), "0.7") if bloc else "0.7")
            label = (
                getattr(a, "name", None)
                or getattr(a, "description", None)
                or str(node)
            )
            labels[node] = str(label)

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        with_labels=show_labels,
        labels=labels,
        arrows=True,
        **kwargs,
    )
    ax.set_axis_off()

    if blocs_seen:
        from matplotlib.patches import Patch
        handles = [
            Patch(color=bloc_colors.get(b, "0.7"), label=b)
            for b in sorted(blocs_seen)
        ]
        ax.legend(handles=handles, title="bloc", loc="upper right")

    return fig, ax
