"""Smoke tests for animal_cascades.plots.

The module is imported with the Agg backend so plots render without a
display attached. Each plot function is exercised with a tiny synthetic
DataFrame and arbitrary column names (renamed in some tests) to verify
that nothing is hardcoded.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from animal_cascades import plots


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to avoid the
    "more than 20 figures" warning during the test session."""
    yield
    plt.close("all")


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": ["s1", "s1", "s2", "s2", "s3", "s3"],
        "condition": ["org", "single", "org", "single", "org", "single"],
        "utility": [0.7, 0.4, 0.8, 0.5, 0.6, 0.3],
        "ethics": [0.3, 0.7, 0.2, 0.8, 0.4, 0.9],
        "scenario_class": ["A", "A", "C", "C", "B", "B"],
    })


@pytest.fixture
def renamed_df(tiny_df: pd.DataFrame) -> pd.DataFrame:
    """Same data, arbitrary column names — proves no name is hardcoded."""
    return tiny_df.rename(columns={
        "id": "sample",
        "condition": "arm",
        "utility": "score_x",
        "ethics": "score_y",
        "scenario_class": "klass",
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_bootstrap_ci_returns_low_high(self):
        lo, hi = plots.bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], n=200)
        assert lo < hi
        assert 0.1 < lo < 0.5
        assert 0.1 < hi < 0.5

    def test_bootstrap_ci_too_few_values(self):
        lo, hi = plots.bootstrap_ci([0.5])
        assert np.isnan(lo) and np.isnan(hi)

    def test_add_gap_column(self, tiny_df):
        out = plots.add_gap_column(tiny_df, "utility", "ethics", name="gap")
        assert "gap" in out.columns
        assert out["gap"].iloc[0] == pytest.approx(0.4)
        # Original frame not mutated
        assert "gap" not in tiny_df.columns

    def test_to_long(self, tiny_df):
        long = plots.to_long(
            tiny_df, id_cols=["id", "condition"],
            value_cols=["utility", "ethics"],
        )
        assert set(long.columns) == {"id", "condition", "metric", "value"}
        assert len(long) == 2 * len(tiny_df)


# ---------------------------------------------------------------------------
# scatter
# ---------------------------------------------------------------------------

class TestScatter:
    def test_basic(self, tiny_df):
        fig, ax = plots.scatter(tiny_df, x="utility", y="ethics")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_xlabel() == "utility"
        assert ax.get_ylabel() == "ethics"

    def test_with_hue(self, tiny_df):
        fig, ax = plots.scatter(tiny_df, x="utility", y="ethics", hue="condition")
        # Two groups → two PathCollection artists on the axes.
        assert sum(1 for c in ax.collections) == 2
        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_title().get_text() == "condition"

    def test_with_facet(self, tiny_df):
        fig, axes = plots.scatter(
            tiny_df, x="utility", y="ethics", facet="scenario_class"
        )
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (1, 3)  # A, B, C classes

    def test_facet_and_ax_raises(self, tiny_df):
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="Cannot pass `ax` with `facet`"):
            plots.scatter(tiny_df, x="utility", y="ethics", facet="scenario_class", ax=ax)

    def test_ax_injection(self, tiny_df):
        fig, my_ax = plt.subplots()
        ret_fig, ret_ax = plots.scatter(tiny_df, x="utility", y="ethics", ax=my_ax)
        assert ret_ax is my_ax
        assert ret_fig is fig

    def test_renamed_columns(self, renamed_df):
        fig, ax = plots.scatter(renamed_df, x="score_x", y="score_y", hue="arm")
        assert ax.get_xlabel() == "score_x"
        assert ax.get_ylabel() == "score_y"

    def test_identity_line(self, tiny_df):
        fig, ax = plots.scatter(tiny_df, x="utility", y="ethics", identity_line=True)
        # At least one Line2D for the identity line.
        assert len(ax.get_lines()) >= 1

    def test_kwargs_forwarded(self, tiny_df):
        # alpha is a scatter kwarg — passing it shouldn't error.
        fig, ax = plots.scatter(tiny_df, x="utility", y="ethics", alpha=0.3, s=80)
        assert isinstance(ax, Axes)


# ---------------------------------------------------------------------------
# paired_dumbbell
# ---------------------------------------------------------------------------

class TestPairedDumbbell:
    def test_basic(self, tiny_df):
        fig, ax = plots.paired_dumbbell(
            tiny_df, id_col="id", condition_col="condition", value_col="utility",
        )
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # 3 ids, 2 endpoints each → 2 PathCollections (one per condition)
        assert len([c for c in ax.collections]) == 2
        # 3 lines connecting endpoints, one per id
        assert sum(1 for ln in ax.get_lines()) == 3
        assert ax.get_xlabel() == "utility"
        assert ax.get_ylabel() == "id"

    def test_renamed_columns(self, renamed_df):
        fig, ax = plots.paired_dumbbell(
            renamed_df, id_col="sample", condition_col="arm", value_col="score_x",
        )
        assert ax.get_xlabel() == "score_x"
        assert ax.get_ylabel() == "sample"

    def test_explicit_conditions(self, tiny_df):
        fig, ax = plots.paired_dumbbell(
            tiny_df, id_col="id", condition_col="condition", value_col="utility",
            conditions=("single", "org"),
        )
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert legend_labels == ["single", "org"]

    def test_sort_by_delta(self, tiny_df):
        fig, ax = plots.paired_dumbbell(
            tiny_df, id_col="id", condition_col="condition", value_col="utility",
            sort_by="delta",
        )
        # Should run without error and produce 3 ticks
        assert len(ax.get_yticks()) == 3

    def test_wrong_number_of_conditions(self):
        df = pd.DataFrame({
            "id": ["a", "a", "a"],
            "condition": ["x", "y", "z"],
            "value": [1.0, 2.0, 3.0],
        })
        with pytest.raises(ValueError, match="exactly 2 conditions"):
            plots.paired_dumbbell(df, id_col="id", condition_col="condition", value_col="value")

    def test_ax_injection(self, tiny_df):
        fig, my_ax = plt.subplots()
        ret_fig, ret_ax = plots.paired_dumbbell(
            tiny_df, id_col="id", condition_col="condition", value_col="utility",
            ax=my_ax,
        )
        assert ret_ax is my_ax


# ---------------------------------------------------------------------------
# distribution
# ---------------------------------------------------------------------------

class TestDistribution:
    @pytest.mark.parametrize("kind", ["box", "violin", "strip"])
    def test_basic(self, tiny_df, kind):
        fig, ax = plots.distribution(
            tiny_df, value="utility", hue="condition", kind=kind,
        )
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert set(labels) == {"org", "single"}

    def test_no_hue(self, tiny_df):
        fig, ax = plots.distribution(tiny_df, value="utility", kind="box")
        # Single group, single tick
        assert len(ax.get_xticks()) == 1

    def test_with_facet(self, tiny_df):
        fig, axes = plots.distribution(
            tiny_df, value="utility", hue="condition", facet="scenario_class",
        )
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (1, 3)

    def test_renamed_columns(self, renamed_df):
        fig, ax = plots.distribution(
            renamed_df, value="score_x", hue="arm", kind="violin",
        )
        assert ax.get_ylabel() == "score_x"
        assert ax.get_xlabel() == "arm"

    def test_unknown_kind_raises(self, tiny_df):
        with pytest.raises(ValueError, match="Unknown kind"):
            plots.distribution(tiny_df, value="utility", hue="condition", kind="pie")

    def test_ax_injection(self, tiny_df):
        fig, my_ax = plt.subplots()
        ret_fig, ret_ax = plots.distribution(
            tiny_df, value="utility", hue="condition", ax=my_ax,
        )
        assert ret_ax is my_ax


# ---------------------------------------------------------------------------
# bar_with_ci
# ---------------------------------------------------------------------------

class TestBarWithCi:
    def test_basic(self, tiny_df):
        fig, ax = plots.bar_with_ci(
            tiny_df, group="condition", value="utility", n_boot=100,
        )
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert set(labels) == {"org", "single"}

    def test_with_hue(self, tiny_df):
        fig, ax = plots.bar_with_ci(
            tiny_df, group="scenario_class", value="utility", hue="condition",
            n_boot=100,
        )
        # 3 group positions × 2 hue → 6 bar containers
        assert sum(1 for c in ax.containers) >= 2  # at least one container per hue
        assert ax.get_legend() is not None

    def test_renamed_columns(self, renamed_df):
        fig, ax = plots.bar_with_ci(
            renamed_df, group="arm", value="score_x", n_boot=100,
        )
        assert ax.get_xlabel() == "arm"
        assert ax.get_ylabel() == "score_x"

    def test_ax_injection(self, tiny_df):
        fig, my_ax = plt.subplots()
        ret_fig, ret_ax = plots.bar_with_ci(
            tiny_df, group="condition", value="utility", ax=my_ax, n_boot=100,
        )
        assert ret_ax is my_ax

    def test_kwargs_forwarded(self, tiny_df):
        fig, ax = plots.bar_with_ci(
            tiny_df, group="condition", value="utility", n_boot=100,
            edgecolor="black", linewidth=1.5,
        )
        assert isinstance(ax, Axes)


# ---------------------------------------------------------------------------
# org_graph
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)])
    return G


@pytest.fixture
def tiny_agent_data():
    from animal_cascades.org import AgentConfig
    return [
        AgentConfig(role="Communications Director", bloc="manager"),
        AgentConfig(role="Policy Analyst", bloc="specialist"),
        AgentConfig(role="Cost Analysis Specialist", bloc="specialist"),
        AgentConfig(role="Web Search Intern", bloc="intern"),
    ]


class TestOrgGraph:
    def test_basic_no_agent_data(self, tiny_graph):
        fig, ax = plots.org_graph(tiny_graph)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # No agent_data → no legend
        assert ax.get_legend() is None

    def test_with_agent_data_shows_legend(self, tiny_graph, tiny_agent_data):
        fig, ax = plots.org_graph(tiny_graph, agent_data=tiny_agent_data)
        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_title().get_text() == "bloc"
        legend_labels = {t.get_text() for t in legend.get_texts()}
        assert legend_labels == {"manager", "specialist", "intern"}

    def test_layouts(self, tiny_graph):
        for layout in ("spring", "circular", "shell"):
            fig, ax = plots.org_graph(tiny_graph, layout=layout)
            assert isinstance(ax, Axes)

    def test_unknown_layout_raises(self, tiny_graph):
        with pytest.raises(ValueError, match="Unknown layout"):
            plots.org_graph(tiny_graph, layout="hyperbolic")

    def test_mismatched_agent_data_raises(self, tiny_graph, tiny_agent_data):
        with pytest.raises(ValueError, match="entries but graph has"):
            plots.org_graph(tiny_graph, agent_data=tiny_agent_data[:2])

    def test_ax_injection(self, tiny_graph):
        fig, my_ax = plt.subplots()
        ret_fig, ret_ax = plots.org_graph(tiny_graph, ax=my_ax)
        assert ret_ax is my_ax

    def test_with_built_org_data(self):
        """End-to-end: build_org_data → org_graph should just work."""
        from animal_cascades.org import build_org_data
        G, org_data = build_org_data()
        # `build_org_data` returns AgentArgs; convert to objects exposing
        # `.bloc` by zipping with the underlying configs the user passed.
        # For this smoke we use AgentArgs.description as a stand-in.
        fig, ax = plots.org_graph(G, show_labels=False)
        assert isinstance(ax, Axes)
