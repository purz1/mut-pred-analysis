import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from typing import Optional, Dict, Tuple, Union, List
from pathlib import Path

from core.visualization.lifelines_plotting import add_at_risk_counts


def plot_kaplan_meier_curves(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_cols: List[str],
    fontsize: int = 7,
    custom_legends: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    xlabel: str = "Time",
    ylabel: str = "Survival Probability",
    xlim: Tuple[float, float] = (0, 48),
    ylim: Tuple[float, float] = (0, 1.13),
    figsize: Tuple[float, float] = (3, 2.5),
    save_path: Optional[Union[str, Path]] = None,
    max_p_value_to_plot = None,
) -> plt.Figure:

    df = df.copy()

    # ------------------------------------------------------------------
    # Validate boolean, non-overlapping group columns
    # ------------------------------------------------------------------
    group_matrix = df[group_cols].astype(bool)
    membership_count = group_matrix.sum(axis=1)

    if (membership_count > 1).any():
        overlapping = df.loc[membership_count > 1, group_cols]
        raise ValueError(
            "Group columns must be mutually exclusive. "
            f"Found overlapping memberships in {len(overlapping)} rows."
        )

    if (membership_count == 0).any():
        raise ValueError(
            "Each row must belong to exactly one group. "
            "Found rows with no group assignment."
        )

    # Create a single categorical group variable
    df["_group"] = group_matrix.idxmax(axis=1)

    group_order = (
        list(custom_legends.keys())
        if custom_legends
        else group_cols
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axhline(y=0.5, color="black", linestyle="--", linewidth=1)

    fitters = []

    for group in group_order:
        group_data = df[df["_group"] == group]
        if group_data.empty:
            continue

        kmf = KaplanMeierFitter()

        label = custom_legends.get(group, group) if custom_legends else group
        color = colors.get(group) if colors else None

        kmf.fit(
            durations=group_data[time_col],
            event_observed=group_data[event_col],
            label=label,
        )

        fitters.append(kmf)

        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            at_risk_counts=False,
            show_censors=True,
            color=color,
            linewidth=2,
        )

        final_prob = kmf.survival_function_at_times(
            group_data[time_col].max()
        ).item()
        #print(f"Probability ({label}) = {round(final_prob, 3)}")

    # ------------------------------------------------------------------
    # Multivariate log-rank test (valid here)
    # ------------------------------------------------------------------
    stats = multivariate_logrank_test(
        df[time_col],
        df["_group"],
        df[event_col],
    )

    ax.text(
        xlim[1] * 0.8,
        0.1,
        fr"$\it{{P}}$ = {stats.p_value:.3f}",
        fontsize=fontsize - 1,
    )

    # ------------------------------------------------------------------
    # Axes & styling
    # ------------------------------------------------------------------
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=fontsize - 1)
    ax.set_ylabel(ylabel, fontsize=fontsize - 1)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 12))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.01, 0.25))
    ax.xaxis.set_tick_params(labelsize=fontsize-1)
    ax.yaxis.set_tick_params(labelsize=fontsize-1)
    
    if True:
        add_at_risk_counts(
            *fitters,
            ax=ax,
            xticks=np.arange(xlim[0], xlim[1] + 1, 12),
            rows_to_show=["At risk"],
            fontsize=fontsize - 1,
        )

    ax.legend(
        loc="upper right",
        prop={"size": fontsize - 1},
        frameon=False,
    )

    ax.spines[["right", "top"]].set_visible(False)

    if title:
        ax.set_title(title, fontsize=fontsize, fontweight="bold")

    if max_p_value_to_plot is not None and stats.p_value > max_p_value_to_plot:
        plt.close()
        return fig

    return fig