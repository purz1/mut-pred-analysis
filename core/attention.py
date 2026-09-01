"""Attention analysis utilities for MIL model outputs.

Configure the utils module before importing plotting functions:
    import utils
    utils.CELL_NAMES = CELL_NAMES
    utils.CUSTOM_VARIABLES = CUSTOM_VARIABLES
    utils.SAVE_FIGURES = SAVE_FIGURES
    ...
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from core import utils
from core.utils import save_and_show, get_readable_variable_name

MASS_THRESHOLD = 0.5

CELL_COLORS = {
    "Proerythroblasts": "maroon",
    "Erythroblasts": "lightcoral",
    "Megakaryocytes": "darkorange",
    "Blasts": "tomato",
    "Promyelocytes": "darkolivegreen",
    "Myelocytes": "yellowgreen",
    "Metamyelocytes": "springgreen",
    "Basophils": "tab:brown",
    "Neutrophils": "dodgerblue",
    "Band_neutrophils": "steelblue",
    "Segmented_neutrophils": "cornflowerblue",
    "Eosinophils_immature": "mediumpurple",
    "Eosinophils": "blueviolet",
    "Promonocytes": "navy",
    "Monocytes": "gold",
    "Macrophages": "darkkhaki",
    "Lymphocytes": "teal",
    "Plasma_cells": "turquoise",
    "Giant_platelets": "salmon",
    "Platelet_aggregations": "lightsalmon",
}


# ── Internal helpers ──────────────────────────────────────────────────────────


def lorenz_curve(attention):
    a = np.sort(attention)[::-1]
    cum_att = np.cumsum(a) / a.sum()
    cum_cells = np.arange(1, len(a) + 1) / len(a)
    return cum_cells, cum_att


def aggregate_lorenz(bags, num_points=100):
    x_grid = np.linspace(0, 1, num_points)
    curves = np.stack([np.interp(x_grid, *lorenz_curve(a)) for a in bags])
    return x_grid, curves.mean(axis=0), curves.std(axis=0)


def cells_for_mass(attention, threshold=0.5):
    """Fraction of top-attending cells needed to reach `threshold` of total attention."""
    a = np.sort(attention)[::-1]
    cum = np.cumsum(a) / a.sum()
    idx = np.searchsorted(cum, threshold)
    return (idx + 1) / len(a)


def effective_num_cells(attention, eps=1e-12):
    a = attention + eps
    return np.exp(-np.sum(a * np.log(a)))


def compute_metrics_per_bag(attention, cell_types, num_types, eps=1e-8, min_cells=5):
    N = len(attention)
    counts = np.bincount(cell_types, minlength=num_types)
    A = np.bincount(cell_types, weights=attention, minlength=num_types)
    p = counts / (N + eps)
    logR = np.log(A + eps) - np.log(p + eps)
    Hnorm = np.full(num_types, np.nan)
    for t in range(num_types):
        n_t = counts[t]
        if n_t < min_cells:
            continue
        mask = cell_types == t
        a_t = attention[mask]
        a_t = a_t / (A[t] + eps)
        H = -np.sum(a_t * np.log(a_t + eps))
        Hnorm[t] = H / np.log(n_t + eps)
    return logR, Hnorm, counts


def compute_logR_summary(df, target_variables, ct_names, subset=None):
    """Mean log(R) per cell type across bags for the given subset.

    Returns DataFrame: rows = cell types (ct_names order), columns = variables.
    subset: None = all, 'TP', 'TN', 'FP', 'FN', 'pred_pos', 'pred_neg'
    """
    _ct_map = {name: i for i, name in enumerate(ct_names)}
    _n_types = len(ct_names)
    result = {}
    for variable in target_variables:
        attn_col = f"cell_attention_score_{variable}"
        true_col = f"{variable}_true_label"
        pred_col = f"{variable}_pred_label"
        sub = df[["bag_id", "cell_label", attn_col, true_col, pred_col]].dropna(
            subset=[attn_col, true_col]
        )
        if subset == "TP":
            sub = sub[(sub[true_col] == 1) & (sub[pred_col] == 1)]
        elif subset == "TN":
            sub = sub[(sub[true_col] == 0) & (sub[pred_col] == 0)]
        elif subset == "FP":
            sub = sub[(sub[true_col] == 0) & (sub[pred_col] == 1)]
        elif subset == "FN":
            sub = sub[(sub[true_col] == 1) & (sub[pred_col] == 0)]
        elif subset == "pred_pos":
            sub = sub[sub[pred_col] == 1]
        elif subset == "pred_neg":
            sub = sub[sub[pred_col] == 0]
        logR_bags = []
        for _, bag_df in sub.groupby("bag_id"):
            attention = bag_df[attn_col].to_numpy()
            cell_types = bag_df["cell_label"].map(_ct_map).to_numpy()
            valid = ~np.isnan(cell_types)
            attention = attention[valid]
            cell_types = cell_types[valid].astype(int)
            if len(attention) == 0:
                continue
            logR, _, _ = compute_metrics_per_bag(attention, cell_types, _n_types)
            logR_bags.append(logR)
        if logR_bags:
            mean_logR = np.nanmean(np.stack(logR_bags), axis=0)
        else:
            mean_logR = np.full(_n_types, np.nan)
        result[variable] = pd.Series(mean_logR, index=ct_names)
    return pd.DataFrame(result)[target_variables]


def compute_attention_mass_summary(
    df_bm, target_variables, ct_names, subset=None, mass_threshold=MASS_THRESHOLD
):
    """For each bag, take the smallest set of top-attending cells that together
    account for `mass_threshold` fraction of total attention, then average cell-type
    proportions across bags.

    subset: None = all, 'pred_pos', 'pred_neg', 'TP', 'TN', 'FP', 'FN'
    """

    def select_mass_cells(g):
        g = g.sort_values(attn_col, ascending=False)
        cum = g[attn_col].cumsum() / g[attn_col].sum()
        cutoff = cum.index[(cum >= mass_threshold).values][0]
        return g.loc[:cutoff]

    result = {}
    for variable in target_variables:
        attn_col = f"cell_attention_score_{variable}"
        true_col = f"{variable}_true_label"
        pred_col = f"{variable}_pred_label"
        sub = df_bm[["bag_id", "cell_label", attn_col, true_col, pred_col]].dropna(
            subset=[attn_col]
        )
        if subset == "TP":
            sub = sub[(sub[true_col] == 1) & (sub[pred_col] == 1)]
        elif subset == "TN":
            sub = sub[(sub[true_col] == 0) & (sub[pred_col] == 0)]
        elif subset == "FP":
            sub = sub[(sub[true_col] == 0) & (sub[pred_col] == 1)]
        elif subset == "FN":
            sub = sub[(sub[true_col] == 1) & (sub[pred_col] == 0)]
        elif subset == "pred_pos":
            sub = sub[sub[pred_col] == 1]
        elif subset == "pred_neg":
            sub = sub[sub[pred_col] == 0]
        top_cells = (
            sub.groupby("bag_id")
            .apply(select_mass_cells, include_groups=False)
            .reset_index(level=0)
            .reset_index(drop=True)
        )
        counts = top_cells.groupby(["bag_id", "cell_label"]).size()
        totals = top_cells.groupby("bag_id").size()
        proportions = counts / totals.reindex(counts.index, level="bag_id")
        result[variable] = proportions.groupby(level="cell_label").mean()
    return pd.DataFrame(result).reindex(ct_names)[target_variables].fillna(np.nan)


# ── Top-level analysis functions ──────────────────────────────────────────────


def compute_celltype_mass_bins(
    df, target_variables, ct_names, n_bins=10, subset=None, cell_groups=None
):
    """Assign cells to attention-mass decile bins, return mean cell-type proportions per bin.

    subset: None = all bags, 'pred_pos', 'pred_neg'
    cell_groups: dict of label -> [cell_type_names]. When provided, cell types are summed
        into groups before normalizing, so each group is one row in the output (indexed by
        label). This ensures group proportions sum to 1 and avoids seams in stacked plots.
    group_colors: dict of label -> color (used alongside cell_groups, ignored here).
    Returns: dict of variable -> DataFrame (index=ct_names or labels, columns=1..n_bins)
    """
    ct_names_set = set(ct_names)
    if cell_groups is not None:
        groups = [
            (label, [ct for ct in cts if ct in ct_names_set])
            for label, cts in cell_groups.items()
            if any(ct in ct_names_set for ct in cts)
        ]
        ct_to_row = {ct: i for i, (_, cts) in enumerate(groups) for ct in cts}
        n_rows = len(groups)
        row_names = [label for label, _ in groups]
    else:
        ct_to_row = {name: i for i, name in enumerate(ct_names)}
        n_rows = len(ct_names)
        row_names = list(ct_names)
    result = {}
    for variable in target_variables:
        attn_col = f"cell_attention_score_{variable}"
        pred_col = f"{variable}_pred_label"
        sub = df[["bag_id", "cell_label", attn_col, pred_col]].dropna(subset=[attn_col])
        if subset == "pred_pos":
            sub = sub[sub[pred_col] == 1]
        elif subset == "pred_neg":
            sub = sub[sub[pred_col] == 0]
        bag_mats = []
        for _, bag_df in sub.groupby("bag_id"):
            a = bag_df[attn_col].to_numpy()
            ct_arr = bag_df["cell_label"].to_numpy()
            order = np.argsort(a)[::-1]
            a_sorted = a[order]
            ct_sorted = ct_arr[order]
            cum_frac = np.cumsum(a_sorted) / a_sorted.sum()
            bin_ids = np.minimum((cum_frac * n_bins).astype(int), n_bins - 1)
            mat = np.zeros((n_rows, n_bins))
            for b_id, ct_name in zip(bin_ids, ct_sorted):
                if ct_name in ct_to_row:
                    mat[ct_to_row[ct_name], b_id] += 1
            col_sums = mat.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            bag_mats.append(mat / col_sums)
        mean_mat = (
            np.stack(bag_mats).mean(axis=0) if bag_mats else np.zeros((n_rows, n_bins))
        )
        result[variable] = pd.DataFrame(
            mean_mat, index=row_names, columns=range(1, n_bins + 1)
        )
    return result


def plot_celltype_mass_bins(
    bin_data,
    target_variables,
    ct_names,
    n_cols=5,
    title="All bags",
    figsize=None,
    figure_type=None,
    cell_groups=None,
    group_colors=None,
    hspace=None,
    wspace=None,
):
    """Stacked area chart of cell-type proportions per attention-mass decile, one subplot per variable."""
    n_vars = len(target_variables)
    n_rows = (n_vars + n_cols - 1) // n_cols
    if cell_groups is not None:
        ct_to_label = {ct: label for label, cts in cell_groups.items() for ct in cts}
        ct_names = [ct for ct in ct_names if ct in ct_to_label]
        ct_names_set = set(ct_names)
        plot_groups = [
            (
                label,
                group_colors.get(label, "gray"),
                [ct for ct in cts if ct in ct_names_set],
            )
            for label, cts in cell_groups.items()
            if any(ct in ct_names_set for ct in cts)
        ]
        legend_colors = [color for _, color, _ in plot_groups]
        legend_labels = [label for label, _, _ in plot_groups]
        plot_groups_reversed = list(reversed(plot_groups))
    else:
        colors = [CELL_COLORS.get(ct, "gray") for ct in ct_names]
        legend_colors = colors
        legend_labels = [get_readable_variable_name(ct) for ct in ct_names]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize or (n_cols * 2.5, n_rows * 2.8),
        squeeze=False,
        layout="constrained",
    )
    if hspace is not None or wspace is not None:
        fig.set_constrained_layout_pads(
            **{"hspace": hspace} if hspace is not None else {},
            **{"wspace": wspace} if wspace is not None else {},
        )
    for idx, variable in enumerate(target_variables):
        ax = axes[idx // n_cols][idx % n_cols]
        data = bin_data[variable]
        if cell_groups is not None:
            ax.stackplot(
                data.columns,
                [data.loc[label].to_numpy() for label, _, _ in plot_groups_reversed],
                colors=[color for _, color, _ in plot_groups_reversed],
            )
        else:
            ax.stackplot(
                data.columns,
                [data.loc[ct].to_numpy() for ct in ct_names],
                colors=colors,
            )
        ax.set_title(get_readable_variable_name(variable), fontsize=utils.FONTSIZE - 1)
        ax.set_ylabel("Cells (%)", fontsize=utils.FONTSIZE - 1, labelpad=2)
        ax.set_xticks(range(2, data.shape[1] + 1, 2))
        ax.set_xticklabels([f"{i*10}%" for i in range(2, data.shape[1] + 1, 2)])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}"))
        ax.tick_params(labelsize=utils.FONTSIZE - 2)
        ax.set_xlim(1, 10)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    for idx in range(n_vars, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    last_row_start = (n_rows - 1) * n_cols
    for idx in range(n_vars):
        ax = axes[idx // n_cols][idx % n_cols]
        if idx < last_row_start:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        if idx % n_cols != 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in legend_colors]
    fig.suptitle(title, fontsize=utils.FONTSIZE, fontweight="bold")
    fig.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=5,
        fontsize=utils.FONTSIZE - 1,
        frameon=False,
    )
    save_and_show(fig, title=title, figure_type=figure_type)


def plot_heatmap(
    panels,
    cbar_label="Mean proportion",
    annotate_positive_values=False,
    cmap="Blues",
    center=None,
    figsize=None,
    figure_type=None,
    cell_groups=None,
    group_colors=None,
    strip_pts=6,
    strip_gap_pts=1,
):
    """Plot one heatmap figure per entry in `panels` (list of (DataFrame, title_str))."""
    vmin = min(m.min().min() for m, _ in panels) if center is not None else 0
    vmax = max(m.max().max() for m, _ in panels)
    for summary, title in panels:
        data = summary.T
        data.index = [get_readable_variable_name(i) for i in data.index]
        data.columns = [get_readable_variable_name(c) for c in data.columns]
        nrows, ncols = data.shape
        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize
            or (min(9, ncols * 0.78 + 2.6), min(7.8, nrows * 0.29 + 1.3)),
        )
        sns.heatmap(
            data,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            linewidths=0.4,
            linecolor="white",
            annot=False,
            square=True,
            cbar_kws={"label": cbar_label, "shrink": 0.5},
        )
        cbar = ax.collections[0].colorbar
        cbar.set_label(cbar_label, fontsize=utils.FONTSIZE - 1)
        cbar.ax.tick_params(labelsize=utils.FONTSIZE - 2)
        if annotate_positive_values:
            for row_idx, row_label in enumerate(data.index):
                for col_idx, col_label in enumerate(data.columns):
                    val = data.loc[row_label, col_label]
                    if pd.notna(val) and val > 0:
                        ax.text(
                            col_idx + 0.5,
                            row_idx + 0.5,
                            f"{val:.1f}",
                            ha="center",
                            va="center",
                            fontsize=utils.FONTSIZE - 2,
                            color="black",
                        )
        ax.set_title(
            title, pad=10, fontdict={"fontsize": utils.FONTSIZE, "fontweight": "bold"}
        )
        ax.tick_params(axis="x", rotation=45, labelsize=utils.FONTSIZE - 1, length=0)
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation_mode("anchor")
        ax.tick_params(axis="y", rotation=0, labelsize=utils.FONTSIZE - 1, length=0)
        ax.set_xlabel("")
        if cell_groups is not None and group_colors is not None:
            from matplotlib.transforms import blended_transform_factory
            from matplotlib.patches import Rectangle as MplRect

            ct_to_color = {
                ct: group_colors[label]
                for label, cts in cell_groups.items()
                for ct in cts
            }
            readable_to_color = {
                get_readable_variable_name(ct): c for ct, c in ct_to_color.items()
            }
            fig_h_pts = fig.get_size_inches()[1] * 72
            strip_h = strip_pts / fig_h_pts
            gap_h = strip_gap_pts / fig_h_pts
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            for i, col in enumerate(data.columns):
                color = readable_to_color.get(col)
                if color:
                    ax.add_patch(
                        MplRect(
                            (i, -(strip_h + gap_h)),
                            1.0,
                            strip_h,
                            transform=trans,
                            facecolor=color,
                            edgecolor="none",
                            clip_on=False,
                            zorder=2,
                        )
                    )
        plt.tight_layout()
        save_and_show(fig, title=title, figure_type=figure_type)


def plot_importance_heatmaps(
    df,
    target_variables,
    ct_names,
    figsize=None,
    figure_type=None,
    title_pred_pos="Cell type importance in \npositive predictive samples",
    title_pred_neg="Cell type importance in \nnegative predictive samples",
    cell_groups=None,
    group_colors=None,
):
    """Log(R) heatmaps for predicted-positive and predicted-negative subsets."""
    logR_pred_pos = compute_logR_summary(
        df, target_variables, ct_names, subset="pred_pos"
    )
    logR_pred_neg = compute_logR_summary(
        df, target_variables, ct_names, subset="pred_neg"
    )
    plot_heatmap(
        panels=[
            (logR_pred_pos, title_pred_pos),
            (logR_pred_neg, title_pred_neg),
        ],
        cbar_label="Relative attention enrichment (log scale)",
        cmap="RdBu_r",
        center=0,
        annotate_positive_values=False,
        figsize=figsize,
        figure_type=figure_type,
        cell_groups=cell_groups,
        group_colors=group_colors,
    )


def plot_lorenz_overlay(
    df, target_variables, threshold=0.5, figsize=None, title=None, figure_type=None
):
    """Overlay Lorenz curves for all variables, sorted by concentration."""
    if title is None:
        title = (
            "Lorenz curves by variable  (sorted by cell fraction at 50% attention mass)"
        )
    var_data = []
    for variable in target_variables:
        attn_col = f"cell_attention_score_{variable}"
        sub = df[["bag_id", attn_col]].dropna(subset=[attn_col])
        bags = [g[attn_col].to_numpy() for _, g in sub.groupby("bag_id")]
        frac = np.mean([cells_for_mass(a, threshold=threshold) for a in bags])
        var_data.append((variable, bags, frac))
    var_data.sort(key=lambda x: x[2])
    colors = plt.cm.tab20(np.linspace(0, 1, len(var_data)))
    fig, ax = plt.subplots(figsize=figsize or (6, 5))
    for (variable, bags, frac), color in zip(var_data, colors):
        x, mean, _ = aggregate_lorenz(bags)
        ax.plot(
            x,
            mean,
            label=f"{get_readable_variable_name(variable)} ({frac:.2f})",
            color=color,
            linewidth=1.2,
        )
    for thr in (0.5, 0.9):
        ax.axhline(thr, color="black", linestyle="--", linewidth=0.8)
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=0.8)
    ax.set_xlabel("Fraction of cells")
    ax.set_ylabel("Fraction of attention")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    plt.tight_layout()
    save_and_show(fig, title=title, figure_type=figure_type)


def plot_neff_by_variable(
    df,
    target_variables,
    use_cell_fractions=False,
    figsize=None,
    title=None,
    ylabel=None,
    figure_type=None,
):
    """Cross-variable boxplot of effective number of cells (N_eff = exp entropy)."""
    if title is None:
        title = "Attention spread by variable"
    rows = []
    for variable in target_variables:
        attn_col = f"cell_attention_score_{variable}"
        true_col = f"{variable}_true_label"
        sub = df[["bag_id", attn_col, true_col]].dropna(subset=[attn_col, true_col])
        for _, bag_df in sub.groupby("bag_id"):
            a = bag_df[attn_col].to_numpy()
            neff = effective_num_cells(a)
            value = neff / len(a) if use_cell_fractions else neff
            label = "Positive" if bag_df[true_col].iloc[0] == 1 else "Negative"
            rows.append(
                {
                    "Variable": get_readable_variable_name(variable),
                    "N_eff": value,
                    "Label": label,
                }
            )
    df_plot = pd.DataFrame(rows)
    palette = {"Positive": "tab:red", "Negative": "tab:blue"}
    black_palette = {"Positive": "black", "Negative": "black"}
    hue_order = ["Positive", "Negative"]
    readable_order = [get_readable_variable_name(v) for v in target_variables]
    fig, ax = plt.subplots(figsize=figsize or (max(6, len(target_variables) * 0.6), 5))
    sns.boxplot(
        data=df_plot,
        x="Variable",
        y="N_eff",
        hue="Label",
        palette=palette,
        hue_order=hue_order,
        order=readable_order,
        dodge=True,
        width=0.5,
        showfliers=False,
        whis=1.5,
        linewidth=0.8,
        boxprops=dict(alpha=0.8),
        whiskerprops=dict(linestyle="--"),
        capprops=dict(linestyle="--"),
        medianprops=dict(color="black"),
        ax=ax,
    )
    sns.stripplot(
        data=df_plot,
        x="Variable",
        y="N_eff",
        hue="Label",
        palette=black_palette,
        hue_order=hue_order,
        order=readable_order,
        dodge=True,
        size=2,
        linewidth=0,
        alpha=0.3,
        ax=ax,
    )
    ax.set_xticks(range(len(target_variables)))
    ax.set_xticklabels(
        readable_order,
        rotation=45,
        ha="right",
        fontdict={"fontsize": utils.FONTSIZE - 1},
    )
    ax.set_xlabel("")
    ax.set_ylabel(
        ylabel
        or (
            "N effective / total cells"
            if use_cell_fractions
            else "N effective  (exp entropy)"
        ),
        fontdict={"fontsize": utils.FONTSIZE - 1},
    )
    ax.tick_params(axis="y", labelsize=utils.FONTSIZE - 1)
    ax.set_title(title, fontdict={"fontsize": utils.FONTSIZE, "fontweight": "bold"})
    ax.spines[["top", "right"]].set_visible(False)
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(
        handles[:2], lbls[:2], title="", fontsize=utils.FONTSIZE - 1, frameon=False
    )
    plt.tight_layout()
    save_and_show(fig, title=title, figure_type=figure_type)


def select_top_attention_cell_ids(
    df_attn, target_variable, threshold=0.5, modality_filter=None
):
    """Return (database_id, album_id, cell_id) rows for cells in the top-threshold
    attention mass per sample for `target_variable`.

    modality_filter: optional (column, value) tuple to pre-filter df_attn rows,
                     e.g. ('modality', 'BM') for the Fusion model.
    """
    score_col = f"cell_attention_score_{target_variable}"
    df = (
        df_attn
        if modality_filter is None
        else df_attn[df_attn[modality_filter[0]] == modality_filter[1]]
    )
    rows = []
    for (db_id, alb_id), grp in df.groupby(["database_id", "album_id"]):
        scores = grp[score_col].to_numpy(dtype=float)
        total = scores.sum()
        if total == 0:
            continue
        order = np.argsort(scores)[::-1]
        cum = np.cumsum(scores[order]) / total
        n_keep = int(np.searchsorted(cum, threshold)) + 1
        kept_ids = grp.iloc[order[:n_keep]]["cell_id"].to_numpy()
        rows.extend(
            {"database_id": str(db_id), "album_id": str(alb_id), "cell_id": int(cid)}
            for cid in kept_ids
        )
    return pd.DataFrame(rows, columns=["database_id", "album_id", "cell_id"])
