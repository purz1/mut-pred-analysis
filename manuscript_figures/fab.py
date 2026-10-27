"""FAB supplements: composition and positive controls, model performance and errors per FAB group, fingerprint per FAB group.

fab_composition()                 FAB class counts (with missing) and prevalence of every target per class (positive controls: PML-RARA in M3 …)
fab_performance(model)            per target: AUROC, miss rate and false-alarm rate in each FAB group, with the M0–M2 vs M4–M5 test
fab_fingerprint(variable, mod)    δ of the target's features within M0–M2 against M4–M5; the features that differ most between the groups
Inputs: derived/fab (scripts/extract_fab_analysis.py, extract_fab_features.py).
"""
from functools import cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config as C
from .style import MM, W, VARS, vname, vcol

GROUPS = ["M0–M2", "M3", "M4–M5"]
GROUP_COL = {"M0–M2": "#3a6fb0", "M3": "#8c8c8c", "M4–M5": "#e08a1e"}
CLASSES = ["M0", "M1", "M2", "M3", "M4", "M5"]
FAMILY_COL = {"proportion": "#455fa2", "dysplasia": "#a92b27", "size": "#4e867a", "N:C": "#9b9231",
              "nuclear shape": "#5f4480", "cell shape": "#b95a1e", "qc": "#8c8c8c"}
CONTRAST = ("M0–M2", "M4–M5")


@cache
def _t(name):
    return pd.read_csv(C.FAB / f"{name}.csv")


def _p(p):
    return r"$\it{P}$ < 0.001" if p < 0.001 else rf"$\it{{P}}$ = {p:.3f}"


def fab_composition(model="BM"):
    comp = _t("fab_composition")
    comp = comp[comp.model == model].set_index("fab").n
    prev = _t("fab_target_prevalence")
    prev = prev[(prev.model == model) & (prev.level == "class")]
    variables = [v for v in VARS if v in set(prev.variable)]
    with plt.rc_context({"font.size": 5.5}):
        fig = plt.figure(figsize=(W * MM, (30 + 4.2 * len(variables)) * MM))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], left=0.06, right=0.97, top=1 - 8 / (30 + 4.2 * len(variables)), bottom=14 / (30 + 4.2 * len(variables)), wspace=0.5)
        ax = fig.add_subplot(gs[0])
        order = CLASSES + ["missing"]
        vals = [int(comp.get(c, 0)) for c in order]
        ax.bar(range(len(order)), vals, color=["#3a6fb0"] * 3 + ["#8c8c8c"] + ["#e08a1e"] * 2 + ["#d9d9d9"], width=0.7)
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=5)
        ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
        ax.set_ylabel("Samples")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("FAB class", fontsize=6, fontweight="bold")
        ax = fig.add_subplot(gs[1])
        pv = prev.pivot(index="variable", columns="group", values="prevalence").reindex(index=variables, columns=CLASSES) * 100
        n = prev.pivot(index="variable", columns="group", values="n_pos").reindex(index=variables, columns=CLASSES)
        im = ax.imshow(pv.to_numpy(float), cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(len(variables)):
            for j in range(len(CLASSES)):
                if pd.notna(pv.iat[i, j]):
                    ax.text(j, i, f"{pv.iat[i, j]:.0f}\n({int(n.iat[i, j])})", ha="center", va="center", fontsize=4,
                            color="white" if pv.iat[i, j] > 55 else "k")
        ax.set_xticks(range(len(CLASSES)), CLASSES)
        ax.set_yticks(range(len(variables)), [vname(v) for v in variables])
        for t, v in zip(ax.get_yticklabels(), variables):
            t.set_color(vcol(v))
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        ax.set_title("Positive samples, % of the class (n)", fontsize=6, fontweight="bold", pad=14)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=4.5)
    return fig


def _forest(ax, d, variables, col, xlabel, xlim=None, minn=1):
    """One marker + CI per FAB group and target."""
    for i, v in enumerate(variables):
        for k, g in enumerate(GROUPS):
            r = d[(d.variable == v) & (d.group == g)]
            if len(r) and pd.notna(r[col].iat[0]) and r["n"].iat[0] >= minn:
                y = i + (k - 1) * 0.22
                r = r.iloc[0]
                ax.plot([r[col + "_low"], r[col + "_high"]], [y, y], color=GROUP_COL[g], lw=0.7)
                ax.scatter(r[col], y, s=7, color=GROUP_COL[g], zorder=3, lw=0)
    ax.set_yticks(range(len(variables)))
    ax.set_ylim(len(variables) - 0.4, -0.6)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="y", lw=0.2, color="0.85")
    ax.spines[["top", "right"]].set_visible(False)


def fab_performance(model):
    perf, err, con = _t("fab_performance"), _t("fab_errors"), _t("fab_contrast")
    perf, err, con = (t[t.model == model] for t in (perf, err, con))
    variables = [v for v in VARS if v in set(perf.variable)]
    err = err.rename(columns={"rate": "rate", "low": "rate_low", "high": "rate_high"})
    H = 18 + 5.4 * len(variables)
    with plt.rc_context({"font.size": 5.5}):
        fig, axes = plt.subplots(1, 3, figsize=(W * MM, H * MM), sharey=True)
        fig.subplots_adjust(left=0.17, right=0.99, top=1 - 14 / H, bottom=9 / H, wspace=0.08)
        _forest(axes[0], perf, variables, "auroc", "AUROC", (0.3, 1.02))
        _forest(axes[1], err[err.kind == "miss"], variables, "rate", "Miss rate (FN / positives)", (-0.02, 1.02))
        _forest(axes[2], err[err.kind == "false_alarm"], variables, "rate", "False-alarm rate (FP / negatives)", (-0.02, 0.6))
        axes[0].set_yticks(range(len(variables)), [vname(v) for v in variables])
        for t, v in zip(axes[0].get_yticklabels(), variables):
            t.set_color(vcol(v))
        for ax, kind in zip(axes[1:], ("miss", "false_alarm")):
            for i, v in enumerate(variables):
                r = con[(con.variable == v) & (con.kind == kind)]
                if len(r) and pd.notna(r.padj.iat[0]) and r.padj.iat[0] < 0.05:
                    ax.text(1.0, i, "*", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)
        axes[0].axvline(0.5, color="0.7", lw=0.4, ls=":")
        handles = [plt.Line2D([], [], marker="o", color=GROUP_COL[g], lw=0.7, ms=3, label=f"{g}") for g in GROUPS]
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.0), fontsize=5.5)
    return fig


def fab_fingerprint(variable="NPM1", modality="BM", n_top=10):
    d = _t("fab_feature_effects")
    d = d[(d.variable == variable) & (d.modality == modality)].dropna(subset=[f"delta_{CONTRAST[0]}", f"delta_{CONTRAST[1]}"]).copy()
    a, b = (f"delta_{g}" for g in CONTRAST)
    d["label"] = d.cell_type.fillna("").str.replace("_", " ") + " – " + d.feature.fillna(d.feature_col).str.replace("_", " ")
    top = pd.concat([d[d.hypothesis], d[~d.hypothesis].reindex(d[~d.hypothesis].delta_diff.abs().sort_values(ascending=False).index).head(n_top)])
    with plt.rc_context({"font.size": 5.5}):
        fig = plt.figure(figsize=(W * MM, 92 * MM))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], left=0.07, right=0.98, top=0.88, bottom=0.12, wspace=0.75)
        ax = fig.add_subplot(gs[0])
        for fam, g in d.groupby("family"):
            ax.scatter(g[a], g[b], s=4, color=FAMILY_COL.get(fam, "k"), alpha=0.7, lw=0, label=fam)
        lim = max(0.5, np.nanmax(np.abs(d[[a, b]].to_numpy())) * 1.05)
        ax.plot([-lim, lim], [-lim, lim], color="0.5", lw=0.5, ls="--")
        ax.axhline(0, color="0.8", lw=0.3); ax.axvline(0, color="0.8", lw=0.3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel(f"Cliff's δ in {CONTRAST[0]}"); ax.set_ylabel(f"Cliff's δ in {CONTRAST[1]}")
        r = d[[a, b]].corr(method="spearman").iat[0, 1]
        ax.set_title(f"{vname(variable)} ({modality}): {len(d)} features, ρ = {r:.2f}", fontsize=6, fontweight="bold", color=vcol(variable))
        for _, row in d[d.hypothesis].iterrows():
            ax.scatter(row[a], row[b], s=14, facecolor="none", edgecolor="k", lw=0.7)
        ax.legend(fontsize=4.5, frameon=False, loc="lower right", markerscale=1.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax2 = fig.add_subplot(gs[1])
        top = top.iloc[::-1].reset_index(drop=True)
        for i, row in top.iterrows():
            ax2.plot([row[a], row[b]], [i, i], color="0.6", lw=0.8)
            ax2.scatter(row[a], i, s=9, color=GROUP_COL[CONTRAST[0]], zorder=3, lw=0)
            ax2.scatter(row[b], i, s=9, color=GROUP_COL[CONTRAST[1]], zorder=3, lw=0)
            if pd.notna(row.get("padj_interaction")) and row.padj_interaction < 0.05:
                ax2.text(1.0, i, "*", transform=ax2.get_yaxis_transform(), ha="right", va="center", fontsize=8)
        ax2.axvline(0, color="0.8", lw=0.3)
        ax2.set_yticks(range(len(top)), [("◆ " if h else "") + t for t, h in zip(top.label, top.hypothesis)], fontsize=4.8)
        ax2.set_xlabel("Cliff's δ"); ax2.spines[["top", "right"]].set_visible(False)
        ax2.set_title("Pre-specified (◆) and largest differences between the groups", fontsize=5.5, pad=6)
        handles = [plt.Line2D([], [], marker="o", color=GROUP_COL[g], lw=0, ms=3, label=g) for g in CONTRAST]
        ax2.legend(handles=handles, fontsize=5, frameon=False, loc="lower right")
    return fig
