"""Co-occurring targets: overview of the label correlations and the model's coupling, and the fingerprint of a target adjusted for another.

cooc_overview(model)        phi of the true labels (lower triangle) and of the model's predicted labels (upper triangle); nested pairs hatched
cooc_retained()             one bar per pair (X | Y): share of the effect of X's significant features that remains after adjusting for Y
cooc_specificity(modality)  detail: one panel per pair (X | Y): δ of X's features before (x) and after (y) adjusting for Y
Inputs: derived/cooccurrence (scripts/extract_cooccurrence.py, extract_cooccurrence_features.py).
"""
from functools import cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import config as C
from .style import MM, W, VARS, vname, vcol

FAMILY_COL = {"proportion": "#455fa2", "dysplasia": "#a92b27", "size": "#4e867a", "N:C": "#9b9231",
              "nuclear shape": "#5f4480", "cell shape": "#b95a1e", "qc": "#8c8c8c"}
NCOL = 4


@cache
def _t(name):
    return pd.read_csv(C.DERIVED / "cooccurrence" / f"{name}.csv")


def cooc_overview(model="BM"):
    pc, pm = _t("pair_counts"), _t("pair_model")
    pm = pm[pm.model == model]
    n = len(VARS)
    true, pred, nested = (np.full((n, n), np.nan) for _ in range(3))
    idx = {v: i for i, v in enumerate(VARS)}
    for r in pc.itertuples():
        i, j = idx[r.x], idx[r.y]
        lo, hi = max(i, j), min(i, j)
        true[lo, hi] = r.phi
        nested[lo, hi] = nested[hi, lo] = r.nested
    for r in pm.itertuples():
        i, j = idx[r.x], idx[r.y]
        pred[min(i, j), max(i, j)] = r.phi_pred
    with plt.rc_context({"font.size": 5.5}):
        fig, ax = plt.subplots(figsize=(W * MM * 0.7, W * MM * 0.62), constrained_layout=True)
        m = np.where(np.isnan(true), pred, true)
        im = ax.imshow(m, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
        for i in range(n):
            for j in range(n):
                if i != j and nested[i, j] == 1:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", edgecolor="0.4", lw=0))
                if i == j:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, color="0.92", lw=0))
        ax.set_xticks(range(n), [vname(v) for v in VARS], rotation=60, ha="right", fontsize=4.8)
        ax.set_yticks(range(n), [vname(v) for v in VARS], fontsize=4.8)
        for t, v in zip(ax.get_xticklabels(), VARS):
            t.set_color(vcol(v))
        for t, v in zip(ax.get_yticklabels(), VARS):
            t.set_color(vcol(v))
        ax.tick_params(length=0)
        ax.set_title(f"φ: true labels (lower triangle) and {model} model's predicted labels (upper triangle)", fontsize=5.5, pad=4)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=4.5)
    return fig


def cooc_specificity(modality="BM"):
    d = _t("cooc_feature_effects")
    s = _t("cooc_summary")
    d, s = d[d.modality == modality], s[s.modality == modality]
    pairs = list(s[["x", "y"]].itertuples(index=False, name=None))
    nrow = -(-len(pairs) // NCOL)
    with plt.rc_context({"font.size": 5.5}):
        fig, axes = plt.subplots(nrow, NCOL, figsize=(W * MM, (8 + 50 * nrow) * MM), squeeze=False, constrained_layout=True)
        for ax, (x, y) in zip(axes.flat, pairs):
            g = d[(d.x == x) & (d.y == y)].dropna(subset=["delta_all", "delta_adj"])
            sig = g.padj_all < 0.05
            ax.scatter(g.delta_all[~sig], g.delta_adj[~sig], s=3, color="0.8", lw=0)
            ax.scatter(g.delta_all[sig], g.delta_adj[sig], s=9, color="k", lw=0)
            lim = max(0.5, np.nanmax(np.abs(g[["delta_all", "delta_adj"]].to_numpy())) * 1.05)
            ax.plot([-lim, lim], [-lim, lim], color="0.5", lw=0.5, ls="--")
            ax.axhline(0, color="0.85", lw=0.3); ax.axvline(0, color="0.85", lw=0.3)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            r = s[(s.x == x) & (s.y == y)].iloc[0]
            ratio = f"\nmedian ratio {r.median_ratio:.2f}" if pd.notna(r.median_ratio) else ""
            ax.set_title(f"{vname(x)}\nadjusted for {vname(y)}", fontsize=4.9, fontweight="bold", color=vcol(x))
            ax.text(0.03, 0.97, f"{int(r.n_sig_all)} significant{ratio}", transform=ax.transAxes, va="top", fontsize=4.4)
            ax.set_xlabel("δ unadjusted", fontsize=5); ax.set_ylabel("δ adjusted", fontsize=5)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes.flat[len(pairs):]:
            ax.axis("off")
    return fig


def cooc_retained():
    d, s = _t("cooc_feature_effects"), _t("cooc_summary")
    pairs = list(s[s.modality == "BM"][["x", "y"]].itertuples(index=False, name=None))
    with plt.rc_context({"font.size": 5.5}):
        fig, axes = plt.subplots(1, 2, figsize=(W * MM * 0.8, (22 + 7.5 * len(pairs)) * MM), sharey=True, constrained_layout=True)
        for ax, modality in zip(axes, ("BM", "PB")):
            for i, (x, y) in enumerate(pairs):
                g = d[(d.modality == modality) & (d.x == x) & (d.y == y) & (d.padj_all < 0.05)].dropna(subset=["delta_adj"])
                r = s[(s.modality == modality) & (s.x == x) & (s.y == y)].iloc[0]
                if len(g) < 3:
                    ax.text(2, i, f"{len(g)} significant features", va="center", fontsize=4.5, color="0.5")
                    continue
                ratio = (g.delta_adj / g.delta_all) * 100
                q1, med, q3 = np.percentile(ratio, [25, 50, 75])
                ax.barh(i, med, color=vcol(x), alpha=0.75, height=0.6)
                ax.plot([q1, q3], [i, i], color="k", lw=0.8)
                ax.text(max(q3, med) + 4, i, f"n = {len(g)}", va="center", fontsize=4.8)
            ax.axvline(100, color="0.4", lw=0.5, ls="--")
            ax.axvline(0, color="k", lw=0.5)
            ax.set_xlim(-10, 175)
            ax.set_ylim(len(pairs) - 0.4, -0.6)
            ax.set_xlabel("Effect remaining after adjusting (%)")
            ax.set_title(modality, fontsize=6, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_yticks(range(len(pairs)), [f"{vname(x)}  |  {vname(y)}" for x, y in pairs])
        for t, (x, _) in zip(axes[0].get_yticklabels(), pairs):
            t.set_color(vcol(x))
    return fig
