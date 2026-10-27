"""VAF of detected (TP) and missed (FN) positives, one panel per target with a VAF.

vaf_by_outcome(modality)   box + points of the VAF for FN and TP; AUROC of the VAF for TP vs FN (95% CI) and the FDR-adjusted p.
Inputs: derived/error_analysis/vaf_by_outcome.csv and vaf_outcome_stats.csv (scripts/extract_vaf_by_outcome.py).
"""
from functools import cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config as C
from .style import MM, W, VARS, OUT_COL, vname, vcol

NCOL = 4


def _p(p):
    return "FDR p < 0.001" if p < 0.001 else f"FDR p = {p:.3f}"


@cache
def _tables():
    return (pd.read_csv(C.ERRORS / "vaf_by_outcome.csv"), pd.read_csv(C.ERRORS / "vaf_outcome_stats.csv"))


def vaf_by_outcome(modality):
    samples, stats = _tables()
    st = stats[stats.modality == modality].set_index("variable")
    variables = [v for v in VARS if v in st.index]
    nrow = -(-len(variables) // NCOL)
    with plt.rc_context({"font.size": 5.5}):
        fig, axes = plt.subplots(nrow, NCOL, figsize=(W * MM, 44 * nrow * MM), squeeze=False, constrained_layout=True)
        rng = np.random.default_rng(0)
        for ax, v in zip(axes.flat, variables):
            d = samples[(samples.modality == modality) & (samples.variable == v)]
            for x, o in enumerate(["FN", "TP"]):
                y = d.loc[d.outcome == o, "vaf"].to_numpy()
                if len(y):
                    ax.boxplot(y, positions=[x], widths=0.5, showfliers=False, medianprops=dict(color="k", lw=0.8),
                               boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
                    ax.scatter(x + rng.uniform(-0.17, 0.17, len(y)), y, s=3, color=OUT_COL[o], alpha=0.7, lw=0, zorder=3)
            r = st.loc[v]
            txt = f"AUROC {r.auroc:.2f} ({r.auroc_low:.2f}–{r.auroc_high:.2f})\n{_p(r.padj)}" if pd.notna(r.get("auroc")) else "n too small"
            ax.text(0.5, 1.0, txt, transform=ax.transAxes, ha="center", va="bottom", fontsize=4.6)
            ax.set_title(vname(v), color=vcol(v), fontsize=6, fontweight="bold", pad=17)
            ax.set_xticks([0, 1], [f"FN\n(n={int(r.n_fn)})", f"TP\n(n={int(r.n_tp)})"])
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylabel("VAF (%)")
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes.flat[len(variables):]:
            ax.axis("off")
    return fig
