"""Cohort genetics: oncoprint of the targets and their pairwise co-occurrence.

Port of 11_cytomorphology_analysis.ipynb, "Figure 2 - Study cohort characteristics" (oncoprint, plot_cooccurrence).
Patients = subjects with a BM or PB sample in the modelling cohort; labels from the BM label file (one row per patient).
Co-occurrence (row, column) = % of patients positive for the row target that are also positive for the column target,
among those with the column target known.
"""
from functools import cache

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from core.data_loading import load_cohort, load_labels
from .style import MM, W, VARS, vname, vcol, axes_mm

POS, NEG, MISSING = "#174E6D", "#ffffff", "#C6C6C6"
COOC_CMAP = plt.get_cmap("Blues")


@cache
def labels():
    """Patients × targets (1 / 0 / NaN)."""
    subjects = set(load_cohort("BM").subject_id) | set(load_cohort("PB").subject_id)
    L = load_labels("BM")
    return L[L.subject_id.isin(subjects)].set_index("subject_id")[VARS].astype(float)


def cooccurrence():
    """(% matrix, count matrix): row target positive and column target positive / row positive with column known."""
    L = labels()
    pos, known = (L == 1).astype(float), L.notna().astype(float)
    both = pos.T @ pos
    return both / (pos.T @ known) * 100, both


def oncoprint(ax, axb, order):
    L = labels()
    m = L[order].sort_values(by=order, ascending=False)
    img = np.empty((len(order), len(m), 3))
    for val, c in ((1, POS), (0, NEG)):
        img[m.T.values == val] = matplotlib.colors.to_rgb(c)
    img[m.T.isna().values] = matplotlib.colors.to_rgb(MISSING)
    ax.imshow(img, aspect="auto", interpolation="none")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([vname(v) for v in order])
    for tl, v in zip(ax.get_yticklabels(), order):
        tl.set_color(vcol(v))
    ax.set_xticks([]); ax.set_xlabel(f"Patients (n = {len(m)})")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    left = np.zeros(len(order))
    for val, c in ((1, POS), (0, NEG), (np.nan, MISSING)):
        n = (m.isna() if np.isnan(val) else m == val).sum().values
        axb.barh(range(len(order)), n, left=left, color=c, height=0.8, ec="0.6" if val == 0 else "none", lw=0.2)
        left += n
    for i, v in enumerate(order):
        n1, n = (m[v] == 1).sum(), m[v].notna().sum()
        axb.text(n1 + 6, i, f"{n1 / n:.0%}", va="center", fontsize=4.5)
    axb.set_ylim(len(order) - 0.5, -0.5); axb.set_xlim(0, len(m))
    axb.set_xticks([0, len(m)]); axb.tick_params(labelleft=False, left=False)
    axb.spines[["top", "right", "left"]].set_visible(False)
    axb.set_xlabel("Patients")


def heatmap(ax, cax):
    pct, n = cooccurrence()
    ax.imshow(pct.values, cmap=COOC_CMAP, vmin=0, vmax=100)
    for i in range(len(VARS)):
        for j in range(len(VARS)):
            ax.text(j, i, f"{pct.iat[i, j]:.0f}", ha="center", va="center", fontsize=4,
                    color="white" if pct.iat[i, j] > 60 else "black")
    ax.set_xticks(range(len(VARS))); ax.set_yticks(range(len(VARS)))
    ax.set_xticklabels([vname(v) for v in VARS], rotation=60, ha="right", rotation_mode="anchor")
    ax.set_yticklabels([vname(v) for v in VARS])
    for tl, v in zip(ax.get_xticklabels() + ax.get_yticklabels(), VARS + VARS):
        tl.set_color(vcol(v))
    ax.set_xticks(np.arange(len(VARS) + 1) - .5, minor=True); ax.set_yticks(np.arange(len(VARS) + 1) - .5, minor=True)
    ax.grid(which="minor", color="white", lw=0.5); ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_ylabel("Patients positive for"); ax.set_xlabel("% also positive for")
    cb = plt.colorbar(plt.cm.ScalarMappable(matplotlib.colors.Normalize(0, 100), COOC_CMAP), cax=cax)
    cb.set_label("Co-occurrence (%)"); cb.outline.set_visible(False); cb.ax.tick_params(length=1.5)


def make():
    H = 210
    fig = plt.figure(figsize=(W * MM, H * MM))
    order = labels().eq(1).sum().sort_values(ascending=False).index.tolist()
    oncoprint(axes_mm(fig, 30, 6, 112, 52, H), axes_mm(fig, 144, 6, 28, 52, H), order)
    h = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="0.6", lw=0.3, label=l)
         for c, l in ((POS, "Mutated / abnormal"), (NEG, "Wild-type / normal"), (MISSING, "Missing"))]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(86 / W, 1 - 66 / H), ncol=3, frameon=False)
    s = 100  # heatmap side (mm), square cells
    heatmap(axes_mm(fig, 30, 78, s, s, H), axes_mm(fig, 30 + s + 4, 78 + s / 2 - 15, 2.5, 30, H))
    for lab, y in (("A", 3), ("B", 75)):
        fig.text(2 / W, 1 - y / H, lab, fontsize=8, fontweight="bold", va="top")
    return fig
