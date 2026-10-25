"""Morphometry of dysplastic vs non-dysplastic cells of the same type, within each sample (BM, PB).

Data: scripts/extract_morphometry_dysplasia_correlation.py → derived/morph_dysplasia/morph_dysplasia_effects.csv.
Per sample with ≥ 5 flagged and ≥ 5 unflagged cells: Cliff's δ flagged vs unflagged; cell = median δ across samples,
stars = Wilcoxon signed-rank of the per-sample δs (BH-FDR per modality), shown if |δ| ≥ 0.1. Rows need ≥ 10
samples. Measures as Fig 4/5 E.
Redesign of figures/mockups/v3/morph_dysplasia_heatmap.py on the shared δ scale (±1) of Fig 4/5.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config as C
from .fig4 import STATS, STAT_NAME
from .style import MM, W, cmap, norm, ctname, lineage_col, axes_mm

FLAG = {"apl_hypergranulated": "hypergranular", "apl_bilobed": "bilobed nucleus", "apl_auer": "Auer rods",
        "vacuolated": "vacuolated", "mds_asynchrony": "N:C asynchrony", "mds_dysmorphic": "dysmorphic",
        "mds_multinucleated": "multinucleated", "mgk_multinucleated": "separated nuclei",
        "mgk_hypolobated": "hypolobated", "neut_gran_hypo": "hypogranular", "neut_gran_hyper": "hypergranular",
        "neut_segm_hypo": "hyposegmented", "neut_segm_hyper": "hypersegmented"}
CELL = 3.2  # mm per heatmap cell
MIN_SAMPLES = 10
MIN_DELTA = 0.1  # stars only from |δ| ≥ 0.1 (with hundreds of samples tiny shifts are significant)


def effects():
    r = pd.read_csv(C.DERIVED / "morph_dysplasia/morph_dysplasia_effects.csv")
    return r[(r.n_samples >= MIN_SAMPLES) & r.feature.isin(STATS)]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def heatmap(fig, H, x, y, mod):
    d = effects()
    d = d[d.modality == mod]
    rows = list(dict.fromkeys(zip(d.cell_type, d.flag)))
    ax = axes_mm(fig, x, y, len(STATS) * CELL, len(rows) * CELL, H)
    for i, (ct, fl) in enumerate(rows):
        q = d[(d.cell_type == ct) & (d.flag == fl)].set_index("feature")
        for j, f in enumerate(STATS):
            v = q.at[f, "median_delta"]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=cmap(norm(v)), ec="white", lw=0.4))
            if abs(v) >= MIN_DELTA:
                ax.text(j + 0.5, i + 0.62, stars(q.at[f, "padj"]), ha="center", va="center", fontsize=4.2,
                        color="white" if abs(v) > 0.6 else "#111")
        ax.text(len(STATS) + 0.2, i + 0.5, f"{q.n_samples.iat[0]}", fontsize=3.8, va="center", color="#666")
        ax.add_patch(plt.Rectangle((-0.55, i + 0.15), 0.35, 0.7, fc=lineage_col(ct), ec="none", clip_on=False))
    ax.set_xlim(0, len(STATS)); ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(STATS)) + 0.5, [STAT_NAME[s] for s in STATS], rotation=60, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(np.arange(len(rows)) + 0.5, [f"{ctname(ct)} – {FLAG.get(fl, fl)}" for ct, fl in rows])
    ax.tick_params(length=0, pad=1); ax.tick_params(axis="y", pad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Bone marrow" if mod == "BM" else "Peripheral blood", fontsize=5.8, fontweight="bold", pad=3)
    return len(rows) * CELL


def make():
    n = effects().drop_duplicates(["modality", "cell_type", "flag"]).modality.value_counts()
    H = 8 + n.max() * CELL + 21
    fig = plt.figure(figsize=(W * MM, H * MM))
    heatmap(fig, H, 48, 8, "BM")
    heatmap(fig, H, 136, 8, "PB")
    yb = 8 + n["PB"] * CELL + 22
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axes_mm(fig, 112, yb, 34, 1.6, H),
                      orientation="horizontal")
    cb.set_label("Cliff's δ, dysplastic vs other cells (median across samples)"); cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.5)
    fig.text(112 / W, 1 - (yb + 9) / H, "* FDR < 0.05, ** < 0.01, *** < 0.001 (shown if |δ| ≥ 0.1)\n"
             "n = samples with ≥ 5 dysplastic and ≥ 5 other cells", fontsize=4.4, color="#444", va="top",
             linespacing=1.4)
    return fig
