"""Fig 4/5 A: UMAP of sample-level morphology with genetic-abnormality centroids.

Adapted from 11_cytomorphology_analysis.ipynb, section "UMAP of morphology features": features = cell differentials,
morphometry and dysplasia associated with ≥ 1 target in feature_effects.csv (same statistics as panels B, C, E:
rank-sum, BH-FDR < 0.05 per target × modality × feature type, ≥ 10 samples per group); median imputation,
standardisation, UMAP (random_state 42). Centroid = mean embedding of the samples positive for a target.
"""
from functools import cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from core.data_loading import load_cohort, feature_lists
from . import config as C
from .style import GROUPS, VARS, vcol

LABELS = {"monosomy5_del_5q": "Monosomy 5 /\ndel(5q)", "IDH_mutations": "IDH1/2", "RUNX1-RUNX1T1_OR_t_8_21": "RUNX1-RUNX1T1",
          "NPM1_AND_FLT3_ITD": "NPM1 & FLT3-ITD", "complex_karyotype": "Complex karyotype",
          "monosomal_karyotype": "Monosomal karyotype", "RTK_RAS_mutations": "RTK/RAS", "CpG_mutations": "CpG",
          "FLT3_mutations": "FLT3-ITD/TKD", "FLT3_ITD": "FLT3-ITD"}
GROUP_LABELS = {"Signaling / RTK-RAS": "Signaling/ RTK/RAS"}


def selected_features(df, mod):
    """Features significant for ≥ 1 target, in notebook column order (differentials, morphometry, dysplasia)."""
    fl = feature_lists(df, mod)
    r = pd.read_csv(C.FINGERPRINTS / "feature_effects.csv")
    sig = set(r.loc[(r.modality == mod) & r.variable.isin(VARS) & r.associated, "feature_col"])
    return [c for c in fl["cell_differential"] + fl["cell_statistic"] + fl["dysplasia"] if c in sig]


@cache
def embedding(mod):
    """(cohort, 2-D UMAP embedding, feature columns used)."""
    import umap  # slow import (numba)
    df = load_cohort(mod)
    cols = selected_features(df, mod)
    X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df[cols].values))
    return df, umap.UMAP(n_components=2, random_state=42).fit_transform(X), cols


def draw(ax, mod, fontsize=5, marker_size=30):
    """Samples (grey), target centroids (group colour, black edge) with repelled labels, group legend below the axes."""
    from adjustText import adjust_text
    df, emb, _ = embedding(mod)
    ax.scatter(emb[:, 0], emb[:, 1], c="lightgray", s=marker_size * 0.15, alpha=0.6, linewidths=0, rasterized=True)
    texts, cx_, cy_ = [], [], []
    for v in VARS:
        mask = df[v].fillna(0).astype(bool).to_numpy()
        if not mask.any():
            continue
        cx, cy = emb[mask, 0].mean(), emb[mask, 1].mean()
        ax.scatter(cx, cy, s=marker_size, color=vcol(v), zorder=5, edgecolors="black", linewidths=0.4)
        texts.append(ax.text(cx, cy, LABELS.get(v, v), fontsize=fontsize, color=vcol(v), fontweight="bold", zorder=6,
                             path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]))  # readable over markers
        cx_.append(cx); cy_.append(cy)
    pad = 0.06 * np.ptp(emb[:, 1])  # headroom so labels stay inside the panel
    ax.set_xlim(emb[:, 0].min(), emb[:, 0].max()); ax.set_ylim(emb[:, 1].min() - pad, emb[:, 1].max() + pad)
    adjust_text(texts, x=cx_, y=cy_, ax=ax, arrowprops=dict(arrowstyle="-", color="black", lw=0.5), expand=(2.0, 2.0),
                force_text=(2.5, 2.5))  # labels repelled from each other and from the centroid markers
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor="black",
                          markeredgewidth=0.4, markersize=marker_size * 0.15, label=GROUP_LABELS.get(g, g))
               for g, c, _ in GROUPS]
    ax.legend(handles=handles, fontsize=fontsize, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0), borderaxespad=0,
              ncol=4)
    ax.set_axis_off()
