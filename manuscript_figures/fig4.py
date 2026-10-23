"""Fig 4 (BM) / Fig 5 (PB): cell composition, dysplasia, model attention and morphometry, rows = targets.

A  UMAP of sample morphology with mutation centroids (umap_centroids.py) + legend
B  cell proportions (balloons) | C  dysplasia (balloons) | D  attention enrichment | N_eff/N bars
E  morphometry: one heatmap per measure (columns = cell types with ≥ 1 significant cell)

Association = Pos vs Neg Cliff's δ, shown / coloured where FDR < 0.05 (BH per variable × modality × feature type).
Model focus (C, E): dot = feature further enriched in the top-attention cells, grey × = not testable.
Input: derived/fig4_fingerprints (scripts/extract_fig4_fingerprint_results.py).
"""
from functools import cache

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import config as C, umap_centroids
from .style import (MM, W, GROUPS, VARS, LINEAGE_COL, CT_LINEAGE, CT_NAME, CT_SHORT, DIFF_ROWS, MORPH_CT, FEAT_NAME,
                    cmap, norm, vname, vcol, ctname, lineage_col, axes_mm, size_of)

hmap = plt.get_cmap("RdBu_r"); hnorm = matplotlib.colors.TwoSlopeNorm(0, -3, 2)  # attention enrichment (log)
STATS = ["entire_area", "nucleus_area", "entire_N:C-ratio", "nucleus_solidity", "nucleus_eccentricity"]
STAT_NAME = {"entire_area": "cell area", "nucleus_area": "nucleus area", "entire_N:C-ratio": "N:C ratio",
             "nucleus_solidity": "nucleus solidity", "nucleus_eccentricity": "nucleus eccentricity"}
DYS_CT = {"Myeloid_precursors": "Myel. prec.", "Erythroblasts": "Erythrobl.", "Promyelocytes": "Promyel.",
          "Eosinophils": "Mat. eos.", "Neutrophils": "Neutroph.", "Lymphocytes": "Lymphoc.", "Megakaryocytes": "Megakar."}
# dysplasia category colours, as in the 11_cytomorphology_analysis.ipynb balloon plots
DYS_COL = {"Myeloid_precursors_hypergranulated_proportion": "#f8d097", "Myeloid_precursors_bilobed_proportion": "#f8d097",
           "Myeloid_precursors_auer_proportion": "#f8d097",
           "APL-apl_hypergranulated_proportion": "#f8d097", "Blasts-apl_hypergranulated_proportion": "#f8d097",
           "Myelocytes-apl_hypergranulated_proportion": "#f8d097",
           "Erythroblasts_asynchrony_proportion": "#b6a0bc", "Erythroblasts_dysmorphic_proportion": "#b6a0bc",
           "Erythroblasts_multinucleated_proportion": "#b6a0bc",
           "Megakaryocytes_multinucleated_proportion": "#abc2ca", "Megakaryocytes_hypolobated_proportion": "#abc2ca",
           "Megakaryocytes-vacuolated_proportion": "#abc2ca",
           "Neutrophils-neut_gran_normal_proportion": "#a7cbab", "Neutrophils-neut_gran_hypo_proportion": "#a7cbab",
           "Neutrophils-neut_gran_hyper_proportion": "#a7cbab"}
DYS_COL_DEFAULT = "#f9d7c9"  # vacuolated
CT_ORDER = list(CT_LINEAGE)


@cache
def effects():
    return pd.read_csv(C.FINGERPRINTS / "feature_effects.csv")


def load(mod):
    r = effects()
    e = pd.read_csv(C.FINGERPRINTS / "attention_enrichment.csv")
    e = e[e.modality == mod].pivot(index="cell_type", columns="variable", values="log_enrichment")
    s = pd.read_csv(C.FINGERPRINTS / "attention_spread.csv")
    return r[r.modality == mod], e, s[(s.modality == mod) & (s.pred == 1)]


def _block(rows, cols, key):
    """Targets × cols matrices: δ, significant, model focus (enriched), not testable, FDR p."""
    m = np.full((len(VARS), len(cols)), np.nan)
    p = np.full_like(m, np.nan)
    sig, use, noeval = (np.zeros_like(m, bool) for _ in range(3))
    for r in rows.itertuples():
        k = key(r)
        if k in cols and r.variable in VARS:
            i, c = VARS.index(r.variable), cols.index(k)
            m[i, c], sig[i, c], p[i, c] = r.delta_pos_vs_neg, r.padj_pos_vs_neg < 0.05, r.padj_pos_vs_neg
            use[i, c] = sig[i, c] and r.model_use == "enriched"
            noeval[i, c] = sig[i, c] and r.model_use == "not evaluable"
    return m, sig, use, noeval, p


def dysplasia_block(mod):
    """(labels, strip colours, matrices) for dysplasia features significant for ≥ 1 target."""
    eff = effects()
    d = eff[(eff.modality == mod) & (eff.feature_type == "dysplasia") & eff.variable.isin(VARS)]
    hit = d[d.padj_pos_vs_neg < 0.05]
    order = {c: n for n, c in enumerate(MORPH_CT[mod])}
    cols = sorted(hit.feature_col.unique(), key=lambda f: (order.get(hit[hit.feature_col == f].cell_type.iat[0], 99), f))
    first = d.drop_duplicates("feature_col").set_index("feature_col")
    labels = [f"{DYS_CT.get(first.cell_type[f], ctname(first.cell_type[f]))}: {FEAT_NAME.get(first.feature[f], first.feature[f])}"
              for f in cols]
    return labels, [DYS_COL.get(f, DYS_COL_DEFAULT) for f in cols], _block(d, cols, lambda r: r.feature_col)


def morphometry_blocks(mod):
    """[(measure, cell types, matrices)] over cell types with ≥ 1 significant cell in any measure."""
    eff = effects()
    d = eff[(eff.modality == mod) & (eff.feature_type == "cell_statistic") & eff.feature.isin(STATS) & eff.variable.isin(VARS)]
    hit = set(d[d.padj_pos_vs_neg < 0.05].cell_type)
    present = set(d.cell_type)
    cts = [c for c in dict.fromkeys(CT_ORDER + ["Band_neutrophils", "Segmented_neutrophils"]) if c in present and c in hit]
    return [(STAT_NAME[s], cts, _block(d[d.feature == s], cts, lambda r: r.cell_type)) for s in STATS]


def draw_morphometry(fig, mod, left, right, top, bottom, title_y, gap_cells=1.2):
    """E: square-cell heatmaps (rows = targets), tilted cell-type labels on top, colour strips on both axes."""
    blocks = morphometry_blocks(mod)
    fw, fh = fig.get_size_inches()
    ncol, nb = sum(len(c) for _, c, _ in blocks), len(blocks)
    cell = min((top - bottom) * fh / len(VARS), (right - left) * fw / (ncol + gap_cells * (nb - 1)))
    cw, ph = cell / fw, len(VARS) * cell / fh
    gap = min(max(gap_cells * cw, ((right - left) - ncol * cw) / (nb - 1)), 4 * cw)
    x = left
    for j, (t, cts, (m, sig, use, noeval, _)) in enumerate(blocks):
        nc = len(cts)
        ax = fig.add_axes([x, top - ph, nc * cw, ph]); x += nc * cw + gap
        rgba = cmap(norm(np.nan_to_num(m)))
        rgba[~sig] = (1, 1, 1, 1)  # not significant / not tested: white
        ax.imshow(rgba, aspect="equal")
        yy, xx = np.where(use)
        ax.scatter(xx, yy, s=2.8, color="k", lw=0, zorder=4)
        yy, xx = np.where(noeval)
        ax.scatter(xx, yy, marker="x", s=5, color="#8a8a8a", lw=0.45, zorder=4)
        ax.xaxis.tick_top()
        ax.set_xticks(range(nc))
        ax.set_xticklabels([CT_SHORT.get(c, ctname(c)) for c in cts], rotation=60, fontsize=4.2, ha="left",
                           rotation_mode="anchor")
        for k, c in enumerate(cts):
            ax.add_patch(plt.Rectangle((k - 0.5, -1.25), 1, 0.55, fc=lineage_col(c), ec="white", lw=0.2, clip_on=False))
        ax.set_yticks(range(len(VARS)))
        ax.set_yticklabels([vname(v) for v in VARS] if j == 0 else [], fontsize=4.6)
        if j == 0:
            for i, v in enumerate(VARS):
                ax.add_patch(plt.Rectangle((-1.25, i - 0.5), 0.55, 1, fc=vcol(v), ec="none", clip_on=False))
        for tl, v in zip(ax.get_yticklabels(), VARS):
            tl.set_color(vcol(v))
        ax.set_xticks(np.arange(nc + 1) - .5, minor=True); ax.set_yticks(np.arange(len(VARS) + 1) - .5, minor=True)
        ax.grid(which="minor", color="0.88", lw=0.3); ax.tick_params(which="both", length=0, pad=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.text(x - gap - nc * cw / 2, title_y, t.replace(" ", "\n", 1) if nc < 8 and " " in t else t,
                 ha="center", va="top", fontsize=5.2, fontweight="bold")
        i = 0
        for _, _, vs in GROUPS[:-1]:
            i += len(vs); ax.axhline(i - 0.5, color="0.3", lw=0.4)


def _legend(fig, H, x, y, h):
    lg = axes_mm(fig, x, y, 73, h, H); lg.set_xlim(0, 73); lg.set_ylim(h, 0); lg.axis("off")
    lg.text(0, 0.5, "Association, Pos vs Neg (B, C, E)", fontsize=5.3, fontweight="bold", va="top")
    for k in range(50):
        lg.add_patch(Rectangle((k * 0.6, 4.3), 0.6, 1.5, fc=cmap(k / 49), ec="none"))
    lg.text(0, 7.9, "−1", fontsize=4.4, va="center"); lg.text(15, 7.9, "0", fontsize=4.4, ha="center", va="center")
    lg.text(30, 7.9, "+1", fontsize=4.4, ha="right", va="center"); lg.text(31.5, 5.0, "Cliff's δ", fontsize=4.5, va="center")
    for k, (p, t) in enumerate([(0.05, "0.05"), (0.01, "0.01"), (0.001, "<0.001")]):
        lg.scatter(48 + k * 7.5, 5.0, s=size_of(p) ** 2, color="#999", ec="#333", lw=0.25)
        lg.text(48 + k * 7.5, 8.3, t, fontsize=4.1, ha="center", va="center")
    lg.text(48, 1.6, "FDR-adj. p (B, C)", fontsize=4.4, va="center")
    lg.text(0, 10.3, "shown (B, C) or coloured (E) only if FDR < 0.05; white in E = n.s. or not testable", fontsize=4.2,
            va="center", color="#444")
    lg.text(0, 13.5, "Model attention (D)", fontsize=5.3, fontweight="bold", va="top")
    for k in range(50):
        lg.add_patch(Rectangle((k * 0.6, 17.3), 0.6, 1.5, fc=hmap(hnorm(-3 + 5 * k / 49)), ec="none"))
    lg.text(0, 20.9, "−3", fontsize=4.4, va="center"); lg.text(18, 20.9, "0", fontsize=4.4, ha="center", va="center")
    lg.text(30, 20.9, "+2", fontsize=4.4, ha="right", va="center")
    lg.text(31.5, 18.0, "attention enrichment (log)", fontsize=4.5, va="center")
    lg.text(0, 24.5, "bars: attention spread N$_{eff}$/N (predicted positives)", fontsize=4.4, va="center", color="#444")
    lg.text(0, 27.5, "Model focus (C, E)", fontsize=5.3, fontweight="bold", va="top")
    lg.add_patch(Rectangle((0.7, 30.6), 2.6, 2.4, fc=cmap(0.85), ec="none"))  # E: cell with dot
    lg.scatter(2, 31.8, s=2.8, color="k", lw=0)
    lg.scatter(5.6, 31.8, s=20, color=cmap(0.85), ec="#333", lw=0.25, snap=False)  # C: balloon with dot
    lg.scatter(5.6, 31.8, s=2.8, color="k", lw=0, snap=False)
    lg.text(8.5, 31.8, "dot: feature further enriched in the model's top-attention cells\n"
            "(Pos top cells vs all Pos cells; see Supp. variable cards)", fontsize=4.5, va="center", linespacing=1.2)
    lg.add_patch(Rectangle((0.7, 35.4), 2.6, 2.4, fc=cmap(0.85), ec="none"))
    lg.scatter(2, 36.6, s=5, marker="x", color="#8a8a8a", lw=0.5)
    lg.text(5, 36.6, "×: too few top-attention cells / samples\nto test model focus", fontsize=4.5, va="center",
            linespacing=1.2)


def make(mod="BM"):
    r, e, s = load(mod)
    cts = DIFF_ROWS[mod]
    dys_lab, dys_col, (dm, dsig, duse, dnoeval, dp) = dysplasia_block(mod)
    # B | C | D geometry: rows = targets (grouped), square cells, equal column spacing
    rh, gpr = 2.8, 0.8
    cw = rh
    xB0 = 24
    wB = len(cts) * cw
    xY0 = xB0 + wB + 7          # C dysplasia
    wY = len(dys_lab) * cw
    # gap to D: clear the tilted dysplasia labels (≈0.84 mm/char at 4.3 pt, 60° → ×0.5 horizontally) + letter D
    reach = max((k + 0.5) * cw + 0.42 * len(l) for k, l in enumerate(dys_lab)) - wY
    xD0 = xY0 + wY + max(7, reach + 6)  # D attention enrichment
    xN0 = xD0 + len(cts) * cw + 4       # N_eff/N bars
    wN = min(W - xN0 - 2, 20)
    ys = {v: (VARS.index(v) + 0.5) * rh + gpr * gi for gi, (_, _, vs) in enumerate(GROUPS) for v in vs}
    hM = len(VARS) * rh + gpr * (len(GROUPS) - 1)

    yA, hA = 1, 48
    yH, hH = yA + hA + 1, 22
    yM = yH + hH + 1
    yE = yM + hM + 11           # E block titles; tilted labels below, heatmaps from yE + 16
    hE = 16 + 37
    H = yE + hE + 3
    fig = plt.figure(figsize=(W * MM, H * MM))

    def lab(x, y, t):
        fig.text(x / W, 1 - y / H, t, fontsize=9, fontweight="bold", va="top")

    # A + legend
    lab(0, yA, "A")
    umap_centroids.draw(axes_mm(fig, 4, yA, 92, hA - 7, H), mod)  # group legend below the axes
    _legend(fig, H, 104, yA + 3, hA - 3)

    # headers: cell types (B, D) and dysplasia features (C), tilted 60°, colour strip below
    lab(0, yH, "B"); lab(xY0 - 4.5, yH, "C"); lab(xD0 - 4.5, yH, "D")
    hd = axes_mm(fig, 0, yH, W, hH, H); hd.set_xlim(0, W); hd.set_ylim(hH, 0); hd.axis("off")
    for x0 in (xB0, xD0):
        for k, ct in enumerate(cts):
            x = x0 + (k + 0.5) * cw
            hd.add_patch(Rectangle((x - cw / 2, hH - 1.6), cw, 1.1, fc=LINEAGE_COL[CT_LINEAGE.get(ct, "Blasts")], ec="white", lw=0.3))
            hd.text(x, hH - 2.2, CT_NAME.get(ct, ct), rotation=60, ha="left", va="bottom", fontsize=4.3, rotation_mode="anchor")
    for k, (l, c) in enumerate(zip(dys_lab, dys_col)):
        x = xY0 + (k + 0.5) * cw
        hd.add_patch(Rectangle((x - cw / 2, hH - 1.6), cw, 1.1, fc=c, ec="white", lw=0.3))
        hd.text(x, hH - 2.2, l, rotation=60, ha="left", va="bottom", fontsize=4.3, rotation_mode="anchor")
    hd.text(xN0, hH - 0.8, "attention spread\nN$_{eff}$/N", fontsize=4.3, va="bottom", color="#444", linespacing=1.1)

    m = axes_mm(fig, 0, yM, W, hM, H); m.set_xlim(0, W); m.set_ylim(hM, 0); m.axis("off")
    for v, y in ys.items():
        m.text(xB0 - 2.2, y, vname(v), ha="right", va="center", fontsize=4.6, color=vcol(v))
        for x0 in (xB0, xY0, xD0):  # target-group colour strips
            m.add_patch(Rectangle((x0 - 1.7, y - rh / 2), 1.1, rh, fc=vcol(v), ec="none"))
        # B proportions (balloons) and D attention enrichment (tiles)
        m.plot([xB0, xB0 + wB], [y, y], color="#eeeeee", lw=0.3, zorder=0)
        for k, ct in enumerate(cts):
            xb, xd = xB0 + (k + 0.5) * cw, xD0 + k * cw
            m.plot([xb, xb], [y - rh / 2, y + rh / 2], color="#eeeeee", lw=0.3, zorder=0)
            q = r[(r.variable == v) & (r.feature_col == f"{ct}_proportion")]
            if len(q) and q.padj_pos_vs_neg.iloc[0] < 0.05:
                q = q.iloc[0]
                m.scatter(xb, y, s=size_of(q.padj_pos_vs_neg) ** 2, color=cmap(norm(q.delta_pos_vs_neg)),
                          ec="#333", lw=0.25, zorder=3)
            # PB model merges band + segmented neutrophils: one tile spanning both columns
            if ct == "Segmented_neutrophils" and "Band_neutrophils" in cts:
                continue
            ect, ww = ct, cw
            if ct == "Band_neutrophils":
                ect, ww = "Neutrophils", 2 * cw
            if ect in e.index and v in e.columns and not np.isnan(e.loc[ect, v]):
                m.add_patch(Rectangle((xd, y - rh / 2), ww, rh, fc=hmap(hnorm(e.loc[ect, v])), ec="white", lw=0.3))
        # C dysplasia (balloons; dot = model focus, grey × = not testable)
        j = VARS.index(v)
        m.plot([xY0, xY0 + wY], [y, y], color="#eeeeee", lw=0.3, zorder=0)
        for k in range(len(dys_lab)):
            x = xY0 + (k + 0.5) * cw
            m.plot([x, x], [y - rh / 2, y + rh / 2], color="#eeeeee", lw=0.3, zorder=0)
            if not dsig[j, k]:
                continue
            m.scatter(x, y, s=size_of(dp[j, k]) ** 2, color=cmap(norm(dm[j, k])), ec="#333", lw=0.25, zorder=3, snap=False)
            if duse[j, k]:
                m.scatter(x, y, s=2.8, color="k", lw=0, zorder=4, snap=False)
            if dnoeval[j, k]:  # fixed size, may extend past small balloons (balloon size stays true)
                m.scatter(x, y, marker="x", s=7, color="#8a8a8a", lw=0.5, zorder=4, snap=False)
        # N_eff/N: median bar, IQR line
        q = s[s.variable == v].neff_frac
        if len(q):
            lo, md, hi = q.quantile([0.25, 0.5, 0.75])
            m.add_patch(Rectangle((xN0, y - rh * 0.3), md * wN, rh * 0.6, fc="#9e9e9e", ec="none"))
            m.plot([xN0 + lo * wN, xN0 + hi * wN], [y, y], color="#333", lw=0.4)
    m.plot([xN0, xN0], [0, hM], color="#bbb", lw=0.4)
    m.text(xN0, hM + 0.3, "0", fontsize=4, ha="center", va="top"); m.text(xN0 + wN, hM + 0.3, "1", fontsize=4, ha="center", va="top")

    lab(0, yE - 3, "E")
    f = lambda mm: 1 - mm / H  # noqa: E731
    draw_morphometry(fig, mod, left=24 / W, right=168 / W, top=f(yE + 16), bottom=f(yE + hE), title_y=f(yE - 1.5))
    return fig
