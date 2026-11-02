"""Supplementary variable cards: one page per target.

Groups: Neg = negative samples, Pos = positive samples (all cells), Pos top = the model's top-attention cells of the
positive samples (attention-mass threshold 0.5, as the Fig 4/5 model-focus test). Neg / Pos values = cohort sample
features (as the Fig 4/5 association test); Pos top values = top cells re-aggregated per sample.
Points: per-sample outcome of the modality's model (FP among Neg, FN among Pos).
Brackets / stars: Neg vs Pos FDR (association) and Pos vs Pos top FDR (model focus), from feature_effects.csv.
Pos top without a bracket = not tested (< 10 samples with top-cell values; n shown), as the grey × in Fig 4/5.
Inputs: derived/fig4_fingerprints (scripts/extract_fig4_fingerprint_results.py), derived/error_analysis (outcomes).

Card = A differentials (BM, PB) · B framed groups: box plots of the associated dysplasia / morphometry features that are
model focus (dot in Fig 4/5: further enriched in the top-attention cells), one frame per measure × direction (higher / lower
in Pos), panels BM then PB in lineage order, with a strip of example cells (4 typical top-attention cells of Pos samples +
1 striking cell, cells.py) with morphometry sliders; long cards continue on a second page. Morphometry = the Fig 4/5 E measures.
CARDS[target] = {"extra": [frame, ...]} adds hand-picked frames (e.g. a single associated feature that completes the story),
{"frames": [...]} replaces B; frame = {"panels": [(mod, col), ...]}; default frames whose panels are all in an extra frame are
dropped. make(target) returns the list of pages.
"""
from functools import cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.data_loading import load_cohort
from . import cells, config as C, errors
from .fig4 import STATS, effects
from .style import (MM, W, GROUP_COL, CT_SHORT, DIFF_ROWS, MORPH_CT, FEAT_NAME, OUT_COL, vname, vcol, ctname, lineage_col,
                    axes_mm)

GROUPS = ["Neg", "Pos", "Pos top"]
GROUP_LABEL = {"Neg": "Neg.", "Pos": "Pos.", "Pos top": "Pos.\n(top cells)"}
CORRECT = "#7a7a7a"
MEASURES = STATS  # as Fig 4/5 E; compactness left out (nucleus compactness ~ solidity, rho 0.88-0.96)
MEASURE_NAME = {"entire_area": "cell area", "nucleus_area": "nucleus area", "entire_N:C-ratio": "N:C ratio",
                "nucleus_solidity": "nucleus solidity", "nucleus_eccentricity": "nucleus eccentricity",
                "nucleus_compactness": "nucleus compactness", "entire_compactness": "cell compactness"}
SHOWN = ("enriched",)  # model-use categories shown in C
MAX_FRAMES, MAX_PANELS = 8, 4

# targets with a card (as in 11_cytomorphology_analysis.ipynb); IDH2, FLT3-ITD, NPM1 + FLT3-ITD and monosomal karyotype
# are covered by IDH mutations, FLT3 mutations, NPM1 and complex karyotype
CARD_VARS = ["PML-RARA", "CBFB-MYH11", "RUNX1-RUNX1T1_OR_t_8_21", "IDH_mutations", "CpG_mutations", "NPM1",
             "FLT3_mutations", "KIT", "RTK_RAS_mutations", "complex_karyotype", "monosomy5_del_5q", "TP53", "STAG2"]
CARDS = {  # target -> {"extra": [frame, ...]} and / or {"frames": [frame, ...]} (replaces frames())
    "NPM1": {"extra": [{"panels": [("BM", "Blasts-cup_proportion"), ("PB", "Blasts-cup_proportion")]}]},  # cup-shaped blasts
}


# ── data ─────────────────────────────────────────────────────────────────────
@cache
def _top():
    return pd.read_parquet(C.FINGERPRINTS / "sample_features_top.parquet")


@cache
def _cohort(mod):
    return load_cohort(mod)


def _effect(var, mod, col):
    e = effects()
    r = e[(e.variable == var) & (e.modality == mod) & (e.feature_col == col)]
    return r.iloc[0] if len(r) else None


def groups(var, mod, col):
    """{group: (values, outcomes)}; outcome = TN/FP (Neg) or TP/FN (Pos, Pos top), NaN if not predicted."""
    df = _cohort(mod)
    o = errors.samples()
    o = o[(o.modality == mod) & (o.variable == var)][["database_id", "album_id", "outcome"]]
    top = _top()
    top = top[(top.modality == mod) & (top.variable == var)]
    out = {}
    for g, d in (("Neg", df[df[var] == 0]), ("Pos", df[df[var] == 1]), ("Pos top", top)):
        if col not in d:
            out[g] = (np.array([]), np.array([]))
            continue
        d = d[["database_id", "album_id", col]].merge(o, how="left").dropna(subset=[col])
        out[g] = (d[col].to_numpy(float), d.outcome.to_numpy(object))
    return out


def stars(p):
    return "" if not p < 0.05 else "***" if p < 0.001 else "**" if p < 0.01 else "*"


def frames(var):
    """Default C: [{"key", "panels", "images"}] — associated features shown per SHOWN, grouped by measure × direction.

    Frames with a model-focus feature first, then by strongest effect; ≤ MAX_PANELS per frame (model focus, then |δ|).
    """
    e = effects()
    d = e[(e.variable == var) & e.associated & e.model_use.isin(SHOWN)
          & ((e.feature_type == "dysplasia") | ((e.feature_type == "cell_statistic") & e.feature.isin(MEASURES)))].copy()
    if d.empty:
        return []
    d["focus"] = d.model_use.eq("enriched")
    d["strength"] = d.delta_pos_vs_neg.abs()
    d["up"] = d.delta_pos_vs_neg > 0
    lineage = {c: i for i, c in enumerate(MORPH_CT["BM"] + ["Band_neutrophils", "Segmented_neutrophils"])}
    out = []
    for (feat, up), g in d.groupby(["feature", "up"]):
        g = g.sort_values(["focus", "strength"], ascending=False).head(MAX_PANELS)
        g = g.assign(o=g.cell_type.map(lineage).fillna(99)).sort_values(["modality", "o"])
        out.append(dict(key=(g.focus.any(), g.strength.max()), panels=list(zip(g.modality, g.feature_col)),
                        images=(g.feature_type == "dysplasia").all()))
    return sorted(out, key=lambda f: f["key"], reverse=True)[:MAX_FRAMES]


# ── panels ───────────────────────────────────────────────────────────────────
def _points(ax, pos, vals, outs, horizontal, rng, width=0.28):
    jit = pos + rng.uniform(-width, width, len(vals))
    col = np.array([OUT_COL[o] if o in ("FP", "FN") else CORRECT for o in outs])
    err = np.isin(outs, ["FP", "FN"])
    for m, s, z in ((~err, 0.6, 3), (err, 1.4, 4)):  # errors drawn larger, on top
        xy = (vals[m], jit[m]) if horizontal else (jit[m], vals[m])
        ax.scatter(*xy, s=s, c=col[m], lw=0, zorder=z, alpha=0.8 if z == 3 else 1, rasterized=True)


def _boxes(ax, pos, data, horizontal, width=0.7, min_n=1):
    for p, (g, vals) in zip(pos, data):
        if len(vals) < min_n:
            continue
        ax.boxplot(vals, positions=[p], widths=width, orientation="horizontal" if horizontal else "vertical",
                   showfliers=False, patch_artist=True, zorder=2,
                   boxprops=dict(fc=GROUP_COL[g], ec="#333", lw=0.4), medianprops=dict(color="#111", lw=0.6),
                   whiskerprops=dict(color="#333", lw=0.4), capprops=dict(color="#333", lw=0.4))


def differentials(ax, var, mod):
    """Horizontal Neg / Pos / Pos top boxes per cell type; stars = Neg vs Pos FDR, dot = enriched in top cells."""
    rng = np.random.default_rng(0)
    cts = DIFF_ROWS[mod]
    labels = []
    for i, ct in enumerate(cts):
        col = f"{ct}_proportion"
        gr = groups(var, mod, col)
        pos = i + np.array([-0.27, 0, 0.27])
        _boxes(ax, pos, [(g, gr[g][0]) for g in GROUPS], True, width=0.24)
        for p, g in zip(pos, GROUPS):
            _points(ax, p, *gr[g], True, rng, width=0.08)
        r = _effect(var, mod, col)
        s = stars(r.padj_pos_vs_neg) if r is not None else ""
        labels.append(f"{ctname(ct)} {s}".strip() + (" •" if r is not None and s and r.model_use == "enriched" else ""))
    ax.set_yticks(range(len(cts))); ax.set_yticklabels(labels)
    ax.set_ylim(len(cts) - 0.5, -0.5); ax.set_xlim(left=0)
    ax.set_xlabel("% of cells"); ax.tick_params(axis="y", length=0, pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{mod} cell differentials", fontweight="bold", fontsize=6)
    for i, ct in enumerate(cts):
        ax.add_patch(plt.Rectangle((-0.035, i - 0.45), 0.012, 0.9, transform=ax.get_yaxis_transform(), fc=lineage_col(ct),
                                   ec="none", clip_on=False))


def feature_label(mod, col):
    r = effects()
    r = r[r.feature_col == col].iloc[0]
    ct = ctname(r.cell_type)
    ct = (CT_SHORT.get(r.cell_type, ct) if len(ct) > 12 else ct).lower()  # long names abbreviated to fit the panel
    if r.feature_type == "dysplasia":
        return f"{mod}: {FEAT_NAME.get(r.feature, r.feature)}\n{ct}"
    if r.feature_type == "cell_statistic":
        return f"{mod}: {ct}\n{MEASURE_NAME.get(r.feature, r.feature)}"
    return f"{mod}: {ct}"


def _bracket(ax, x0, x1, y, p, h):
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color="#333", lw=0.4, clip_on=False)
    ax.text((x0 + x1) / 2, y, stars(p) or "n.s.", ha="center", va="bottom", fontsize=5 if stars(p) else 4.3)


def feature_box(ax, var, mod, col):
    """Neg / Pos (/ Pos top if any data) for one feature; brackets: Neg vs Pos, Pos vs Pos top (if tested)."""
    rng = np.random.default_rng(0)
    k, unit = C.to_um(col, mod) if not col.endswith("_proportion") else (1.0, "")
    gr = {g: (v * k, o) for g, (v, o) in groups(var, mod, col).items()}
    gs = [g for g in GROUPS if g != "Pos top" or len(gr[g][0])]
    _boxes(ax, range(len(gs)), [(g, gr[g][0]) for g in gs], False, width=0.6, min_n=5)  # < 5 samples: points only
    for p, g in enumerate(gs):
        _points(ax, p, *gr[g], False, rng, width=0.2)
    allv = np.concatenate([gr[g][0] for g in gs])
    lo, hi = np.nanpercentile(allv, [1, 99]) if len(allv) else (0, 1)  # extreme points clipped
    span = hi - lo or 1
    ax.set_ylim(lo - 0.04 * span, hi + 0.36 * span)
    r = _effect(var, mod, col)
    if r is not None:
        _bracket(ax, 0, 1, hi + 0.10 * span, r.padj_pos_vs_neg, 0.03 * span)
        if len(gs) == 3 and not np.isnan(r.padj_top_vs_pos):
            _bracket(ax, 1, 2, hi + 0.24 * span, r.padj_top_vs_pos, 0.03 * span)
        elif len(gs) == 3:
            ax.text(2, hi + 0.12 * span, f"n = {len(gr['Pos top'][0])}\nnot tested", ha="center", va="bottom",
                    fontsize=4.2, color="#777", linespacing=1.1)
    ax.set_xticks(range(len(gs))); ax.set_xticklabels([GROUP_LABEL[g] for g in gs])
    ax.set_xlim(-0.6, 2.6)
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(feature_label(mod, col), fontweight="bold", fontsize=5.3)
    if col.endswith("_proportion"):
        ax.set_ylabel("% of cells")
    elif unit:
        ax.set_ylabel(unit + (" (px size unconfirmed)" if mod in C.UM_PER_PX_UNCONFIRMED else ""))


# ── card: differentials + frames with example cells ──────────────────────────
IMG, BOX_W, BOX_GAP, PAGE_H, N_TYPICAL = 18, 19, 10, 245, 4  # cell image, box panel, gap (mm), page height, typical cells
CELL_W = IMG + 2


def _select(var, frs):
    """Per frame, the example cells: N_TYPICAL typical top-attention cells of Pos samples (cell types of the frame's panels
    in turn, one per sample, TP first) and 1 striking cell; [(frame, [(cell, effects row, cells of the type, type, measures,
    kind)])]."""
    e = effects()
    used, blocks = set(), []
    for f in frs:
        rows = [e[(e.variable == var) & (e.modality == m) & (e.feature_col == col)].iloc[0] for m, col in f["panels"]]
        pools = []
        for r in rows:
            c, ct, flag = cells.cell_pool(var, r)
            pools.append((r, c, ct, flag, cells.measures(var, r.modality, ct, None if flag else r.feature)))
        strip = []
        for k in range(N_TYPICAL):
            r, c, ct, flag, ms = pools[k % len(pools)]
            t = cells.typical(cells.candidates(c, ms, flag, used), ms, 1)
            used |= set(t.key); strip += [(t.iloc[0], r, c, ct, ms, "typical")] if len(t) else []
        r, c, ct, flag, ms = max(pools, key=lambda p: abs(np.nan_to_num(p[0].delta_top_vs_pos)))
        s = cells.striking(cells.candidates(c, ms, flag, used), ms, c)
        used |= set(s.key); strip += [(s.iloc[0], r, c, ct, ms, "striking")] if len(s) else []
        blocks.append((f, strip))
    return blocks


def _frame_geo(f):
    """(width of the box panels, strip below the boxes?, frame height) in mm."""
    bw = len(f["panels"]) * (BOX_W + BOX_GAP)
    below = 12 + bw + (N_TYPICAL + 1) * CELL_W > W - 2
    strip_h = 2 + IMG + 0.7 + 3 * 1.9 + 1.5
    return bw, below, (31 + strip_h if below else max(31, 3 + strip_h + 1))


def _draw_frame(fig, H, y, var, f, strip):
    bw, below, h = _frame_geo(f)
    fr = axes_mm(fig, 2, y, W - 4, h, H); fr.set_xticks([]); fr.set_yticks([]); fr.patch.set_alpha(0)
    for sp in fr.spines.values():
        sp.set_color("#888"); sp.set_linewidth(0.6)
    for j, (mod, col) in enumerate(f["panels"]):
        feature_box(axes_mm(fig, 12 + j * (BOX_W + BOX_GAP), y + 7, BOX_W, 18, H), var, mod, col)
    x, yc = (8, y + 31) if below else (12 + bw, y + 3)
    for cell, r, c, ct, ms, kind in strip:
        lab = f"{r.modality} {ct.replace('_', ' ').lower()}"
        lab, col = (f"{lab} · striking", "#b2182b") if kind == "striking" else (lab, "#555")
        cells.cell_block(fig, x, yc, IMG, H, cell, c, ms, cells.tag_flags(var, r.modality, ct), lab, col, mod=r.modality)
        x += CELL_W
    return h


def _legend(fig, H, frames=True):
    lg = axes_mm(fig, 4, H - 11, W - 8, 9, H); lg.axis("off")
    txt = ("Boxes: Neg / Pos / top-attention cells of Pos samples, points = samples (red / orange = FP / FN); stars: Neg vs Pos and "
           "Pos vs Pos top FDR ('not tested' = < 10 samples with top-cell values;\n< 5 samples: points only); • = enriched in "
           "top-attention cells.")
    if frames:
        txt += (f"\nCells (one scale; edge cells skipped): {N_TYPICAL} typical top-attention cells of Pos samples, one per sample, TP first | "
                "1 striking (red label); chips = classifier flags of the target's associated\ndysplasia features. Sliders (per cell): ▼ = "
                "this cell; bars = middle 50% of Neg (blue) / Pos (orange) cells of the type; areas in µm² (PB pixel size assumed equal to BM, not yet confirmed).")
    lg.text(0.5, 0.5, txt, ha="center", va="center", fontsize=4.1, color="#444", transform=lg.transAxes, linespacing=1.3)


def make(var):
    """List of figures (pages): page 1 = title + A (cell differentials, BM and PB) + B frames; more pages if long.

    B = the associated dysplasia / morphometry features that are model focus (frames()), one frame per measure × direction:
    box plots (all panels of the frame) + a strip of example cells with sliders (cells.py)."""
    spec = CARDS.get(var, {})
    extra = spec.get("extra", [])
    shown = {p for f in extra for p in f["panels"]}
    blocks = _select(var, [f for f in spec.get("frames", frames(var)) if not set(f["panels"]) <= shown] + extra)
    sel = [(r.modality, cell.to_frame().T) for _, strip in blocks for cell, r, *_ in strip]
    if sel:
        cells.write_list(sel, f"card_{var}")
    pages, y, cur = [], 12 + 58, []
    for f, strip in blocks:  # paginate
        h = _frame_geo(f)[2]
        if y + h + 12 > PAGE_H and cur:
            pages.append(cur); cur, y = [], 10
        cur.append((f, strip)); y += h + 3
    pages.append(cur)
    figs = []
    for p, page in enumerate(pages):
        H = (12 + 58 if p == 0 else 10) + sum(_frame_geo(f)[2] + 3 for f, _ in page) + (14 if not page else 0) + 14
        fig = plt.figure(figsize=(W * MM, H * MM))
        title = vname(var) + (f" (continued, {p + 1}/{len(pages)})" if p else f"  ({p + 1}/{len(pages)})" if len(pages) > 1 else "")
        fig.text(0.5, 1 - 2 / H, title, ha="center", va="top", fontsize=8, fontweight="bold", color=vcol(var))
        y = 10
        if p == 0:
            fig.text(1 / W, 1 - 12 / H, "A", fontsize=9, fontweight="bold", va="top")
            differentials(axes_mm(fig, 32, 16, 52, 46, H), var, "BM")
            differentials(axes_mm(fig, 120, 16, 52, 46, H), var, "PB")
            y = 72
            if page:
                fig.text(1 / W, 1 - 70 / H, "B", fontsize=9, fontweight="bold", va="top")
            else:
                fig.text(0.5, 1 - 76 / H, "no associated dysplasia / morphometry feature is enriched in the top-attention cells",
                         ha="center", fontsize=5, color="#777")
        for f, strip in page:
            y += _draw_frame(fig, H, y, var, f, strip) + 3
        _legend(fig, H, bool(blocks))
        figs.append(fig)
    return figs
