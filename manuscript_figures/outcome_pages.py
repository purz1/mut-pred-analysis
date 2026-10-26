"""Supplement (TP/FP pages): top-attention cells of true- and false-positive samples, one page per card target (BM and PB models).

Cells are chosen by attention only: the model's top-attention cells (attention-mass threshold 0.5, as Fig 4/5) of the TP and
FP samples (errors.samples). Random draw (fixed per cell): ≤ PER_SAMPLE cells per sample, ≤ N_CELLS per panel; no quality,
cell-type or morphometry filter (only cells whose crop window lies inside the field image and whose image is available).
Cells are ordered by cell type (lineage order), then inside a type by similarity of the MIL cell features (Ward leaf order,
TP and FP of a modality share one order). Frame and label = cell type; chips = classifier flags of the dysplasia features
associated with the target in that modality; LQ = low-quality score ≥ 0.5; number = sample of the panel (1 = highest predicted
probability; samples() gives the table).
"""
from functools import cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist

from . import cells, errors
from .crops import usable
from .fig4 import effects
from .style import MM, W, DIFF_ROWS, LINEAGE_COL, CT_SHORT, OUT_COL, axes_mm, ctname, lineage_col, vcol, vname

COLS, N_CELLS, PER_SAMPLE, SEED = 8, 48, 3, 0
GAP, LBL = 2.5, 2.4  # gap between the two panels, row pitch below a tile for the cell-type name (mm)
TILE = (W - 4 - GAP) / 2 / COLS - 0.4
SAMPLE, KEY = cells.SAMPLE, cells.KEY


def panel_cells(var, mod, outcome):
    """(cells of the panel in cell-type order, n samples with this outcome, n samples shown)."""
    o = errors.samples()
    o = o[(o.modality == mod) & (o.variable == var) & (o.outcome == outcome)][SAMPLE + ["prob"]]
    o = o.astype({"database_id": str, "album_id": str})
    t = cells.top_cells(mod, var).merge(o).merge(cells.cell_table(mod)).merge(usable(mod))
    t = t[t.in_field]  # crop window inside the field image, image available
    # random order fixed per cell (hash of its key), so excluding a cell does not reshuffle the others
    key = t.database_id + "/" + t.album_id + "/" + t.cell_id.astype(str)
    t = t.assign(r=pd.util.hash_pandas_object(key, index=False, hash_key=f"{SEED:016d}")).sort_values("r")
    t = t.groupby(SAMPLE).head(PER_SAMPLE).head(N_CELLS)
    order = {ct: i for i, ct in enumerate(DIFF_ROWS[mod])}
    t = t.assign(o=t.class_label.map(order).fillna(99)).sort_values(["o", "att"], ascending=[True, False])
    return t, len(o), t[SAMPLE].drop_duplicates().shape[0]


def similarity_order(panels, mod):
    """Order the cells inside each cell-type block by similarity: leaf order (Ward, optimal leaf ordering) of the MIL cell
    features of the TP and FP cells of the type together, so neighbouring tiles look alike and TP / FP share one order."""
    allc = pd.concat([p.assign(panel=k) for k, p in panels.items()], ignore_index=True)
    X = cells.embed(allc[KEY], mod)
    allc["pos"] = 0
    for ct, g in allc.groupby("class_label"):
        if len(g) > 2:
            allc.loc[g.index[leaves_list(optimal_leaf_ordering(linkage(X[g.index], "ward"), pdist(X[g.index])))], "pos"] = \
                np.arange(len(g))
    return {k: g.sort_values(["o", "pos"]).drop(columns=["panel", "pos"]) for k, g in allc.groupby("panel")}


@cache
def selection(var):
    """{(modality, outcome): (cells, n samples, n shown)} of one page (similarity-ordered)."""
    sel = {(m, oc): panel_cells(var, m, oc) for m in ("BM", "PB") for oc in ("TP", "FP")}
    for m in ("BM", "PB"):
        o = similarity_order({oc: sel[m, oc][0] for oc in ("TP", "FP")}, m)
        sel.update({(m, oc): (o[oc].reset_index(drop=True),) + sel[m, oc][1:] for oc in ("TP", "FP")})
    return sel


def sample_numbers(t):
    """(database_id, album_id) → number among the samples shown (1 = highest predicted probability)."""
    sm = t.drop_duplicates(SAMPLE).sort_values("prob", ascending=False)
    return {k: i + 1 for i, k in enumerate(zip(sm.database_id, sm.album_id))}


def samples(var):
    """Sample table of a page (source data): number in the figure → sample."""
    tab = []
    for (m, oc), (t, *_) in selection(var).items():
        if len(t):
            num = sample_numbers(t)
            g = t.groupby(SAMPLE).agg(prob=("prob", "first"), n_cells=("cell_id", "size")).reset_index()
            tab.append(g.assign(target=var, modality=m, outcome=oc, number=[num[k] for k in zip(g.database_id, g.album_id)]))
    return pd.concat(tab).sort_values(["modality", "outcome", "number"])[
        ["target", "modality", "outcome", "number", "database_id", "album_id", "prob", "n_cells"]]


def flags(var, mod):
    e = effects()
    d = e[(e.variable == var) & (e.modality == mod) & (e.feature_type == "dysplasia") & e.associated]
    return list(dict.fromkeys(cells.DYS_COL[f] for f in d.feature if f in cells.DYS_COL))


def short(ct):
    n = ctname(ct)
    return (CT_SHORT.get(ct, n) if len(n) > 11 else n).lower()


def panel_h(t):
    return 4 + int(np.ceil(len(t) / COLS)) * (TILE + LBL) + 2 if len(t) else 8


def draw_panel(fig, H, x, y, var, mod, outcome, sel, ref_area):
    t, n_samples, n_shown = sel
    head = f"{mod} · {outcome}" + (f" · {len(t)} cells from {n_shown} of {n_samples} samples" if len(t) else "")
    fig.text((x + 0.3) / W, 1 - (y + 0.4) / H, head, fontsize=5.2, fontweight="bold", va="top", color=OUT_COL[outcome])
    if not len(t):
        fig.text((x + 0.3) / W, 1 - (y + 4.5) / H, f"no {'false' if outcome == 'FP' else 'true'} positives",
                 fontsize=4.6, va="top", color="#777", style="italic")
        return
    fl = flags(var, mod)
    med = t[cells.MEASURES].median()
    num = sample_numbers(t)
    for j, (_, r) in enumerate(t.iterrows()):
        tx, ty = x + (j % COLS) * (TILE + 0.4), y + 4 + (j // COLS) * (TILE + LBL)
        ax = axes_mm(fig, tx, ty, TILE, TILE, H)
        cells.draw_cell(ax, r.fillna(med), ref_area, fl, frame=lineage_col(r.class_label), mod=mod)
        fig.text((tx + TILE / 2) / W, 1 - (ty + TILE + 0.25) / H, short(r.class_label), ha="center", va="top",
                 fontsize=3.4, color="#555")
        ax.text(0.93, -0.93, str(num[(r.database_id, r.album_id)]), ha="right", va="bottom", fontsize=3.1, color="#222",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))  # sample number
        if r.low_quality >= 0.5:
            ax.text(0.9, 0.9, "LQ", fontsize=3.2, ha="right", va="top", color="#fff",
                    bbox=dict(boxstyle="round,pad=0.12", fc="#555", ec="none"))


def page(var):
    sel = selection(var)
    cells.write_list([(m, sel[m, oc][0]) for m, oc in sel], f"outcome_{var}")
    rows = {m: max(panel_h(sel[m, oc][0]) for oc in ("TP", "FP")) for m in ("BM", "PB")}
    H = 9 + sum(h + 2 for h in rows.values()) + 15
    fig = plt.figure(figsize=(W * MM, H * MM))
    fig.text(2 / W, 1 - 2 / H, f"{vname(var)} — top-attention cells of true- and false-positive samples", fontsize=7,
             fontweight="bold", color=vcol(var), va="top")
    y, pw = 9, (W - 4 - GAP) / 2
    for mod in ("BM", "PB"):
        ref_area = np.nanpercentile(cells.cell_table(mod).entire_area, 98)
        for k, oc in enumerate(("TP", "FP")):
            draw_panel(fig, H, 2 + k * (pw + GAP), y, var, mod, oc, sel[mod, oc], ref_area)
        y += rows[mod] + 2
    lg = [plt.Rectangle((0, 0), 1, 1, fc="none", ec=c, lw=1, label=l) for l, c in LINEAGE_COL.items()
          if l != "Myeloid precursors"]
    fig.legend(handles=lg, loc="upper center", bbox_to_anchor=(0.5, 1 - y / H), ncol=7, frameon=False, fontsize=4.4,
               handlelength=1, handleheight=1, columnspacing=1)
    fig.text(0.5, 1 - (y + 6) / H, "Top-attention cells (attention mass "
             f"0.5) of TP / FP samples, random draw (≤ {PER_SAMPLE} per sample, ≤ {N_CELLS} per panel), no quality or "
             "cell-type filter (edge cells skipped);\nordered by cell type, then by similarity of the MIL cell features "
             "(TP and FP share one order within a type).\nFrame and label = cell type; chips = classifier flags of "
             "the target's associated dysplasia features; LQ = low-quality score ≥ 0.5;\nbottom-right number = sample "
             f"(per panel; 1 = highest predicted probability; at most {PER_SAMPLE} cells per sample).", ha="center",
             va="top", fontsize=4.4, color="#444", linespacing=1.35)
    return fig
