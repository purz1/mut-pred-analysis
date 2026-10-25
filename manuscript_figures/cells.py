"""Cell-level data and drawing shared by the cell-image figures: Fig 6 gallery (gallery.py), S1 cards (cards.py) and
S2 pages (outcome_pages.py).

Data: the cell-level parquet (class, low-quality score, morphometry, classifier flags); derived/gallery/top_cells.parquet =
top-attention cells (attention-mass threshold 0.5, as Fig 4/5) of every sample per target and modality
(scripts/extract_gallery_cells.py); cell images derived/cell_crops (scripts/extract_cell_crops.py, window in crops.py).
Areas are converted with config.UM_PER_PX (PLACEHOLDER).
Cell choice for the cards:
    typical   closest to the top cells' median (frame measure ×2, other slider measures ×1); TP samples first
    striking  most extreme in the frame measure in the effect direction (within the 2nd–98th percentile)
Chips: classifier flags of the cell for the dysplasia features associated with the target for that cell type.
Sliders (real units, zoomed): ▼ = this cell (open = outside the range); bars = middle 50% of Neg (blue) / Pos (orange)
cells of the type.
"""
from functools import cache

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

from core.data_loading import FEAT_DIR, FILES, load_cohort
from . import config as C, errors
from .crops import usable
from .fig4 import STATS, effects
from .style import W, GROUP_COL, axes_mm

MEASURES = STATS
SAMPLE = ["database_id", "album_id"]
KEY = SAMPLE + ["cell_id"]
SLIDER_NAME = {"entire_area": "area", "nucleus_area": "nucl.", "entire_N:C-ratio": "N:C",
               "nucleus_solidity": "solid.", "nucleus_eccentricity": "eccen."}
DYS_COL = {"bilobed": "apl_bilobed", "hypergranulated": "apl_hypergranulated", "auer": "apl_auer",
           "vacuolated": "vacuolated", "asynchrony": "mds_asynchrony", "dysmorphic": "mds_dysmorphic",
           "multinucleated": "mds_multinucleated", "hypolobated": "mgk_hypolobated"}
TAG = {"apl_bilobed": "bilobed", "apl_hypergranulated": "hypergran.", "apl_auer": "Auer", "vacuolated": "vacuol.",
       "mds_asynchrony": "asynchr.", "mds_dysmorphic": "dysmorph.", "mds_multinucleated": "multinucl.",
       "mgk_hypolobated": "hypolob."}
CLASS_GROUP = {"Myeloid_precursors": ["Blasts", "Promyelocytes", "Myelocytes"]}
# dysplasia category colours as fig4.DYS_COL (11_cytomorphology_analysis.ipynb balloon plots)
CHIP_COL = {"apl_bilobed": "#f8d097", "apl_hypergranulated": "#f8d097", "apl_auer": "#f8d097", "vacuolated": "#f9d7c9",
            "mds_asynchrony": "#b6a0bc", "mds_dysmorphic": "#b6a0bc", "mds_multinucleated": "#b6a0bc",
            "mgk_hypolobated": "#abc2ca"}


# ── data ─────────────────────────────────────────────────────────────────────
@cache
def _cohort(mod):
    return load_cohort(mod)


@cache
def cell_table(mod):
    """All cells of the cell-level parquet (ids as str): class, low-quality score, morphometry (areas in µm²), flags."""
    f = FEAT_DIR / FILES[mod]["parquet"]
    have = set(pq.read_schema(f).names)
    c = pd.read_parquet(f, columns=KEY + ["class_label", "low_quality"] + MEASURES + [t for t in TAG if t in have])
    c = c.astype({**{m: float for m in MEASURES}, "database_id": str, "album_id": str, "cell_id": "int64"})
    for m in MEASURES:  # px → µm² for areas (C.UM_PER_PX is a PLACEHOLDER)
        c[m] *= C.to_um(m, mod)[0]
    return c.assign(key=c.database_id + "/" + c.album_id + "/" + c.cell_id.astype(str))


@cache
def top_cells(mod, var):
    """Top-attention cells of all samples for one target: ids, attention score att, att / the sample's total attention w."""
    t = pd.read_parquet(C.TOP_CELLS, filters=[("modality", "==", mod), ("variable", "==", var)], columns=KEY + ["att", "w"])
    return t.astype({"database_id": str, "album_id": str, "cell_id": "int64"})


@cache
def load(var, mod):
    """Cells (low-quality < 0.5) of one modality with pos (target label), top (top-attention for var), outcome, in_field."""
    c = cell_table(mod)
    c = c[c.low_quality < 0.5]
    df = _cohort(mod)[SAMPLE + [var]].rename(columns={var: "pos"})
    o = errors.samples()
    o = o[(o.modality == mod) & (o.variable == var)][SAMPLE + ["outcome"]]
    top = top_cells(mod, var)[KEY].assign(top=True)
    c = c.merge(df, on=SAMPLE).merge(o, how="left").merge(top, how="left").merge(usable(mod), how="left")
    c["top"] = c.top.eq(True)
    return c


def embed(t, mod):
    """384-d MIL cell features (h5) of the cells in t (columns database_id, album_id, cell_id; RangeIndex)."""
    X = np.zeros((len(t), 384), np.float32)
    with h5py.File(FEAT_DIR / FILES[mod]["h5"], "r") as f:
        for (d, a), g in t.groupby(SAMPLE):
            h = f[d][a]
            pos = {c: i for i, c in enumerate(h["cell_ids"][:])}
            X[g.index] = h["features"][:][[pos[c] for c in g.cell_id]]
    return X


# ── cell choice (cards) ──────────────────────────────────────────────────────
def measures(var, mod, ct, first=None, n=3):
    """[(measure, direction)]: frame measure first, then the measures most associated for this cell type (|δ|)."""
    e = effects()
    d = e[(e.variable == var) & (e.modality == mod) & (e.cell_type == ct) & e.feature.isin(MEASURES)]
    d = d.assign(s=d.delta_pos_vs_neg.abs().where(d.associated, 0)).sort_values("s", ascending=False)
    sign = dict(zip(d.feature, np.sign(d.delta_pos_vs_neg)))
    order = list(dict.fromkeys(([first] if first else []) + d.feature.tolist() + MEASURES))[:n]
    return [(m, sign.get(m, 1) or 1) for m in order]


def tag_flags(var, mod, ct):
    """Flag columns of the dysplasia features associated with var for this cell type (incl. class groups)."""
    e = effects()
    d = e[(e.variable == var) & (e.modality == mod) & (e.feature_type == "dysplasia") & e.associated]
    d = d[(d.cell_type == ct) | d.cell_type.map(lambda g: ct in CLASS_GROUP.get(g, []))]
    return list(dict.fromkeys(DYS_COL[f] for f in d.feature if f in DYS_COL))


def cell_pool(var, row):
    """(cells of the panel's cell type, cell type, flag) for an effects row; class groups → most flagged class."""
    c = load(var, row.modality)
    flag = DYS_COL.get(row.feature) if row.feature_type == "dysplasia" else None
    ct = row.cell_type
    if ct in CLASS_GROUP:
        g = c[c.class_label.isin(CLASS_GROUP[ct]) & c.top & (c.pos == 1) & (c[flag] if flag else True)]
        ct = g.class_label.mode().iat[0] if len(g) else CLASS_GROUP[ct][0]
    return c[c.class_label == ct], ct, flag


def candidates(c, ms, flag=None, used=(), top=True):
    cand = c[(c.pos == 1) & (c.top if top else True) & c.in_field.eq(True)].dropna(subset=[m for m, _ in ms])
    if flag:
        cand = cand[cand[flag]]
    return cand[~cand.key.isin(used)].assign(tp=lambda x: x.outcome.eq("TP"))


def typical(cand, ms, n):
    q = (cand[[m for m, _ in ms]].rank(pct=True) - 0.5).abs()
    w = np.ones(len(ms)); w[0] = 2
    cand = cand.assign(atyp=(q * w).sum(axis=1)).sort_values(["tp", "atyp"], ascending=[False, True])
    return cand.drop_duplicates(SAMPLE).head(n)


def striking(cand, ms, ref):
    m, s = ms[0]
    lo, hi = np.nanpercentile(ref[m], [2, 98])  # avoid segmentation artefacts at the extremes
    cand = cand[cand[m].between(lo, hi)]
    return cand.sort_values(["tp", m], ascending=[False, s < 0]).head(1)


# ── drawing ──────────────────────────────────────────────────────────────────
def crop_path(r, mod):
    return C.CROPS / mod / str(r.database_id) / str(r.album_id) / f"{int(r.cell_id)}.png"


def write_list(cells, name):
    """Cell list for scripts/extract_cell_crops.py: [(modality, DataFrame with database_id, album_id, cell_id)]."""
    d = C.CROPS / "lists"
    d.mkdir(parents=True, exist_ok=True)
    pd.concat(c[KEY].assign(modality=m) for m, c in cells).to_csv(d / f"{name}.csv", index=False)


def draw_cell(ax, r, ref_area, tags=(), frame="#bbb", letter=None, mod="BM"):
    """Cached crop (scripts/extract_cell_crops.py) or, if missing, a stand-in drawn from the cell's measurements;
    + chips (flags the cell carries) and letter."""
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal"); ax.axis("off")
    p = crop_path(r, mod)
    if p.exists():
        ax.imshow(plt.imread(p), extent=(-1, 1, -1, 1))
        ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fc="none", ec=frame, lw=0.4 if frame == "#bbb" else 1.0))
        _chips(ax, r, tags, letter)
        return
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fc="#f3ece6", ec=frame, lw=0.4 if frame == "#bbb" else 1.0))
    R = min(0.85 * np.sqrt(r.entire_area / ref_area), 0.95)
    ax.add_patch(Circle((0, 0), R, fc="#e7c6d6", ec="#c9a2b8", lw=0.4))
    na = np.clip(r["entire_N:C-ratio"], 0.1, 0.95) * R ** 2
    e = float(np.clip(r.nucleus_eccentricity, 0, 0.95))
    a = np.sqrt(na / np.sqrt(1 - e ** 2)); b = a * np.sqrt(1 - e ** 2)
    if r.nucleus_solidity < 0.93 or r.get("apl_bilobed", False):  # indented / bilobed nucleus: two lobes
        for s in (-1, 1):
            ax.add_patch(Ellipse((s * a * 0.45, 0), a * 1.05, b * 1.6, angle=s * 20, fc="#5b3a8c", ec="none"))
    else:
        ax.add_patch(Ellipse((0, 0), 2 * a, 2 * b, angle=30, fc="#5b3a8c", ec="none"))
    _chips(ax, r, tags, letter)


def _chips(ax, r, tags, letter):
    for k, t in enumerate(f for f in tags if bool(r.get(f, False))):
        ax.text(-0.94, -0.92 + k * 0.29, TAG[t], fontsize=3.3, color="#222", ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc=CHIP_COL[t], ec="#555", lw=0.25))
    if letter:
        ax.text(-0.9, 0.9, letter, fontsize=4.6, fontweight="bold", ha="left", va="top", color="#111",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))


def zoom(*iqrs, pad=0.4):
    """Axis range around the given (q1, q3) bars, padded by pad × their span (zoom on where groups differ)."""
    lo, hi = min(q[0] for q in iqrs), max(q[1] for q in iqrs)
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def slider(ax, ref, m, v, fontsize=3.4):
    """Real units, zoomed on the Neg (blue) and Pos (orange) per-cell IQRs; ▼ = cell (at the edge if outside)."""
    iq = {g: np.nanpercentile(ref.loc[ref.pos == (g == "Pos"), m], [25, 75]) for g in ("Neg", "Pos")}
    lo, hi = zoom(*iq.values())
    pad = 0.30 * (hi - lo)  # room for the label
    ax.set_xlim(lo - pad, hi); ax.set_ylim(0, 1); ax.axis("off")
    ax.plot([lo, hi], [0.4, 0.4], color="#ccc", lw=0.4)
    for g, y in (("Neg", 0.55), ("Pos", 0.25)):
        ax.plot(iq[g], [y, y], color=GROUP_COL[g], lw=1.1, solid_capstyle="butt")
    inside = lo <= v <= hi
    ax.scatter([np.clip(v, lo, hi)], [0.9], marker="v", s=4, lw=0.4, clip_on=False, zorder=3,
               color="#111" if inside else "none", edgecolors="#111")
    ax.text(lo - pad, 0.4, SLIDER_NAME[m], ha="left", va="center", fontsize=fontsize, color="#444")


def reference(c, ct):
    """Cells of one type for the sliders (≤ 200 000, fixed seed)."""
    r = c[c.class_label == ct]
    return r.sample(min(len(r), 200_000), random_state=0)


def cell_block(fig, x, y, img, H, cell, c, ms, tags, label, color="#555", mod="BM"):
    """Label, cell image with chips, sliders below; returns the block height (mm)."""
    ref = c.sample(min(len(c), 200_000), random_state=0)
    fig.text((x + img / 2) / W, 1 - (y + 1.6) / H, label, ha="center", fontsize=3.6, color=color)
    draw_cell(axes_mm(fig, x, y + 2, img, img, H), cell, np.nanpercentile(ref.entire_area, 98), tags, mod=mod)
    for i, (m, _) in enumerate(ms):
        slider(axes_mm(fig, x, y + img + 2.7 + i * 1.9, img, 1.9, H), ref, m, cell[m])
    return 2 + img + 0.7 + len(ms) * 1.9
