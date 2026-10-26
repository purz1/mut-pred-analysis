"""Fig 6: cell gallery (BM) — tiles = embedding clusters of the model's top-attention cells — and their 2D embedding.

Build (scripts/extract_gallery_cells.py → derived/gallery/): per target, the top-attention cells (attention-mass threshold
0.5, as Fig 4/5) of all TP samples (errors.samples) are pooled, embedded with the 384-d MIL cell features, reduced
(PCA N_PCA) and clustered (k-means, K clusters, seed SEED); a 2D UMAP is stored for the scatter.
    attention share of a cluster = sum over its cells of (cell attention / total attention of its sample) / total
                                   (every sample weighs about the same; the K clusters sum to 1)
    support of a cluster           = TP samples in which the cluster holds ≥ SUPPORT of the sample's top-cell attention
Tile = for each of the N_TILES largest clusters that has a usable cell (low-quality < 0.5, crop window inside the field,
image available): among the 10% of its usable cells nearest the cluster centre, the cell of the most confident TP sample
not yet shown. FORCE: if none of the tiles carries the target's flag, the shown cluster with the largest share of flagged cells keeps its
tile but its cell is chosen among the flagged cells of the cluster (same rule, nearest the centre; marked † in the figure).
Sliders: morphometry features associated with the target in the tile's cell type on which the cell lies
beyond the median of the Pos cells of that type in the effect direction (≤ MAX_SL, largest |δ| first).
METHODS (build option; the default "kmeans" is the Fig 6 version, the others are alternatives for comparison, written with a
file-name suffix and drawn by make(method)):
    kmeans    the clustering above
    weighted  k-means with every cell weighted by its attention (sample_weight w), so centres follow the attention mass
    medoid    no clusters: N_TILES representative cells that minimise the attention-weighted distance of all pooled cells to
              their nearest shown cell (k-medoids in the PCA space: greedy facility-location start + swaps, shown cells from
              distinct usable samples); "clusters" = the cells nearest to each shown cell
Draw (notebook): make() reads the tables; scatter() the embedding.
"""
from functools import cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from . import cells, config as C, errors
from .crops import field_width
from .fig4 import effects
from .style import (MM, W, CT_SHORT, CT_LINEAGE, DIFF_ROWS, LINEAGE_COL, axes_mm, ctname, lineage_col, vcol, vname)

TARGETS = ["PML-RARA", "CBFB-MYH11", "RUNX1-RUNX1T1_OR_t_8_21", "NPM1", "IDH_mutations", "monosomy5_del_5q"]
MOD = "BM"
K, N_PCA, SEED, SUPPORT, N_TILES, MAX_SL = 12, 50, 0, 0.05, 8, 3
SAMPLE, KEY, MEASURES = cells.SAMPLE, cells.KEY, cells.MEASURES
LETTERS = "abcdefgh"
METHODS = ("kmeans", "weighted", "medoid")
FORCE = {"PML-RARA": "apl_auer"}  # target → classifier flag that one tile must carry (Auer rods are rare among the top cells)


# ── build ────────────────────────────────────────────────────────────────────
def medoids(Z, w, usable, sample, k, passes=5):
    """Weighted k-medoids (Euclidean): k usable cells of distinct samples minimising sum_i w_i · distance(i, nearest pick);
    greedy start (facility location) + swap passes. Returns (pick indices, label of the nearest pick, distance to it)."""
    from sklearn.metrics import pairwise_distances

    cand = np.flatnonzero(usable)
    D = pairwise_distances(Z, Z[cand]).astype(np.float32)  # all cells × candidates
    w = w.astype(np.float32)
    cs = sample[cand]

    def best(cur, taken):
        cost = np.empty(len(cand), np.float32)
        for a in range(0, len(cand), 1500):
            cost[a:a + 1500] = w @ np.minimum(D[:, a:a + 1500], cur[:, None])
        cost[np.isin(cs, list(taken))] = np.inf
        return int(cost.argmin()), float(cost.min())

    picks = []
    cur = np.full(len(Z), D.max(), np.float32)
    for _ in range(k):
        j, _ = best(cur, {cs[q] for q in picks})
        picks.append(j); cur = np.minimum(cur, D[:, j])
    for _ in range(passes):  # swaps: re-optimise each pick given the others
        changed = False
        for q in range(k):
            others = [x for x in picks if x != picks[q]]
            cur = D[:, others].min(1)
            j, cost = best(cur, {cs[x] for x in others})
            if j != picks[q] and cost < w @ np.minimum(D[:, picks[q]], cur) - 1e-6:
                picks[q] = j; changed = True
        if not changed:
            break
    Dp = D[:, picks]
    return cand[picks], Dp.argmin(1), Dp.min(1)


def pool(var, mod=MOD, k=K, method="kmeans", embedding=True):
    """(pooled top-attention cells of the TP samples with cluster, distance to centre / shown cell d, UMAP u0 / u1,
    scanner width; n_tp). method: see METHODS (medoid: k = N_TILES groups, d = distance to the shown cell)."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    o = errors.samples()
    o = o[(o.modality == mod) & (o.variable == var) & (o.outcome == "TP")][SAMPLE + ["prob"]]
    o = o.astype({"database_id": str, "album_id": str})
    t = cells.top_cells(mod, var).merge(o)
    t = t.merge(cells.cell_table(mod)[KEY + ["class_label", "low_quality"]], on=KEY).rename(columns={"class_label": "cell_label"})
    t = t.reset_index(drop=True)
    Z = PCA(N_PCA, random_state=SEED).fit_transform(cells.embed(t, mod))
    if method == "medoid":
        ok = cells.load(var, mod)
        ok = ok[ok.in_field.eq(True)][KEY].assign(u=True)
        usable = t[KEY].merge(ok, on=KEY, how="left").u.eq(True).to_numpy()
        pick, lab, d = medoids(Z, t.w.to_numpy(), usable, (t.database_id + "/" + t.album_id).to_numpy(), N_TILES)
        t["cluster"], t["d"] = lab, d
    else:
        km = KMeans(k, n_init=10, random_state=SEED).fit(Z, sample_weight=t.w.to_numpy() if method == "weighted" else None)
        t["cluster"] = km.labels_
        t["d"] = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)
    if embedding:
        import umap
        u = umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=SEED).fit_transform(Z)
        t["u0"], t["u1"] = u[:, 0], u[:, 1]
    return t.merge(field_width(mod), on=SAMPLE, how="left"), len(o)


def summary(t):
    """Per cluster: n_cells, attention share, support, dominant label (+ share), low-quality and old-scanner shares."""
    wn = t.w / t.groupby(SAMPLE).w.transform("sum")
    sup = wn.groupby([t.cluster, t.database_id, t.album_id]).sum().ge(SUPPORT).groupby("cluster").sum()
    s = t.groupby("cluster").agg(n_cells=("w", "size"), share=("w", "sum"), lq=("low_quality", lambda x: (x >= 0.5).mean()),
                                 old=("width", lambda x: (x == 2464).mean()), dom=("cell_label", lambda x: x.mode().iat[0]),
                                 dom_share=("cell_label", lambda x: x.value_counts(normalize=True).iat[0]))
    s["share"] /= s.share.sum()
    s["support"] = sup
    return s.sort_values("share", ascending=False)


def choose_tiles(var, t, s, n_tp, mod=MOD, n=N_TILES, method="kmeans"):
    """Tile cells (rows of cells.load with cluster, share, support, n_tp, dom, prob) in cluster-rank order."""
    c = cells.load(var, mod)
    cand = t[KEY + ["cell_label", "prob", "cluster", "d"]].merge(c[c.in_field.eq(True)], on=KEY)
    rows, used = [], set()
    for cl in s.index:
        g = cand[cand.cluster == cl]
        if g.empty:
            continue
        near = g.nsmallest(1 if method == "medoid" else max(10, len(g) // 10), "d")
        fresh = near[~(near.database_id + "/" + near.album_id).isin(used)]
        pick = (fresh if len(fresh) else near).sort_values("prob", ascending=False).head(1)
        used.add(pick.database_id.iat[0] + "/" + pick.album_id.iat[0])
        rows.append(pick.assign(share=s.at[cl, "share"], support=s.at[cl, "support"], n_tp=n_tp, dom=s.at[cl, "dom"], forced=""))
        if len(rows) == n:
            break
    flag = FORCE.get(var)
    if flag and not any(r[flag].eq(True).any() for r in rows):
        flagged = cand[cand[flag].eq(True)]
        share, count = cand.groupby("cluster")[flag].mean(), flagged.groupby("cluster").size()
        shown = [r.cluster.iat[0] for r in rows]
        best = max((cl for cl in shown if count.get(cl, 0) >= 5), key=lambda cl: share[cl], default=None)
        if best is not None:
            j = shown.index(best)
            g = flagged[flagged.cluster == best]
            near = g.nsmallest(1 if method == "medoid" else max(10, len(g) // 10), "d")
            others = {r.database_id.iat[0] + "/" + r.album_id.iat[0] for k, r in enumerate(rows) if k != j}
            fresh = near[~(near.database_id + "/" + near.album_id).isin(others)]
            pick = (fresh if len(fresh) else near).sort_values("prob", ascending=False).head(1)
            rows[j] = pick.assign(share=rows[j].share.iat[0], support=rows[j].support.iat[0], n_tp=n_tp,
                                  dom=rows[j].dom.iat[0], forced=flag)
    return pd.concat(rows, ignore_index=True)


def pos_like(var, ct, cell, ref, mod=MOD):
    """Morphometry features (largest |δ| first) associated with the target for this cell type on which the cell lies beyond
    the median of the Pos cells of the type, in the effect direction."""
    e = effects()
    d = e[(e.variable == var) & (e.modality == mod) & (e.cell_type == ct) & e.feature.isin(MEASURES) & e.associated]
    out = []
    for r in d.assign(a=d.delta_pos_vs_neg.abs()).sort_values("a", ascending=False).itertuples():
        if np.sign(r.delta_pos_vs_neg) * (cell[r.feature] - ref.loc[ref.pos == 1, r.feature].median()) >= 0:
            out.append(r.feature)
    return out


def build(var, mod=MOD, method="kmeans"):
    """(pooled cells, tile table rows, cluster table rows) of one target, as written to derived/gallery/."""
    t, n_tp = pool(var, mod, method=method, embedding=method == "kmeans")
    s = summary(t)
    tiles = choose_tiles(var, t, s, n_tp, mod, method=method)
    c = cells.load(var, mod)
    refs = {ct: cells.reference(c, ct) for ct in tiles.class_label.unique()}
    sl = [", ".join(pos_like(var, r.class_label, r, refs[r.class_label], mod)[:MAX_SL]) for _, r in tiles.iterrows()]
    letter = {r.cluster: LETTERS[j] for j, (_, r) in enumerate(tiles.iterrows())}
    tt = pd.DataFrame(dict(target=var, tile=[LETTERS[j] for j in range(len(tiles))], cell_type=tiles.class_label.values,
                           cluster=tiles.cluster.values, attention_share=tiles.share.values,
                           support=tiles.support.astype(int).values, n_tp=n_tp, dominant_label=tiles.dom.values,
                           database_id=tiles.database_id.values, album_id=tiles.album_id.values, cell_id=tiles.cell_id.values,
                           sample_probability=tiles.prob.values, sliders=sl, forced=tiles.forced.values))
    ct = s.reset_index().assign(target=var, tile=lambda d: d.cluster.map(letter).fillna(""), n_tp=n_tp) \
        .rename(columns={"share": "attention_share", "dom": "dominant_label", "dom_share": "dominant_label_share",
                         "lq": "low_quality_share", "old": "old_scanner_share"})
    ct = ct[["target", "cluster", "tile", "n_cells", "attention_share", "support", "n_tp", "dominant_label",
             "dominant_label_share", "low_quality_share", "old_scanner_share"]]
    return t.assign(target=var), tt, ct


# ── draw ─────────────────────────────────────────────────────────────────────
def suffix(method):
    return "" if method == "kmeans" else f"_{method}"


@cache
def tables(method="kmeans"):
    """(tile table, cluster table) written by scripts/extract_gallery_cells.py."""
    ids = {"database_id": str, "album_id": str}
    tiles = pd.read_csv(C.GALLERY / f"gallery_tiles{suffix(method)}.csv", dtype=ids).fillna({"sliders": "", "forced": ""})
    return tiles, pd.read_csv(C.GALLERY / f"gallery_clusters{suffix(method)}.csv")


def short(ct):
    n = ctname(ct)
    return (CT_SHORT.get(ct, n) if len(n) > 10 else n).lower()


HEAD, PAD, BAR_H, LEG_H, CAP_H = 5.8, 1.2, 4.6, 6.5, 11.5  # block title band, row padding, cluster bar, legend, caption (mm)
CAP, SL_H, COLS = 2.4, 1.9, 4  # caption under a tile, slider height, tiles per row
BW = (W - 4 - 2) / 2
PITCH = (BW - 3) / COLS
TILE = PITCH - 1


def prepare(var, tiles, clusters):
    t = tiles[tiles.target == var].reset_index(drop=True)
    c = cells.load(var, MOD)
    rows = t.merge(c, on=KEY, how="left")
    cts = list(dict.fromkeys(rows.class_label))
    return dict(var=var, cells=rows, clus=clusters[clusters.target == var].set_index("cluster"),
                sl=[s.split(", ") if s else [] for s in t.sliders], refs={ct: cells.reference(c, ct) for ct in cts},
                tags=list(dict.fromkeys(f for ct in cts for f in cells.tag_flags(var, MOD, ct))),
                ref_area=np.nanpercentile(c.entire_area, 98))


def rowh(n):
    return TILE + CAP + n * SL_H + PAD


def layout(blocks):
    """Row heights from the slider counts; the two blocks of a figure row share their row heights."""
    for b in blocks:
        b["rowh"] = [max([len(s) for s in b["sl"][i:i + COLS]], default=0) for i in (0, COLS)]
    for k in range(0, len(blocks), 2):
        for i in (0, 1):
            blocks[k]["rowh"][i] = blocks[k + 1]["rowh"][i] = max(blocks[k]["rowh"][i], blocks[k + 1]["rowh"][i])
    for b in blocks:
        b["rowh"] = [rowh(n) for n in b["rowh"]]
        b["h"] = HEAD + sum(b["rowh"]) + BAR_H + 1.0


def draw_bar(fig, H, x, y, w, clus, tiles, title=None):
    """Thin bar: attention share of the clusters of the tiles (letters, colour = dominant lineage), rest grey."""
    fig.text(x / W, 1 - y / H, title or f"attention by embedding cluster (k = {K}; letters = tiles)", fontsize=3.1, color="#555",
             va="top")
    ax = axes_mm(fig, x, y + 1.8, w, 1.9, H); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    left = 0
    for j, (_, r) in enumerate(tiles.iterrows()):
        v = clus.at[r.cluster, "attention_share"]
        ax.add_patch(plt.Rectangle((left, 0), v, 1, fc=lineage_col(r.dominant_label), ec="white", lw=0.4))
        if v >= 0.04:
            ax.text(left + v / 2, 0.5, LETTERS[j], ha="center", va="center", fontsize=3.0, fontweight="bold", color="#111")
        left += v
    ax.add_patch(plt.Rectangle((left, 0), 1 - left, 1, fc="#e3e3e3", ec="white", lw=0.4))
    if 1 - left >= 0.1:
        ax.text(left + (1 - left) / 2, 0.5, f"other {1 - left:.0%}", ha="center", va="center", fontsize=2.9, color="#777")


def draw_block(fig, H, x, y, h, b, bar_title=None):
    fr = axes_mm(fig, x, y, BW, h, H); fr.set_xticks([]); fr.set_yticks([]); fr.patch.set_alpha(0)
    for sp in fr.spines.values():
        sp.set_color("#ccc"); sp.set_linewidth(0.5)
    fig.text((x + 2) / W, 1 - (y + 1.2) / H, vname(b["var"]), fontsize=6.5, fontweight="bold", color=vcol(b["var"]), va="top")
    for j, (_, cell) in enumerate(b["cells"].iterrows()):
        tx, ty = x + 2 + (j % COLS) * PITCH, y + HEAD + (j // COLS) * b["rowh"][0]
        ct = cell.class_label
        cells.draw_cell(axes_mm(fig, tx, ty, TILE, TILE, H), cell, b["ref_area"], b["tags"], frame=lineage_col(ct),
                        letter=LETTERS[j], mod=MOD)
        fig.text((tx + TILE / 2) / W, 1 - (ty + TILE + 0.3) / H, short(ct) + (" †" if cell.forced else ""), ha="center", va="top",
                 fontsize=3.3, color="#555")
        for i, m in enumerate(b["sl"][j]):
            cells.slider(axes_mm(fig, tx, ty + TILE + CAP + i * SL_H, TILE, SL_H, H), b["refs"][ct], m, cell[m])
    draw_bar(fig, H, x + 2, y + HEAD + sum(b["rowh"]) + 0.1, BW - 4, b["clus"], b["cells"], bar_title)


def legend(fig, H, y, blocks):
    cts = set()
    for b in blocks:
        cts |= set(b["cells"].class_label) | set(b["cells"].dominant_label)
    order = {ct: i for i, ct in enumerate(DIFF_ROWS[MOD] + list(CT_LINEAGE))}
    hs = []
    for lin, col in LINEAGE_COL.items():
        mem = sorted((ct for ct in cts if CT_LINEAGE.get(ct) == lin), key=lambda ct: order.get(ct, 99))
        if mem:
            names = ", ".join(ctname(ct).lower() for ct in mem)
            hs.append(plt.Rectangle((0, 0), 1, 1, fc=col, ec="none", label=lin if names == lin.lower() else f"{lin}: {names}"))
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 1 - y / H), ncol=3, frameon=False, fontsize=4.3,
               handlelength=1, handleheight=0.9, columnspacing=1.2, labelspacing=0.35, borderaxespad=0)


CAPTION = {
    "kmeans": (f"the {N_TILES} largest of {K} embedding clusters (k-means on the MIL cell features) of the top-attention cells of all TP "
               "samples (BM model), one cell each (nearest the cluster centre,\nmost confident TP sample not yet shown)",
               None),
    "weighted": (f"the {N_TILES} largest of {K} attention-weighted embedding clusters (k-means on the MIL cell features, every cell weighted by "
                 "its attention) of the top-attention cells of all TP samples (BM model),\none cell each (nearest the cluster centre, most "
                 "confident TP sample not yet shown)", None),
    "medoid": (f"{N_TILES} representative cells of the top-attention cells of all TP samples (BM model): the cells, from distinct samples, that "
               "minimise the attention-weighted distance of all these cells to their nearest shown cell\n(k-medoids on the MIL cell features)",
               "attention share of the cells nearest to each shown cell (letters = tiles)"),
}


def make(method="kmeans"):
    tiles, clusters = tables(method)
    head, bar_title = CAPTION[method]
    blocks = [prepare(v, tiles, clusters) for v in TARGETS]
    layout(blocks)
    row_h = [max(blocks[k]["h"], blocks[k + 1]["h"]) for k in range(0, len(blocks), 2)]
    H = 3 + sum(h + 2 for h in row_h) + LEG_H + CAP_H + (1.9 if tiles.forced.ne("").any() else 0)
    fig = plt.figure(figsize=(W * MM, H * MM))
    y = 3
    for r, h in enumerate(row_h):
        for k, b in enumerate(blocks[2 * r:2 * r + 2]):
            draw_block(fig, H, 2 + k * (BW + 2), y, h, b, bar_title)
        y += h + 2
    legend(fig, H, y + 0.2, blocks)
    fig.text(0.5, 1 - (H - CAP_H / 2 - 0.3) / H,
             f"Cells a–h (BM crops, one scale): {head}; cluster shares and supporting TP samples: source table (notebook, Figure 6).\n"
             f"Sliders (≤ {MAX_SL} per cell): only morphometry features associated with the target in the cell's type on which the cell lies beyond the "
             "median of the Pos cells of that type (effect direction);\n"
             "▼ = this cell (open = outside the axis) against the middle 50% of Neg (blue) and Pos (orange) cells, axis zoomed, real units "
             "(areas in µm²).\n"
             "Frame = cell-type colour; chips = classifier flags (dysplasia colours). Bar = " + (
                 "attention share of the cells nearest to each shown cell (letters = tiles, colour = dominant lineage)"
                 if method == "medoid" else
                 "attention share of the clusters (letters = tiles, colour = dominant lineage, grey = other clusters)") +
             "; each sample weighted equally." + (
                 "\n† = cell chosen among the Auer-rod-flagged cells of its cluster (Auer rods are rare among the top cells)."
                 if tiles.forced.ne("").any() else ""),
             ha="center", va="center", fontsize=4.1, color="#444", linespacing=1.3)
    return fig


# ── 2D embedding of the pooled cells ─────────────────────────────────────────
CELL_COLORS = {  # as in the cell-attention visualisation notebook
    "Proerythroblasts": "maroon", "Erythroblasts": "lightcoral", "Megakaryocytes": "darkorange", "Blasts": "tomato",
    "Promyelocytes": "darkolivegreen", "Myelocytes": "yellowgreen", "Metamyelocytes": "springgreen",
    "Basophils": "tab:brown", "Neutrophils": "dodgerblue", "Eosinophils_immature": "mediumpurple",
    "Eosinophils": "blueviolet", "Promonocytes": "navy", "Monocytes": "gold", "Lymphocytes": "teal",
    "Plasma_cells": "turquoise"}
TILE_COLORS = plt.get_cmap("tab10").colors[:N_TILES]


def scatter():
    """UMAP of the pooled top-attention cells of the TP samples per target: coloured by cell type (left) and by cluster
    (right; the 8 tile clusters coloured, the others grey); letters = tile cells."""
    pc = pd.read_parquet(C.GALLERY / "cluster_cells.parquet")
    tiles, _ = tables()
    pw, ph, blk = 41, 41, 88
    H = 3 + 3 * (ph + 8) + 22
    fig = plt.figure(figsize=(W * MM, H * MM))
    halo = [pe.withStroke(linewidth=1.2, foreground="white")]
    present = set()
    for k, var in enumerate(TARGETS):
        x0, y0 = 2 + (k % 2) * blk, 3 + (k // 2) * (ph + 8)
        p = pc[pc.target == var]
        t = tiles[tiles.target == var].merge(p[KEY + ["u0", "u1"]], on=KEY)
        letter = dict(zip(t.cluster, t.tile))
        fig.text((x0 + 0.5) / W, 1 - (y0 + 0.2) / H, vname(var) + f"  ({len(p):,} cells, {int(t.n_tp.iat[0])} TP samples)",
                 fontsize=6, fontweight="bold", color=vcol(var), va="top")
        for j, mode in enumerate(("cell type", "cluster")):
            ax = axes_mm(fig, x0 + j * (pw + 2), y0 + 5.5, pw, ph, H)
            if mode == "cell type":
                for lab, g in p.groupby("cell_label"):
                    ax.scatter(g.u0, g.u1, s=0.5, c=CELL_COLORS.get(lab, "grey"), lw=0, rasterized=True)
                present |= set(p.cell_label)
            else:
                other = ~p.cluster.isin(letter)
                ax.scatter(p.u0[other], p.u1[other], s=0.5, c="#d6d6d6", lw=0, rasterized=True)
                for cl, lt in letter.items():
                    g = p[p.cluster == cl]
                    ax.scatter(g.u0, g.u1, s=0.5, c=[TILE_COLORS[LETTERS.index(lt)]], lw=0, rasterized=True)
            for _, r in t.iterrows():
                ax.text(r.u0, r.u1, r.tile, fontsize=4.2, fontweight="bold", ha="center", va="center", zorder=5,
                        path_effects=halo)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("#888")
            ax.set_title(mode, fontsize=4.6, pad=1.5, color="#555")
    hs = [plt.Line2D([], [], marker="o", ls="", ms=3, color=CELL_COLORS.get(ct, "grey"), label=ctname(ct))
          for ct in DIFF_ROWS[MOD] if ct in present]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1 - (H - 15) / H - 0.0), ncol=7, frameon=False,
               fontsize=4.3, handletextpad=0.2, columnspacing=1.0)
    fig.text(0.5, 1 - (H - 8) / H, "UMAP of the 384-d MIL cell features of the top-attention cells (attention mass 0.5) of the TP samples "
             "(BM model; same cells as Fig 6). Left: coloured by cell type; right: the 8 clusters shown in Fig 6 coloured, other clusters grey.\n"
             "Letters = the tile cells a–h of Fig 6 (nearest the cluster centre in the 50-d PCA space, so not always central in the UMAP).",
             ha="center", va="center", fontsize=4.3, color="#444", linespacing=1.3)
    return fig
