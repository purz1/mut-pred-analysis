"""Extract the cell tables of the cell-image figures once (Fig 6 gallery, variable cards, TP/FP pages).

Run from the repository root:
    .venv/bin/python scripts/extract_gallery_cells.py [--skip-top]     # --skip-top: reuse top_cells.parquet
    .venv/bin/python scripts/extract_gallery_cells.py --method weighted|medoid   # alternative Fig 6 tile selection (gallery.METHODS)

Outputs (git-ignored), data/<DATA_VERSION_IDENTIFIER>/derived/gallery/:
    top_cells.parquet      per modality × card target (cards.CARD_VARS): the top-attention cells (attention-mass threshold
                           0.5, core.attention.select_top_attention_cell_ids) of all samples of the modality's model, with
                           attention score att and w = att / total attention of the sample
    cluster_cells.parquet  per gallery target (gallery.TARGETS, BM): the top-attention cells of the TP samples with
                           cell_label, PCA(50)+k-means cluster, distance d to the cluster centre, 2D UMAP (u0, u1), scanner
    gallery_tiles.csv      the Fig 6 tiles: cell, cluster, attention share, support, sliders shown (see gallery.py)
    gallery_clusters.csv   all clusters per target: share, support, dominant label, low-quality / old-scanner shares
    run_info.json
Also writes derived/cell_crops/lists/gallery.csv; fetch the images with
    .venv/bin/python scripts/extract_cell_crops.py data/<DATA_VERSION_IDENTIFIER>/derived/cell_crops/lists/*.csv
(the card and TP/FP page functions write their own lists when run).
"""
import glob
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from core.attention import select_top_attention_cell_ids  # noqa: E402
from core.data_loading import RUN_PATHS  # noqa: E402
from manuscript_figures import cards, cells, config as C, gallery  # noqa: E402

SAMPLE = ["database_id", "album_id"]
KEY = SAMPLE + ["cell_id"]


def attention(mod):
    """Cell attention scores of the modality's model: every sample is a test sample of exactly one fold."""
    keep = lambda c: c in set(KEY) or c.startswith("cell_attention_score_")  # noqa: E731
    return pd.concat(pd.read_csv(f, usecols=keep).astype({"database_id": str, "album_id": str})
                     for f in sorted(glob.glob(f"{REPO / RUN_PATHS[mod]}/fold_*/cell_attention_scores.csv")))


def top_cells_table():
    out = []
    for mod in ("BM", "PB"):
        a = attention(mod)
        for var in cards.CARD_VARS:
            t0 = time.time()
            sc = f"cell_attention_score_{var}"
            tot = a.groupby(SAMPLE)[sc].sum().rename("tot").reset_index()
            t = select_top_attention_cell_ids(a, var).merge(a[KEY + [sc]], on=KEY).merge(tot, on=SAMPLE)
            out.append(t.assign(modality=mod, variable=var, w=t[sc] / t.tot).rename(columns={sc: "att"})
                       [["modality", "variable"] + KEY + ["att", "w"]])
            print(f"[{mod}] {var:26s} {len(t):7d} top cells  ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(out, ignore_index=True)


def alternative(method):
    """Fig 6 tiles with an alternative selection method (gallery.METHODS) → gallery_tiles_<method>.csv etc."""
    tiles, clusters = [], []
    for var in gallery.TARGETS:
        t0 = time.time()
        _, tt, ct = gallery.build(var, method=method)
        tiles.append(tt); clusters.append(ct)
        print(f"{var:26s} tiles {''.join(tt.tile)} ({time.time() - t0:.0f}s)", flush=True)
    tiles = pd.concat(tiles, ignore_index=True)
    sfx = gallery.suffix(method)
    tiles.to_csv(C.GALLERY / f"gallery_tiles{sfx}.csv", index=False)
    pd.concat(clusters, ignore_index=True).to_csv(C.GALLERY / f"gallery_clusters{sfx}.csv", index=False)
    cells.write_list([(gallery.MOD, tiles)], f"gallery{sfx}")
    print("written to", C.GALLERY)


def main(skip_top):
    C.GALLERY.mkdir(parents=True, exist_ok=True)
    if not (skip_top and C.TOP_CELLS.exists()):
        top_cells_table().to_parquet(C.TOP_CELLS, index=False, row_group_size=500_000)
    pcs, tiles, clusters = [], [], []
    for var in gallery.TARGETS:
        t0 = time.time()
        p, tt, ct = gallery.build(var)
        pcs.append(p); tiles.append(tt); clusters.append(ct)
        print(f"{var:26s} {len(p):6d} pooled cells, tiles {''.join(tt.tile)} ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(pcs, ignore_index=True).to_parquet(C.GALLERY / "cluster_cells.parquet", index=False)
    tiles = pd.concat(tiles, ignore_index=True)
    tiles.to_csv(C.GALLERY / "gallery_tiles.csv", index=False)
    pd.concat(clusters, ignore_index=True).to_csv(C.GALLERY / "gallery_clusters.csv", index=False)
    cells.write_list([(gallery.MOD, tiles)], "gallery")
    json.dump(dict(created=datetime.now().isoformat(timespec="seconds"), run=RUN_PATHS, targets=gallery.TARGETS,
                   card_targets=cards.CARD_VARS, k=gallery.K, pca=gallery.N_PCA, seed=gallery.SEED,
                   support=gallery.SUPPORT, tiles=gallery.N_TILES, max_sliders=gallery.MAX_SL),
              open(C.GALLERY / "run_info.json", "w"), indent=1)
    print("written to", C.GALLERY)


if __name__ == "__main__":
    if "--method" in sys.argv:
        alternative(sys.argv[sys.argv.index("--method") + 1])
    else:
        main("--skip-top" in sys.argv)
