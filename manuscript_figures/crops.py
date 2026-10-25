"""Cell-crop window shared by scripts/extract_cell_crops.py and the cell selection of the figures.

BM fields: 2464 × 2056 or 1232 × 1028 px, inferred per sample from the largest box coordinate (clean split, 391 / 175
samples); crops are WIN × WIN px at the morphometry (1232-px) scale, larger for cells whose box exceeds 0.9 × WIN.
PB images are single-cell crops (always "in field").
"""
from functools import cache

import h5py
import numpy as np
import pandas as pd

from core.data_loading import FEAT_DIR, FILES
from .config import CROP_SCALE, CROPS

WIN = 234  # px at the morphometry scale (= PB crop size); fits ~98% of BM cells
FIELD = {2464: (2464, 2056), 1232: (1232, 1028)}
BAD_SAMPLE = 5  # samples with ≥ 5 failed images are skipped entirely (7 PB samples of VisionDB_2022_veri serve no images)


def window(box, width):
    """(x0, y0, x1, y1) native window, side at the morphometry scale; box = (x0, x1, y0, y1) native px."""
    x0, x1, y0, y1 = box
    scale = CROP_SCALE[width]
    side = np.maximum(WIN, np.maximum(x1 - x0, y1 - y0) * scale / 0.9)  # scalars or arrays
    half = side / scale / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - half, cy - half, cx + half, cy + half), side


@cache
def in_field(mod):
    """database_id, album_id, cell_id, in_field (crop window fully inside the field image)."""
    rows = []
    with h5py.File(FEAT_DIR / FILES[mod]["h5"], "r") as f:
        for d in f:
            for a in f[d]:
                h = f[d][a]
                b = np.stack([h[k][:].astype(float) for k in ("x0", "x1", "y0", "y1")], axis=1)
                ok = np.ones(len(b), bool)
                if mod == "BM":
                    w = 2464 if b[:, 1].max() > 1232 else 1232
                    (x0, y0, x1, y1), _ = window(b.T, w)
                    ok = (x0 >= 0) & (y0 >= 0) & (x1 <= FIELD[w][0]) & (y1 <= FIELD[w][1])
                rows.append(pd.DataFrame(dict(database_id=d, album_id=a, cell_id=h["cell_ids"][:], in_field=ok)))
    return pd.concat(rows, ignore_index=True)


def usable(mod):
    """in_field(mod) with in_field False also for cells whose image could not be fetched (CROPS/missing.csv) and for
    all cells of samples with ≥ BAD_SAMPLE failed images."""
    d = in_field(mod)
    f = CROPS / "missing.csv"
    if f.exists():
        m = pd.read_csv(f, dtype={"database_id": str, "album_id": str})
        m = m[m.modality == mod]
        bad = m.groupby(["database_id", "album_id"]).size().loc[lambda n: n >= BAD_SAMPLE].index  # unreadable samples
        m = m.assign(missing=True)[["database_id", "album_id", "cell_id", "missing"]]
        d = d.merge(m, how="left")
        d = d.assign(in_field=d.in_field & d.missing.isna()
                     & ~d.set_index(["database_id", "album_id"]).index.isin(bad)).drop(columns="missing")
    return d


@cache
def field_width(mod="BM"):
    """database_id, album_id, width: BM field width (2464 or 1232 px = scanner / resolution group), from the largest box x."""
    rows = []
    with h5py.File(FEAT_DIR / FILES[mod]["h5"], "r") as f:
        for d in f:
            for a in f[d]:
                rows.append((d, a, 2464 if f[d][a]["x1"][:].max() > 1232 else 1232))
    return pd.DataFrame(rows, columns=["database_id", "album_id", "width"])
