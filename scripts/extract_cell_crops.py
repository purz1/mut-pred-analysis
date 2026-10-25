"""Cache cell image crops for the manuscript figures (one fixed scale for all cells).

Run from the repository root:
    .venv/bin/python scripts/extract_cell_crops.py LIST.csv [LIST.csv ...]

LIST.csv: one row per cell with modality, database_id, album_id, cell_id (written by the figure / mockup code).
Image name and box come from the MIL h5 (FILES[mod]["h5"]); images are fetched from the image server (IMAGE_SERVER,
must be running, reads the NAS). Cells already cached are skipped.

Window logic: manuscript_figures/crops.py (shared with the cell selection, which skips cells whose window leaves
the field). Scale: BM fields come at two resolutions (2464 × 2056 and 1232 × 1028 px); the morphometry is on the 1232-px scale,
so 2464-px fields are halved (config.CROP_SCALE). PB images are single-cell crops of WIN × WIN px at that scale.
BM crop = WIN × WIN window at the morphometry scale centred on the cell box, i.e. the same scale as PB; cells whose box
exceeds 0.9 × WIN (megakaryocytes, macrophages) get a larger window, resized to WIN (zoom < 1 in the index).
Parts outside the field are filled grey (edge = outside fraction in the index).

Outputs (git-ignored): data/<DATA_VERSION_IDENTIFIER>/derived/cell_crops/
    <mod>/<database_id>/<album_id>/<cell_id>.png
    index.csv   key, modality, image, box (native px), scale, zoom, edge
    missing.csv cells whose image could not be fetched (skipped by the cell selection, crops.usable)
"""
import io
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from core.data_loading import FEAT_DIR, FILES  # noqa: E402
from manuscript_figures import config as C  # noqa: E402
from manuscript_figures.crops import WIN, window  # noqa: E402

IMAGE_SERVER = "http://localhost:5432/api"
FILL = (235, 235, 235)


def fetch(path):
    r = requests.get(f"{IMAGE_SERVER}/{path}", timeout=60)
    r.raise_for_status()
    im = Image.open(io.BytesIO(r.content))
    im.load()
    return im.convert("RGB")


def crop_bm(im, box):
    win, side = window(box, im.width)
    win = [int(round(v)) for v in win]
    canvas = Image.new("RGB", (win[2] - win[0], win[3] - win[1]), FILL)
    inside = (max(win[0], 0), max(win[1], 0), min(win[2], im.width), min(win[3], im.height))
    canvas.paste(im.crop(inside), (inside[0] - win[0], inside[1] - win[1]))
    edge = 1 - (inside[2] - inside[0]) * (inside[3] - inside[1]) / canvas.width / canvas.height
    return canvas.resize((WIN, WIN), Image.LANCZOS), C.CROP_SCALE[im.width], WIN / side, edge


def main(lists):
    cells = pd.concat(pd.read_csv(p, dtype={"database_id": str, "album_id": str}) for p in lists)
    cells = cells.drop_duplicates(["modality", "database_id", "album_id", "cell_id"])
    index_f = C.CROPS / "index.csv"
    index = pd.read_csv(index_f, dtype={"database_id": str, "album_id": str}) if index_f.exists() else pd.DataFrame()
    done = set(index.key) if len(index) else set()
    rows, missing = [], []
    for (mod, db, al), g in cells.groupby(["modality", "database_id", "album_id"]):
        g = g[~(mod + "/" + db + "/" + al + "/" + g.cell_id.astype(str)).isin(done)]
        if not len(g):
            continue
        with h5py.File(FEAT_DIR / FILES[mod]["h5"], "r") as f:
            h = f[db][al]
            pos = {c: i for i, c in enumerate(h["cell_ids"][:])}
            idx = [pos[c] for c in g.cell_id]
            names = h["image_names"][:][idx].astype(str)
            boxes = np.stack([h[k][:][idx].astype(int) for k in ("x0", "x1", "y0", "y1")], axis=1)
        out = C.CROPS / mod / db / al
        out.mkdir(parents=True, exist_ok=True)
        last = (None, None)  # (image name, image): cells sorted by field → one fetch per field
        for (cid, name, box) in sorted(zip(g.cell_id, names, boxes), key=lambda t: t[1]):
            try:
                im = last[1] if last[0] == name else fetch(f"{db}/{al}/{name}")
            except Exception as e:  # missing / truncated image → recorded, skipped by the cell selection
                print(f"skip {mod} {db}/{al}/{cid}: {e}")
                missing.append(dict(modality=mod, database_id=db, album_id=al, cell_id=cid, error=str(e)[:120]))
                continue
            last = (name, im)
            if mod == "BM":
                crop, scale, zoom, edge = crop_bm(im, box)
            else:
                crop, scale, zoom, edge = im.resize((WIN, WIN)) if im.size != (WIN, WIN) else im, 1.0, 1.0, 0.0
            crop.save(out / f"{cid}.png")
            rows.append(dict(key=f"{mod}/{db}/{al}/{cid}", modality=mod, database_id=db, album_id=al, cell_id=cid,
                             image=name, x0=box[0], x1=box[1], y0=box[2], y1=box[3], scale=scale, zoom=round(zoom, 3),
                             edge=round(edge, 3)))
        print(f"{mod} {db}/{al}: {len(g)} cells")
    if rows:
        pd.concat([index, pd.DataFrame(rows)]).to_csv(index_f, index=False)
    if missing:
        f = C.CROPS / "missing.csv"
        old = pd.read_csv(f, dtype={"database_id": str, "album_id": str}) if f.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(missing)]).drop_duplicates(["modality", "database_id", "album_id", "cell_id"]) \
            .to_csv(f, index=False)
    print(f"{len(rows)} new crops → {C.CROPS}")


if __name__ == "__main__":
    main(sys.argv[1:])
