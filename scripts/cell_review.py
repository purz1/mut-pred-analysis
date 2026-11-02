"""Manual cell review: copy the crops of a top-attention cell table into a review folder, delete the cells you don't
want, collect the rest into a cell list (e.g. training cells for a cell classifier).

Run from the repository root:
    .venv/bin/python scripts/cell_review.py export NAME [--out DIR]
    .venv/bin/python scripts/cell_review.py collect DIR [DIR ...] [--out FILE]

export  NAME = a table written by scripts/extract_top_attention_cells.py (derived/top_attention_cells/NAME.csv).
        Crops not yet cached are fetched first (scripts/extract_cell_crops.py, the image server must be running).
        Copies them to DIR (default derived/cell_review/NAME/), one flat folder per modality:
            <modality>/<database_id>__<album_id>__r<rank>__<cell_id>.png   (sorted by sample, then attention rank)
        and DIR/manifest.csv (file → cell, with the table's columns). Re-running restores deleted images.
collect Cells whose image is still in DIR (any number of review folders) → FILE (default DIR/selected.csv for one
        folder, required for several):
            slide,cell_id,sample_type
            VisionDB_2010/1234,5678,BM
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from manuscript_figures import config as C  # noqa: E402
from manuscript_figures.cells import crop_path  # noqa: E402

REVIEW = C.DERIVED / "cell_review"
DTYPE = {"database_id": str, "album_id": str}


def export(name, out):
    out = Path(out) if out else REVIEW / name
    cells = pd.read_csv(C.DERIVED / "top_attention_cells" / f"{name}.csv", dtype=DTYPE)
    src = [crop_path(r, r.modality) for r in cells.itertuples()]
    if not all(p.exists() for p in src):
        sys.path.insert(0, str(REPO / "scripts"))
        import extract_cell_crops  # noqa: E402
        try:  # unreachable server would mark every cell missing (missing.csv, used by the figure cell selection)
            extract_cell_crops.requests.get(extract_cell_crops.IMAGE_SERVER, timeout=5)
        except extract_cell_crops.requests.ConnectionError:
            sys.exit(f"image server not reachable ({extract_cell_crops.IMAGE_SERVER}); start it and re-run")
        extract_cell_crops.main([C.CROPS / "lists" / f"{name}.csv"])
    cells["file"] = [f"{r.modality}/{r.database_id}__{r.album_id}__r{r.rank:02d}__{r.cell_id}.png"
                     for r in cells.itertuples()]
    for d in cells.modality.unique():
        (out / d).mkdir(parents=True, exist_ok=True)
    ok = [p.exists() for p in src]
    for p, f in zip((p for p, k in zip(src, ok) if k), cells.file[ok]):
        shutil.copy2(p, out / f)
    cells[ok].to_csv(out / "manifest.csv", index=False)
    print(f"{sum(ok)} images → {out}" + (f" ({len(ok) - sum(ok)} without image, see {C.CROPS / 'missing.csv'})"
                                        if not all(ok) else ""))


def collect(dirs, out):
    if not out and len(dirs) > 1:
        sys.exit("--out is required with several review folders")
    sel = []
    for d in map(Path, dirs):
        m = pd.read_csv(d / "manifest.csv", dtype=DTYPE)
        k = m[[(d / f).exists() for f in m.file]]
        print(f"{d}: {len(k)} of {len(m)} cells kept")
        sel.append(k)
    sel = pd.concat(sel).drop_duplicates(["modality", "database_id", "album_id", "cell_id"])
    out = Path(out) if out else Path(dirs[0]) / "selected.csv"
    pd.DataFrame({"slide": sel.database_id + "/" + sel.album_id, "cell_id": sel.cell_id,
                  "sample_type": sel.modality}).to_csv(out, index=False)
    print(f"{len(sel)} cells → {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("name")
    e.add_argument("--out")
    c = sub.add_parser("collect")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--out")
    args = parser.parse_args()
    export(args.name, args.out) if args.cmd == "export" else collect(args.dirs, args.out)


if __name__ == "__main__":
    main()
