"""Join the FAB table to the samples of the modelling cohort and drop the patient identifiers.

Run where the patient-level files are available (the academic environment); the analysis then runs anywhere on the output.
The FAB table has one row per patient: patient id (`henkilotunnus`) and a FAB class column (default `FAB`). The clinical tables link the
patient id to the samples (album_id, year → database_id): BM clinical for the BM and FUSION models, PB clinical for PB.
FAB values are normalised to M0 … M7 (case, spaces and sub-types such as M4eo are reduced to the class number; the original value is kept
in `fab_raw`). Patients with several different FAB values are listed and set to missing.

    .venv/bin/python scripts/prepare_fab_table.py [--fab-file FILE] [--id-col henkilotunnus] [--fab-col FAB]
Writes data/<DATA_VERSION_IDENTIFIER>/derived/fab/fab_by_sample.csv (modality, album_id, database_id, fab, fab_raw).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from config import CLINICAL_DATA_PATH_BM, CLINICAL_DATA_PATH_PB, FAB_DATA_PATH  # noqa: E402
from core.data_loading import DERIVED_DIR  # noqa: E402

OUT = DERIVED_DIR / "fab"


def read(path):
    path = Path(path)
    return pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)


def normalise(fab):
    """M0 … M7 from free text; NaN where no class number is found."""
    return fab.astype(str).str.upper().str.extract(r"M\s*(\d)")[0].radd("M").where(fab.notna())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fab-file", default=FAB_DATA_PATH)
    parser.add_argument("--id-col", default="henkilotunnus")
    parser.add_argument("--fab-col", default="FAB")
    args = parser.parse_args()

    fab = read(args.fab_file).rename(columns={args.id_col: "henkilotunnus", args.fab_col: "fab_raw"})[["henkilotunnus", "fab_raw"]]
    fab = fab.dropna(subset=["fab_raw"]).drop_duplicates()
    fab["fab"] = normalise(fab["fab_raw"])
    multi = fab.groupby("henkilotunnus").fab_raw.nunique()
    if (multi > 1).any():
        print(f"{(multi > 1).sum()} patients with several FAB values: set to missing")
        fab = fab[~fab.henkilotunnus.isin(multi[multi > 1].index)]
    print(f"FAB table: {fab.henkilotunnus.nunique()} patients; unreadable values: {fab.fab.isna().sum()}")

    out = []
    for modality, path in (("BM", CLINICAL_DATA_PATH_BM), ("PB", CLINICAL_DATA_PATH_PB)):
        clin = pd.read_csv(path)[["henkilotunnus", "album_id", "year"]].rename(columns={"year": "database_id"})
        d = clin.merge(fab, on="henkilotunnus", how="left")
        d[["album_id", "database_id"]] = d[["album_id", "database_id"]].astype(str)
        print(f"{modality}: {len(d)} samples, FAB for {d.fab.notna().sum()}")
        out.append(d.assign(modality=modality)[["modality", "album_id", "database_id", "fab", "fab_raw"]])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(out).to_csv(OUT / "fab_by_sample.csv", index=False)
    print(f"Written to {OUT / 'fab_by_sample.csv'}")


if __name__ == "__main__":
    main()
