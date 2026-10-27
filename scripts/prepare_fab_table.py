"""Link the FAB table to the samples of the modelling cohort and drop the patient identifiers.

Run where the patient-level files are available (the academic environment); the analysis then runs anywhere on the output.
The FAB table has one row per patient: patient id (`henkilotunnus`) and FAB (M0 … M7). The clinical tables link the patient id to the
samples (album_id, year → database_id): BM clinical for the BM and FUSION models, PB clinical for PB.

    .venv/bin/python scripts/prepare_fab_table.py [--fab-file FILE]
Writes data/<DATA_VERSION_IDENTIFIER>/derived/fab/fab_by_sample.csv (modality, album_id, database_id, fab).
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fab-file", default=FAB_DATA_PATH)
    args = parser.parse_args()

    path = Path(args.fab_file)
    fab = (pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path))[["henkilotunnus", "FAB"]]
    fab = fab.rename(columns={"FAB": "fab"}).dropna().drop_duplicates()
    assert fab.henkilotunnus.is_unique, "patients with several FAB values"

    out = []
    for modality, clinical in (("BM", CLINICAL_DATA_PATH_BM), ("PB", CLINICAL_DATA_PATH_PB)):
        clin = pd.read_csv(clinical)[["henkilotunnus", "album_id", "year"]].rename(columns={"year": "database_id"})
        d = clin.merge(fab, on="henkilotunnus", how="left").astype({"album_id": str, "database_id": str})
        print(f"{modality}: {len(d)} samples, FAB for {d.fab.notna().sum()}")
        out.append(d.assign(modality=modality)[["modality", "album_id", "database_id", "fab"]])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(out).to_csv(OUT / "fab_by_sample.csv", index=False)
    print(f"Written to {OUT / 'fab_by_sample.csv'}")


if __name__ == "__main__":
    main()
