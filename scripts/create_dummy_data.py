#!/usr/bin/env python3
"""Generate synthetic clinical data files for local testing.

Produces data/dummy/ files that mimic the structure of real patient data.
Patient IDs are sampled from scemila_fusion_v15_predictions.csv so that
merge joins between clinical and prediction data work correctly.

Usage:
    python scripts/create_dummy_data.py
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N = 30  # synthetic patients

SELECTED_VARIABLES = [
    "PML-RARA", "CBFB-MYH11", "RUNX1-RUNX1T1_OR_t_8_21",
    "IDH_mutations", "IDH2", "CpG_mutations",
    "NPM1", "NPM1_AND_FLT3_ITD",
    "FLT3_mutations", "FLT3_ITD", "KIT", "RTK_RAS_mutations",
    "complex_karyotype", "monosomal_karyotype", "monosomy5_del_5q",
    "TP53", "STAG2",
]

# VAF analysis uses this subset
VAF_VARIABLES = [
    "CBFB-MYH11", "IDH2", "IDH_mutations", "KIT", "NPM1",
    "PML-RARA", "RTK_RAS_mutations", "STAG2", "TP53", "monosomy5_del_5q",
]


def main():
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    out = Path("data/dummy")
    out.mkdir(parents=True, exist_ok=True)

    # Sample real album_ids from predictions CSV so joins work
    preds_path = Path("data/scemila_fusion_v15_predictions.csv")
    if preds_path.exists():
        preds = pd.read_csv(preds_path)
        album_ids = preds["album_id"].dropna().astype(str).unique()
        chosen = pd.Series(album_ids).sample(min(N, len(album_ids)), random_state=SEED).reset_index(drop=True)
        database_ids = preds.set_index("album_id")["database_id"].to_dict()
    else:
        chosen = pd.Series([str(i) for i in range(1000, 1000 + N)])
        database_ids = {}

    n = len(chosen)
    hetu = [f"DUMMY{i:04d}" for i in range(n)]
    db_ids = [database_ids.get(aid, 2022) for aid in chosen]

    base = pd.DataFrame({
        "album_id": chosen,
        "database_id": db_ids,
        "henkilotunnus": hetu,
    })

    # ── Survival data ──────────────────────────────────────────────────────────
    treatments = rng.choice(["cyto", "aza", "aza_ven", "palliative"], size=n, p=[0.55, 0.2, 0.15, 0.1])
    surv = base.copy()
    surv["time_dg_to_death"] = rng.exponential(scale=600, size=n).round(1)
    surv["died"] = rng.random(size=n) < 0.45
    surv["treatment"] = treatments
    surv["cyto"] = surv["treatment"] == "cyto"

    surv.to_excel(out / "survival_data_bm.xlsx", index=False)
    surv.to_excel(out / "survival_data_pb.xlsx", index=False)
    print("Wrote survival_data_bm.xlsx and survival_data_pb.xlsx")

    # ── VAF data ───────────────────────────────────────────────────────────────
    vaf = base.copy()
    for var in VAF_VARIABLES:
        # ~30% of samples are positive; VAF for positives 10-80%, negatives NaN
        positive = rng.random(size=n) < 0.3
        values = np.where(positive, rng.uniform(10, 80, size=n), np.nan)
        vaf[var] = np.round(values, 1)

    vaf.to_excel(out / "vaf_values_bm.xlsx", index=False)
    vaf.to_excel(out / "vaf_values_pb.xlsx", index=False)
    print("Wrote vaf_values_bm.xlsx and vaf_values_pb.xlsx")

    # ── Clinical data (ELN analysis) ───────────────────────────────────────────
    eln_cats = rng.choice(["Favorable", "Intermediate", "Adverse"], size=n, p=[0.3, 0.4, 0.3])
    clin = base.rename(columns={"database_id": "year"}).copy()
    clin["age"] = rng.integers(30, 85, size=n)
    clin["ELN1"] = eln_cats

    # Add binary mutation columns matching SELECTED_VARIABLES
    for var in SELECTED_VARIABLES:
        clin[var] = (rng.random(size=n) < 0.2).astype(int)

    clin.to_csv(out / "clinical_data_bm.csv", index=False)
    print("Wrote clinical_data_bm.csv")

    print(f"\nDone — {n} synthetic patients in {out}/")


if __name__ == "__main__":
    main()
