"""Is each dysplasia class reflected in cell morphometry? (cell-level, within-patient)

Run from the repository root:
    .venv/bin/python scripts/extract_morphometry_dysplasia_correlation.py

For every (cell type, dysplasia flag) pair that is used in the figures, compare the morphometry of
flagged vs unflagged cells of that type *within each sample* (Cliff's delta), so that patient-level
differences do not drive the result. Samples need >= MIN_CELLS flagged and >= MIN_CELLS unflagged cells.
Per pair x feature we report the median within-sample delta, the fraction of samples with delta > 0,
and a Wilcoxon signed-rank test of the per-sample deltas against 0 (BH-FDR over all tests per modality).

Cell filters mirror core/cell_feature_aggregation: morphometry uses ~quality_model_filtered,
dysplasia uses low_quality < 0.5 and ~mitotic.

Outputs (git-ignored): data/<DATA_VERSION_IDENTIFIER>/derived/morph_dysplasia/
    morph_dysplasia_effects.csv   one row per modality x cell type x flag x feature
    run_info.json
TODO(refactor): share cell filters / feature lists with core/cell_feature_aggregation.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from config import DATA_VERSION_IDENTIFIER, TRAINING_FEATURES_DIR  # noqa: E402

MIN_CELLS = 5          # per sample, in each of flagged / unflagged
MIN_SAMPLES = 10       # samples needed to report a pair
FEAT_DIR = TRAINING_FEATURES_DIR
OUT_DIR = REPO / f"data/{DATA_VERSION_IDENTIFIER}/derived/morph_dysplasia"
PARQUET = {"BM": "cell_level_features_aml_dg_de-novo-and-secondary_bm_07-04-2026.parquet",
           "PB": "cell_level_features_aml_dg_de-novo-and-secondary_pb_07-04-2026.parquet"}
FEATURES = ["entire_area", "nucleus_area", "entire_N:C-ratio", "nucleus_solidity", "nucleus_eccentricity",
            "nucleus_compactness", "entire_compactness"]

# (cell type(s), flag) pairs as used in the dysplasia panels (see core/cell_feature_aggregation/constants)
PAIRS = {
    "BM": [("Blasts", f) for f in ["apl_hypergranulated", "apl_bilobed", "apl_auer", "vacuolated"]]
          + [("Promyelocytes", f) for f in ["apl_hypergranulated", "apl_bilobed", "apl_auer", "vacuolated"]]
          + [("Myelocytes", f) for f in ["apl_hypergranulated", "vacuolated"]]
          + [("Erythroblasts", f) for f in ["mds_asynchrony", "mds_dysmorphic", "mds_multinucleated", "vacuolated"]]
          + [("Megakaryocytes", f) for f in ["mgk_multinucleated", "mgk_hypolobated"]]
          + [(ct, "vacuolated") for ct in ["Neutrophils", "Eosinophils", "Monocytes", "Lymphocytes"]],
    "PB": [("Blasts", f) for f in ["apl_hypergranulated", "vacuolated"]]
          + [("Myelocytes", "apl_hypergranulated")]
          + [("Neutrophils", f) for f in ["neut_gran_hypo", "neut_gran_hyper", "neut_segm_hypo", "neut_segm_hyper"]]
          + [(ct, "vacuolated") for ct in ["Monocytes", "Lymphocytes"]],
}
CT_MEMBERS = {"Neutrophils_PB": ["Band_neutrophils", "Segmented_neutrophils"]}


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for mod, pairs in PAIRS.items():
        cells = pd.read_parquet(FEAT_DIR / PARQUET[mod])
        cells = cells[~cells["quality_model_filtered"] & (cells["low_quality"] < 0.5) & ~cells["mitotic"]]
        for ct, flag in pairs:
            members = CT_MEMBERS.get(f"{ct}_{mod}", [ct])
            sub = cells[cells["class_label"].isin(members)].dropna(subset=FEATURES)
            deltas = []
            for _, g in sub.groupby(["database_id", "album_id"], sort=False):
                pos, neg = g[g[flag]], g[~g[flag]]
                if len(pos) < MIN_CELLS or len(neg) < MIN_CELLS:
                    continue
                u, _ = mannwhitneyu(pos[FEATURES].to_numpy(float), neg[FEATURES].to_numpy(float), axis=0)
                deltas.append(2 * u / (len(pos) * len(neg)) - 1)
            n = len(deltas)
            d = np.array(deltas) if n else np.empty((0, len(FEATURES)))
            for j, f in enumerate(FEATURES):
                p = wilcoxon(d[:, j]).pvalue if n >= MIN_SAMPLES and np.any(d[:, j] != 0) else np.nan
                rows.append(dict(modality=mod, cell_type=ct, flag=flag, feature=f, n_samples=n,
                                 flagged_cell_fraction=float(sub[flag].mean()) if len(sub) else np.nan,
                                 median_delta=float(np.median(d[:, j])) if n else np.nan,
                                 frac_samples_positive=float((d[:, j] > 0).mean()) if n else np.nan,
                                 p_wilcoxon=p))
            print(f"[{mod}] {ct:15s} {flag:20s} n_samples={n:4d}  ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows)
    res["padj"] = np.nan
    for mod in res.modality.unique():
        m = (res.modality == mod) & res.p_wilcoxon.notna()
        res.loc[m, "padj"] = multipletests(res.loc[m, "p_wilcoxon"], method="fdr_bh")[1]
    res.to_csv(OUT_DIR / "morph_dysplasia_effects.csv", index=False)
    json.dump(dict(created=datetime.now().isoformat(timespec="seconds"), data_version=DATA_VERSION_IDENTIFIER,
                   min_cells_per_group=MIN_CELLS, min_samples=MIN_SAMPLES, features=FEATURES, pairs=PAIRS,
                   method="within-sample Cliff's delta (flagged vs unflagged cells of the cell type); median across "
                          "samples; Wilcoxon signed-rank vs 0; BH-FDR per modality",
                   cell_filters="~quality_model_filtered & low_quality < 0.5 & ~mitotic",
                   runtime_s=round(time.time() - t0)),
              open(OUT_DIR / "run_info.json", "w"), indent=2)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
