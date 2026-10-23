"""Per-sample attention summaries for the error analysis (supplementary error figures; ~minutes).

Run from the repository root:
    .venv/bin/python scripts/extract_error_attention_cache.py
Writes data/<DATA_VERSION_IDENTIFIER>/derived/error_analysis/sample_attention_summary.parquet.

Per modality × variable × labelled sample: outcome (TP/FP/TN/FN), probability, n cells, N_eff (1/Σa²),
cell-type mix and morphology of the TOP highest-attention cells.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from core.data_loading import RUN_PATHS, SELECTED_VARIABLES, FEAT_DIR, FILES, DERIVED_DIR  # noqa: E402

OUT = DERIVED_DIR / "error_analysis"

TOP = 20
CELL_FEATS = ["apl_bilobed", "apl_hypergranulated", "apl_auer", "vacuolated", "mds_dysmorphic", "low_quality",
              "entire_N:C-ratio", "nucleus_compactness", "nucleus_area", "entire_area"]


def main():
    rows = []
    for mod in ("BM", "PB"):
        cells = pd.read_parquet(FEAT_DIR / FILES[mod]["parquet"], columns=["database_id", "album_id", "cell_id"] + CELL_FEATS)
        for c in ("database_id", "album_id"):
            cells[c] = cells[c].astype(str)
        for k in range(1, 6):
            att = pd.read_csv(REPO / RUN_PATHS[mod] / f"fold_{k}" / "cell_attention_scores.csv")
            for c in ("database_id", "album_id"):
                att[c] = att[c].astype(str)
            att = att.merge(cells, on=["database_id", "album_id", "cell_id"], how="left")
            for v in SELECTED_VARIABLES:
                a = f"cell_attention_score_{v}"
                if a not in att:
                    continue
                d = att[att[f"{v}_true_label"].isin([0, 1]) & att[a].notna()]
                for (bag, db, al), g in d.groupby(["bag_id", "database_id", "album_id"]):
                    w = g[a].to_numpy()
                    top = g.nlargest(TOP, a)
                    y, p = g[f"{v}_true_label"].iat[0], g[f"{v}_pred_label"].iat[0]
                    r = dict(modality=mod, variable=v, fold=k, database_id=db, album_id=al,
                             outcome=("TP" if p == 1 else "FN") if y == 1 else ("FP" if p == 1 else "TN"),
                             prob=g[f"{v}_pred_prob"].iat[0], n_cells=len(g), neff=1 / np.sum(w ** 2),
                             low_quality_all=g["low_quality"].mean())
                    r.update({f"top_ct_{ct}": n / len(top) for ct, n in top.cell_label.value_counts().items()})
                    r.update({f"top_{f}": top[f].mean() for f in CELL_FEATS})
                    rows.append(r)
            print(mod, k, len(rows), flush=True)
    out = pd.DataFrame(rows)
    ct = [c for c in out if c.startswith("top_ct_")]
    out[ct] = out[ct].fillna(0)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "sample_attention_summary.parquet")


if __name__ == "__main__":
    main()
