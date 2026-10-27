"""Co-occurrence of the targets in the true labels, and whether the model reproduces it.

pair_counts.csv       every pair of targets: n11 / n10 / n01 / n00 (X+Y+, X+Y−, X−Y+, X−Y−) on the samples labelled for both, phi of the true labels,
                      Haldane odds ratio, `nested` (one target contains or defines the other; trivial co-occurrence, see manuscript_figures/errors.py)
pair_model.csv        per model (BM, PB, FUSION) and pair: phi of the predicted labels, Spearman correlation of the predicted probabilities, and the
                      difference of the predicted-label phi to the true-label phi (positive = the model couples the two targets more than the data do)
Run from the repository root:
    .venv/bin/python scripts/extract_cooccurrence.py
Writes data/<DATA_VERSION_IDENTIFIER>/derived/cooccurrence/.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from config import BM_RUN_PATH, FUSION_RUN_PATH, PB_RUN_PATH, SELECTED_VARIABLES  # noqa: E402
from core.data_loading import DERIVED_DIR  # noqa: E402
from manuscript_figures.errors import NESTED  # noqa: E402

OUT = DERIVED_DIR / "cooccurrence"
RUNS = {"BM": BM_RUN_PATH, "PB": PB_RUN_PATH, "FUSION": FUSION_RUN_PATH}


def phi(a, b):
    """Pearson correlation of two binary vectors (NaN if either is constant)."""
    return np.nan if a.std() == 0 or b.std() == 0 else np.corrcoef(a, b)[0, 1]


def main():
    preds = {m: pd.read_csv(REPO / run / "predictions.csv") for m, run in RUNS.items()}
    truth = preds["BM"]
    rows, model_rows = [], []
    for x, y in combinations(SELECTED_VARIABLES, 2):
        t = truth[[f"{x}_true_label", f"{y}_true_label"]].dropna().astype(int)
        a, b = t.iloc[:, 0].to_numpy(), t.iloc[:, 1].to_numpy()
        n11, n10, n01, n00 = int(((a == 1) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum())
        rows.append(dict(x=x, y=y, n=len(t), n11=n11, n10=n10, n01=n01, n00=n00, phi=phi(a, b),
                         odds_ratio=(n11 + .5) * (n00 + .5) / ((n10 + .5) * (n01 + .5)), nested={x, y} in NESTED))
        for m, p in preds.items():
            keep = p[[f"{x}_true_label", f"{y}_true_label"]].notna().all(axis=1)
            q = p[keep]
            pl = phi(q[f"{x}_pred_label"].to_numpy(int), q[f"{y}_pred_label"].to_numpy(int))
            model_rows.append(dict(model=m, x=x, y=y, phi_true=rows[-1]["phi"], phi_pred=pl, phi_diff=pl - rows[-1]["phi"],
                                   rho_prob=spearmanr(q[f"{x}_pred_prob"], q[f"{y}_pred_prob"])[0]))
    OUT.mkdir(parents=True, exist_ok=True)
    pc, pm = pd.DataFrame(rows), pd.DataFrame(model_rows)
    pc.to_csv(OUT / "pair_counts.csv", index=False)
    pm.to_csv(OUT / "pair_model.csv", index=False)
    print(f"{len(pc)} pairs ({int(pc.nested.sum())} nested). Written to {OUT}")


if __name__ == "__main__":
    main()
