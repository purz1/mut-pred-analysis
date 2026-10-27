"""VAF of true and missed positives, per modality × variable.

Only mutated patients have a VAF, so the question is whether missed positives (FN) have a lower VAF than detected ones (TP),
and whether the model probability follows the VAF. VAF values of other outcomes (FP / TN), where the table has any, are kept
in the per-sample table. BM and FUSION predictions use the BM VAF table, PB predictions the PB one.
Per modality × variable: n, median [IQR] VAF of TP and FN, AUROC of the VAF for TP vs FN with a 95% bootstrap CI, Mann–Whitney p
(BH-FDR within modality).

Run from the repository root:
    .venv/bin/python scripts/extract_vaf_by_outcome.py [--vaf-bm FILE] [--vaf-pb FILE]
Writes data/<DATA_VERSION_IDENTIFIER>/derived/error_analysis/vaf_by_outcome.csv (one row per sample and variable, no patient
identifiers beyond album_id / database_id) and vaf_outcome_stats.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from config import BM_RUN_PATH, FUSION_RUN_PATH, PB_RUN_PATH, SELECTED_VARIABLES, VAF_DATA_PATH_BM, VAF_DATA_PATH_PB  # noqa: E402
from core.data_loading import DERIVED_DIR  # noqa: E402
from core.utils import add_confusion_categories, load_predictions  # noqa: E402

OUT = DERIVED_DIR / "error_analysis"
MIN_N = 3
N_BOOT = 2000


def auroc(tp, fn):
    """P(VAF of a TP > VAF of an FN), ties 0.5."""
    return mannwhitneyu(tp, fn).statistic / (len(tp) * len(fn))


def boot_ci(tp, fn, rng):
    b = [auroc(rng.choice(tp, len(tp)), rng.choice(fn, len(fn))) for _ in range(N_BOOT)]
    return np.percentile(b, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vaf-bm", default=VAF_DATA_PATH_BM)
    parser.add_argument("--vaf-pb", default=VAF_DATA_PATH_PB)
    args = parser.parse_args()
    runs = {"BM": (BM_RUN_PATH, args.vaf_bm), "PB": (PB_RUN_PATH, args.vaf_pb), "FUSION": (FUSION_RUN_PATH, args.vaf_bm)}
    rng = np.random.default_rng(0)

    samples, stats = [], []
    for modality, (run, vaf_path) in runs.items():
        vaf = pd.read_excel(vaf_path)
        pred = add_confusion_categories(load_predictions(REPO / run, SELECTED_VARIABLES), SELECTED_VARIABLES, keep_original=True)
        for d in (vaf, pred):
            d["album_id"], d["database_id"] = d["album_id"].astype(str), d["database_id"].astype(str)
        rows = []
        for v in SELECTED_VARIABLES:
            if v not in vaf.columns:
                continue
            d = vaf[["album_id", "database_id", v]].rename(columns={v: "vaf"}).merge(
                pred[["album_id", "database_id", f"{v}_predicted", f"{v}_pred_prob"]].rename(
                    columns={f"{v}_predicted": "outcome", f"{v}_pred_prob": "pred_prob"}),
                on=["album_id", "database_id"])
            d = d.dropna(subset=["vaf", "outcome"]).assign(modality=modality, variable=v)
            samples.append(d)
            tp, fn = (d.loc[d.outcome == o, "vaf"].to_numpy(float) for o in ("TP", "FN"))
            row = dict(modality=modality, variable=v, n_tp=len(tp), n_fn=len(fn),
                       median_tp=np.median(tp) if len(tp) else np.nan, median_fn=np.median(fn) if len(fn) else np.nan,
                       q25_tp=np.percentile(tp, 25) if len(tp) else np.nan, q75_tp=np.percentile(tp, 75) if len(tp) else np.nan,
                       q25_fn=np.percentile(fn, 25) if len(fn) else np.nan, q75_fn=np.percentile(fn, 75) if len(fn) else np.nan)
            if len(tp) >= MIN_N and len(fn) >= MIN_N:
                row.update(auroc=auroc(tp, fn), p=mannwhitneyu(tp, fn).pvalue)
                row["auroc_low"], row["auroc_high"] = boot_ci(tp, fn, rng)
            rows.append(row)
        r = pd.DataFrame(rows)
        if "p" in r:
            ok = r.p.notna()
            r.loc[ok, "padj"] = multipletests(r.loc[ok, "p"], method="fdr_bh")[1]
        stats.append(r)
        print(f"{modality}: VAF for {len(rows)} variables")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(samples, ignore_index=True)[["modality", "variable", "album_id", "database_id", "outcome", "pred_prob", "vaf"]].to_csv(
        OUT / "vaf_by_outcome.csv", index=False)
    pd.concat(stats, ignore_index=True).to_csv(OUT / "vaf_outcome_stats.csv", index=False)
    print(f"Written to {OUT}")


if __name__ == "__main__":
    main()
