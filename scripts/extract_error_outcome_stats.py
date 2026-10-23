"""Morphological association with outcome group (TN / FN / FP / TP), per modality × variable × feature.

Contrasts within the true class (the class difference itself is Fig 4):
  FN vs TP — do missed positives lack the positive phenotype?
  FP vs TN — do false alarms carry it?
Per feature: Cliff's δ + Mann–Whitney p, BH-FDR within modality × variable × feature type × contrast (same family
as Fig 4/5, scripts/extract_fig4_fingerprint_results.py), and the position of the FN / FP median on the TN→TP axis (0 = TN median, 1 = TP median).
QC rows (cells / sample, low-quality %) are included.

Run from the repository root, after scripts/extract_error_attention_cache.py:
    .venv/bin/python scripts/extract_error_outcome_stats.py
Writes data/<DATA_VERSION_IDENTIFIER>/derived/error_analysis/morph_outcome_stats.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from core.data_loading import load_cohort, DERIVED_DIR  # noqa: E402

OUT = DERIVED_DIR / "error_analysis"

MIN_N = 3
CONTRASTS = {"FN_vs_TP": ("FN", "TP"), "FP_vs_TN": ("FP", "TN"), "TP_vs_TN": ("TP", "TN"), "FN_vs_TN": ("FN", "TN"), "FP_vs_TP": ("FP", "TP")}


def cliffs(a, b):
    if len(a) < MIN_N or len(b) < MIN_N:
        return np.nan, np.nan
    u, p = mannwhitneyu(a, b)
    return 2 * u / (len(a) * len(b)) - 1, p


def main():
    S = pd.read_parquet(OUT / "sample_attention_summary.parquet")
    eff = pd.read_csv(DERIVED_DIR / "fig4_fingerprints" / "feature_effects.csv")
    rows = []
    for mod in ("BM", "PB"):
        df = load_cohort(mod)
        feats = eff[eff.modality == mod].drop_duplicates("feature_col")[["feature_col", "feature_type", "cell_type", "feature"]]
        qc = pd.DataFrame({"feature_col": ["n_cells", "low_quality_all"], "feature_type": "qc",
                           "cell_type": "", "feature": ["cells / sample", "low-quality cells"]})
        feats = pd.concat([feats, qc])
        for v, d in S[S.modality == mod].groupby("variable"):
            d = d.merge(df[["database_id", "album_id"] + feats.feature_col[:-2].tolist()], on=["database_id", "album_id"])
            pos = eff[(eff.modality == mod) & (eff.variable == v)].set_index("feature_col")
            for f in feats.itertuples():
                g = {o: d.loc[d.outcome == o, f.feature_col].dropna().to_numpy() for o in ("TN", "FN", "FP", "TP")}
                med = {o: np.median(x) if len(x) else np.nan for o, x in g.items()}
                span = med["TP"] - med["TN"]
                r = dict(modality=mod, variable=v, **f._asdict(), **{f"n_{o}": len(x) for o, x in g.items()},
                         **{f"median_{o}": m for o, m in med.items()},
                         **{f"mean_{o}": x.mean() if len(x) else np.nan for o, x in g.items()},
                         delta_pos_vs_neg=pos.delta_pos_vs_neg.get(f.feature_col, np.nan),
                         padj_pos_vs_neg=pos.padj_pos_vs_neg.get(f.feature_col, np.nan))
                for c, (a, b) in CONTRASTS.items():
                    r[f"delta_{c}"], r[f"p_{c}"] = cliffs(g[a], g[b])
                    r[f"pos_{a}"] = (med[a] - med["TN"]) / span if span else np.nan
                rows.append(r)
    out = pd.DataFrame(rows).drop(columns="Index")
    for c in CONTRASTS:
        out[f"padj_{c}"] = np.nan
        for _, d in out.groupby(["modality", "variable", "feature_type"]):
            ok = d[f"p_{c}"].notna()
            if ok.any():
                out.loc[d.index[ok], f"padj_{c}"] = multipletests(d.loc[ok, f"p_{c}"], method="fdr_bh")[1]
    out.to_csv(OUT / "morph_outcome_stats.csv", index=False)
    for c in CONTRASTS:
        sig = out[out[f"padj_{c}"] < 0.05]
        print(f"\n{c}: {len(sig)} significant (FDR<0.05), {(out[f'p_{c}'] < 0.05).sum()} nominal")
        print(sig.groupby(["modality", "variable"]).size().to_string())


if __name__ == "__main__":
    main()
