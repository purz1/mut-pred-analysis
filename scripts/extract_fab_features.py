"""Does the morphological fingerprint of a target depend on the FAB class? Positive vs negative samples within FAB groups.

For every modality × target × sample-level feature (the Fig 4/5 features, scripts/extract_fig4_fingerprint_results.py) and each FAB group
(GROUPS of extract_fab_analysis.py):
  delta_<group>   Cliff's δ of positive vs negative samples within the group (positive = positive samples larger), n_pos / n_neg
  delta_diff      δ(CONTRAST[0]) − δ(CONTRAST[1]); p_interaction: z-test of the difference of the two AUCs (= (δ + 1) / 2) with DeLong variances
                  (the groups are disjoint patients), BH-FDR within modality × target × feature type
  p_stratified    van Elteren stratified Wilcoxon test: association of the feature with the target adjusted for FAB group (strata = groups
                  with both classes), BH-FDR as above
  delta_overall   δ of the unstratified fingerprint (feature_effects.csv) for reference
  hypothesis      pre-specified: nuclear shape of blasts for NPM1 (cup-shaped nuclei of NPM1-mutated blasts are described in M1 but not in
                  M5/M6; Rastogi et al.): HYPOTHESIS features, contrast CONTRAST
A group is tested for a feature only with at least MIN_N positive and negative samples. Exploratory; the HYPOTHESIS rows are the only
pre-specified ones.

    .venv/bin/python scripts/extract_fab_features.py
Reads FAB_DATA_PATH (config.py) and derived/fig4_fingerprints/feature_effects.csv; writes derived/fab/fab_feature_effects.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from config import FAB_DATA_PATH, SELECTED_VARIABLES  # noqa: E402
from core.data_loading import DERIVED_DIR, feature_lists, load_cohort  # noqa: E402
from extract_fab_analysis import CONTRAST, GROUPS, group_of  # noqa: E402

OUT = DERIVED_DIR / "fab"
MIN_N = 5
HYPOTHESIS = dict(variable="NPM1", modality="BM",
                  features=["Blasts_nucleus_solidity_median", "Blasts_nucleus_compactness_median", "Blasts_nucleus_eccentricity_median"])


def delong(x, y):
    """AUC of x (positives) over y (negatives) and its DeLong variance."""
    xs, ys = np.sort(x), np.sort(y)
    lo, hi = np.searchsorted(ys, x, "left"), np.searchsorted(ys, x, "right")
    v10 = (lo + 0.5 * (hi - lo)) / len(y)
    lo, hi = np.searchsorted(xs, y, "left"), np.searchsorted(xs, y, "right")
    v01 = (len(x) - hi + 0.5 * (hi - lo)) / len(x)
    return v10.mean(), v10.var(ddof=1) / len(x) + v01.var(ddof=1) / len(y)


def van_elteren(strata):
    """Two-sided p and z of the stratified Wilcoxon test; strata = [(x_pos, y_neg)] with both classes present."""
    num = den = 0.0
    for x, y in strata:
        n1, n0 = len(x), len(y)
        n = n1 + n0
        r = rankdata(np.concatenate([x, y]))
        ties = np.unique(r, return_counts=True)[1]
        var = n1 * n0 / 12 * ((n + 1) - (ties ** 3 - ties).sum() / (n * (n - 1)))
        w = 1 / (n + 1)
        num += w * (r[:n1].sum() - n1 * (n + 1) / 2)
        den += w * w * var
    z = num / np.sqrt(den) if den > 0 else np.nan
    return z, 2 * norm.sf(abs(z))


def main():
    fab = pd.read_csv(FAB_DATA_PATH, dtype={"fab": str})[["subject_id", "fab"]].dropna()
    assert fab.subject_id.is_unique, "a patient has several FAB values"
    ref = pd.read_csv(DERIVED_DIR / "fig4_fingerprints" / "feature_effects.csv")
    rows = []
    for modality in ("BM", "PB"):
        df = load_cohort(modality).merge(fab, on="subject_id", how="inner")
        df["group"] = group_of(df.fab)
        df = df.dropna(subset=["group"])
        feats = feature_lists(df, modality)
        print(f"{modality}: {len(df)} samples with a FAB group: {df.group.value_counts().to_dict()}")
        for var in SELECTED_VARIABLES:
            d = df[df[var].isin([0, 1])]
            for ftype, cols in feats.items():
                part = []
                for c in cols:
                    row = dict(modality=modality, variable=var, feature_type=ftype, feature_col=c)
                    strata, auc = [], {}
                    for g in GROUPS:
                        s = d[d.group == g]
                        x, y = s.loc[s[var] == 1, c].dropna().to_numpy(float), s.loc[s[var] == 0, c].dropna().to_numpy(float)
                        row[f"n_pos_{g}"], row[f"n_neg_{g}"] = len(x), len(y)
                        if len(x) >= MIN_N and len(y) >= MIN_N and not (x.std() == 0 and y.std() == 0 and x[0] == y[0]):
                            auc[g] = delong(x, y)
                            row[f"delta_{g}"] = 2 * auc[g][0] - 1
                            strata.append((x, y))
                    if len(strata) >= 1:
                        row["z_stratified"], row["p_stratified"] = van_elteren(strata)
                    a, b = CONTRAST
                    if a in auc and b in auc:
                        row["delta_diff"] = row[f"delta_{a}"] - row[f"delta_{b}"]
                        z = (auc[a][0] - auc[b][0]) / np.sqrt(auc[a][1] + auc[b][1])
                        row["p_interaction"] = 2 * norm.sf(abs(z))
                    part.append(row)
                part = pd.DataFrame(part)
                for p in ("p_interaction", "p_stratified"):
                    if p in part and part[p].notna().any():
                        ok = part[p].notna()
                        part.loc[ok, p.replace("p_", "padj_")] = multipletests(part.loc[ok, p], method="fdr_bh")[1]
                rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    ann = ref.drop_duplicates(["modality", "feature_col"])[["modality", "feature_col", "cell_type", "feature", "family"]]
    out = out.merge(ann, on=["modality", "feature_col"], how="left").merge(
        ref[["modality", "variable", "feature_col", "delta_pos_vs_neg"]].rename(columns={"delta_pos_vs_neg": "delta_overall"}),
        on=["modality", "variable", "feature_col"], how="left")
    out["hypothesis"] = ((out.variable == HYPOTHESIS["variable"]) & (out.modality == HYPOTHESIS["modality"])
                         & out.feature_col.isin(HYPOTHESIS["features"]))
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "fab_feature_effects.csv", index=False)
    print(f"{len(out)} rows; hypothesis rows: {int(out.hypothesis.sum())}\nWritten to {OUT / 'fab_feature_effects.csv'}")


if __name__ == "__main__":
    main()
