"""Is the fingerprint of a target X shared with a co-occurring target Y? Fig 4/5 features of X adjusted for Y.

For each pair (X, Y) of PAIRS, modality and sample-level feature (as in scripts/extract_fig4_fingerprint_results.py), on the samples labelled for both:
  delta_all      Cliff's δ of X+ against X− (all samples; same sample set as below), p_all, padj_all (BH within modality × pair × feature type)
  delta_ypos     δ of X+ against X− within Y+ samples; delta_yneg within Y− samples (n_<cell> = samples per cell; a stratum needs MIN_N in both classes)
  delta_adj      stratum-weighted mean δ (weights n(X+) · n(X−)); p_adj_strat / padj_strat: van Elteren stratified Wilcoxon test of X given Y
  delta_yalone   δ of Y+ against Y− within X− samples (the effect of Y without X; similar to delta_all would mean the feature may belong to Y)
  p_interaction  difference of delta_ypos and delta_yneg (DeLong z-test on the AUCs), BH within modality × pair × feature type
cooc_summary.csv per modality and pair: features with an unadjusted effect (padj_all < ALPHA), how many keep their sign and stay significant after
adjusting for Y, Spearman correlation of delta_all and delta_adj, median |δ_adj| / |δ_all| of the significant features.
Exploratory (the pairs are chosen from the cell counts of scripts/extract_cooccurrence.py).

    .venv/bin/python scripts/extract_cooccurrence_features.py
Writes derived/cooccurrence/cooc_feature_effects.csv and cooc_summary.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, ranksums, spearmanr
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from core.data_loading import DERIVED_DIR, feature_lists, load_cohort  # noqa: E402
from extract_fab_features import delong, van_elteren  # noqa: E402

OUT = DERIVED_DIR / "cooccurrence"
# (X, Y): the fingerprint of X, adjusted for Y
PAIRS = [("NPM1", "FLT3_ITD"), ("NPM1", "IDH_mutations"), ("NPM1", "CpG_mutations"), ("NPM1", "RTK_RAS_mutations"),
         ("FLT3_ITD", "NPM1"), ("CpG_mutations", "NPM1"),
         ("TP53", "complex_karyotype"), ("TP53", "monosomal_karyotype"),
         ("complex_karyotype", "TP53"), ("monosomal_karyotype", "TP53")]
MIN_N = 8
ALPHA = 0.05


def cliffs(x, y):
    return 2 * delong(x, y)[0] - 1


def main():
    rows = []
    for modality in ("BM", "PB"):
        full = load_cohort(modality)
        feats = feature_lists(full, modality)
        for xv, yv in PAIRS:
            d = full[full[xv].isin([0, 1]) & full[yv].isin([0, 1])]
            n_cells = {f"{a}{b}": int(((d[xv] == a) & (d[yv] == b)).sum()) for a in (1, 0) for b in (1, 0)}
            part_all = []
            for ftype, cols in feats.items():
                part = []
                for c in cols:
                    v = d[[xv, yv, c]].dropna()
                    row = dict(modality=modality, x=xv, y=yv, feature_type=ftype, feature_col=c)
                    strata = {}
                    for ys in (1, 0):
                        s = v[v[yv] == ys]
                        a, b = s.loc[s[xv] == 1, c].to_numpy(float), s.loc[s[xv] == 0, c].to_numpy(float)
                        if len(a) >= MIN_N and len(b) >= MIN_N and np.ptp(np.concatenate([a, b])) > 0:
                            strata[ys] = (a, b)
                            row[f"delta_y{'pos' if ys else 'neg'}"] = cliffs(a, b)
                    a, b = v.loc[v[xv] == 1, c].to_numpy(float), v.loc[v[xv] == 0, c].to_numpy(float)
                    if len(a) >= MIN_N and len(b) >= MIN_N and np.ptp(v[c].to_numpy(float)) > 0:
                        row["delta_all"], row["p_all"] = cliffs(a, b), ranksums(a, b).pvalue
                    if strata:
                        w = {k: len(a) * len(b) for k, (a, b) in strata.items()}
                        row["delta_adj"] = sum(w[k] * row[f"delta_y{'pos' if k else 'neg'}"] for k in strata) / sum(w.values())
                        row["z_strat"], row["p_strat"] = van_elteren(list(strata.values()))
                    if len(strata) == 2:
                        (a1, b1), (a0, b0) = strata[1], strata[0]
                        (m1, v1), (m0, v0) = delong(a1, b1), delong(a0, b0)
                        row["p_interaction"] = 2 * norm.sf(abs((m1 - m0) / np.sqrt(v1 + v0)))
                    s = v[v[xv] == 0]
                    a, b = s.loc[s[yv] == 1, c].to_numpy(float), s.loc[s[yv] == 0, c].to_numpy(float)
                    if len(a) >= MIN_N and len(b) >= MIN_N and np.ptp(s[c].to_numpy(float)) > 0:
                        row["delta_yalone"] = cliffs(a, b)
                    part.append(row)
                part = pd.DataFrame(part)
                for p, q in (("p_all", "padj_all"), ("p_strat", "padj_strat"), ("p_interaction", "padj_interaction")):
                    if p in part and part[p].notna().any():
                        ok = part[p].notna()
                        part.loc[ok, q] = multipletests(part.loc[ok, p], method="fdr_bh")[1]
                part_all.append(part)
            part = pd.concat(part_all, ignore_index=True)
            part = part.assign(**{f"n_{k}": n for k, n in n_cells.items()})
            rows.append(part)
            print(f"{modality} {xv} | {yv}: cells X+Y+ {n_cells['11']}, X+Y− {n_cells['10']}, X−Y+ {n_cells['01']}, X−Y− {n_cells['00']}", flush=True)
    out = pd.concat(rows, ignore_index=True)
    ref = pd.read_csv(DERIVED_DIR / "fig4_fingerprints" / "feature_effects.csv")
    ann = ref.drop_duplicates(["modality", "feature_col"])[["modality", "feature_col", "cell_type", "feature", "family"]]
    out = out.merge(ann, on=["modality", "feature_col"], how="left")

    summ = []
    for (modality, xv, yv), g in out.groupby(["modality", "x", "y"], sort=False):
        sig = g[(g.padj_all < ALPHA) & g.delta_adj.notna()]
        kept = sig[(np.sign(sig.delta_adj) == np.sign(sig.delta_all)) & (sig.padj_strat < ALPHA)]
        both = g.dropna(subset=["delta_all", "delta_adj"])
        summ.append(dict(modality=modality, x=xv, y=yv, n_features=len(both), n_sig_all=len(sig), n_kept=len(kept),
                         frac_kept=len(kept) / len(sig) if len(sig) else np.nan,
                         rho_all_adj=spearmanr(both.delta_all, both.delta_adj)[0] if len(both) > 3 else np.nan,
                         median_ratio=(sig.delta_adj.abs() / sig.delta_all.abs()).median() if len(sig) else np.nan,
                         n_y_alone_similar=int(((g.delta_yalone.abs() > 0.5 * g.delta_all.abs()) & (g.padj_all < ALPHA)
                                                 & (np.sign(g.delta_yalone) == np.sign(g.delta_all))).sum())))
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "cooc_feature_effects.csv", index=False)
    pd.DataFrame(summ).to_csv(OUT / "cooc_summary.csv", index=False)
    print(pd.DataFrame(summ).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
