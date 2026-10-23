"""Extract Fig. 4 (morphological fingerprint) results once and save them for plotting.

Run from the repository root:
    .venv/bin/python scripts/extract_fig4_fingerprint_results.py

Outputs (git-ignored, under data/<DATA_VERSION_IDENTIFIER>/derived/fig4_fingerprints/):
    feature_effects.csv   one row per variable x modality x feature with
                          - association: Pos vs Neg (sample level, aggregated features as in the
                            notebook balloon plots; Wilcoxon rank-sum, BH-FDR per variable x
                            modality x feature type; Cliff's delta)
                          - model use: top-attention cells vs all cells within positive samples
                            (cell-level parquet re-aggregated per sample, as in build_three_group_df)
    fingerprint_tags.csv  associated features only, with model-use status and feature family
    attention_enrichment.csv  cell type x variable attention log-enrichment (predicted positives)
    attention_spread.csv      N_eff / N per bag and variable
    run_info.json         inputs, parameters and status rules

The loading / preprocessing / statistics code mirrors 11_cytomorphology_analysis.ipynb
(cells "Load data", calculate_foldchange_pvalue, build_three_group_df).
Loading / feature lists live in core/data_loading.py (shared with 11_cytomorphology_analysis_V2.ipynb).
"""
import glob
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ranksums
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config import DATA_VERSION_IDENTIFIER  # noqa: E402
from core.data_loading import (  # noqa: E402
    FAMILY, FEAT_DIR, FILES, RUN_PATHS, SELECTED_VARIABLES, STAT_SUFFIXES, feature_lists, load_cohort, preprocess,
)
from core.attention import compute_logR_summary, effective_num_cells, select_top_attention_cell_ids  # noqa: E402
from core.cell_feature_aggregation.aggregate import aggregate_bm_dataset, aggregate_pb_dataset  # noqa: E402

# ── parameters ────────────────────────────────────────────────────────────────
ATTENTION_MODEL = "per_modality"    # "per_modality" = BM model for BM features, PB model for PB features;
                                    # "FUSION" = BM+PB model with top cells chosen per slide modality
ATTENTION_THRESHOLD = 0.5           # cells making up this fraction of the attention mass
ALPHA = 0.05
NOT_USED_RATIO = 0.5                # fallback rule, see classify_status()
MIN_SAMPLES = 10                    # min samples per group for a feature to be tested
                                    # (per-sample cell minimum is applied in aggregation: BM 20 / megas 10, PB 10)
CELL_LINEAGE_ORDER = {
    "BM": ["Blasts", "Proerythroblasts", "Erythroblasts", "Megakaryocytes", "Promyelocytes", "Myelocytes",
           "Metamyelocytes", "Eosinophils_immature", "Eosinophils", "Basophils", "Neutrophils", "Promonocytes",
           "Monocytes", "Macrophages", "Lymphocytes", "Plasma_cells"],
    # PB model input: band + segmented neutrophils merged into "Neutrophils"; platelets not modelled
    "PB": ["Blasts", "Erythroblasts", "Myelocytes", "Metamyelocytes", "Eosinophils", "Basophils", "Neutrophils",
           "Monocytes", "Lymphocytes", "Plasma_cells"],
}

OUT_DIR = REPO / f"data/{DATA_VERSION_IDENTIFIER}/derived/fig4_fingerprints"
# ── statistics ───────────────────────────────────────────────────────────────
def cliffs_delta(a, b):
    """Cliff's delta of a vs b (positive = a larger), as in calculate_foldchange_pvalue."""
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return 2 * u / (len(a) * len(b)) - 1


def compare(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < MIN_SAMPLES or len(b) < MIN_SAMPLES or (np.all(a == a[0]) and np.all(b == a[0])):
        return np.nan, np.nan, len(a), len(b)
    return cliffs_delta(a, b), ranksums(a, b).pvalue, len(a), len(b)


def fdr(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() > 1:
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    elif ok.sum() == 1:
        out[ok] = p[ok]
    return out


def classify_status(d_assoc, d_top_vs_pos, p_top_vs_pos_adj, d_top_vs_neg):
    """Model-use status of an associated feature.

    enriched   : top cells differ further from Neg than all Pos cells (top vs Pos significant, same sign)
    not used   : top vs Pos significant in the opposite sign, or top cells lose most of the
                 association (|delta top vs Neg| < NOT_USED_RATIO * |delta Pos vs Neg|)
    maintained : otherwise
    """
    if np.isnan(d_top_vs_pos):
        return "not evaluable"
    same = np.sign(d_top_vs_pos) == np.sign(d_assoc)
    if p_top_vs_pos_adj < ALPHA and same:
        return "enriched"
    if (p_top_vs_pos_adj < ALPHA and not same) or abs(d_top_vs_neg) < NOT_USED_RATIO * abs(d_assoc):
        return "not used"
    return "maintained"


def parse_feature(col, feature_type):
    """Return (cell_type, feature, family) for a feature column."""
    if feature_type == "cell_differential":
        return col.replace("_proportion", ""), "proportion", "proportion"
    if feature_type == "cell_statistic":
        for s in STAT_SUFFIXES:
            if col.endswith("_" + s):
                return col[: -len(s) - 1], s.replace("_median", ""), FAMILY[s]
    name = col.replace("_proportion", "")
    for sep in ("-", "_"):
        for ct in ["Myeloid_precursors", "APL", "Blasts", "Erythroblasts", "Megakaryocytes", "Promyelocytes",
                   "Myelocytes", "Basophils", "Neutrophils", "Eosinophils", "Lymphocytes", "Plasma_cells"]:
            if name.startswith(ct + sep):
                return ct.replace("APL", "Myeloid_precursors"), name[len(ct) + 1:], "dysplasia"
    return "", name, "dysplasia"


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, enrich, spread = [], [], []
    for modality in ["BM", "PB"]:
        model = modality if ATTENTION_MODEL == "per_modality" else ATTENTION_MODEL
        print(f"[{modality}] loading attention scores ({model} model) …")
        files = sorted(glob.glob(f"{REPO / RUN_PATHS[model]}/fold_*/cell_attention_scores.csv"))
        df_attn = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        if "modality" in df_attn.columns and model == "FUSION":
            df_attn = df_attn[df_attn["modality"] == modality]
        df_attn["database_id"] = df_attn["database_id"].astype(str)
        df_attn["album_id"] = df_attn["album_id"].astype(str)
        attn_vars = [c.replace("cell_attention_score_", "") for c in df_attn.columns if c.startswith("cell_attention_score_")]
        vars_here = [v for v in SELECTED_VARIABLES if v in attn_vars]

        # attention enrichment (= notebook heatmap): mean log(attention share / cell share), predicted-positive bags
        lr = compute_logR_summary(df_attn, vars_here, CELL_LINEAGE_ORDER[modality], subset="pred_pos")
        enrich.append(lr.reset_index(names="cell_type").melt(id_vars="cell_type", var_name="variable", value_name="log_enrichment")
                      .assign(modality=modality, model=model))
        # attention spread: N_eff / N per bag, predicted positives vs negatives
        for v in vars_here:
            col = f"cell_attention_score_{v}"
            for grp, sub in df_attn.dropna(subset=[col, f"{v}_true_label", f"{v}_pred_label"]).groupby("bag_id"):
                a = sub[col].to_numpy(float)
                if a.sum() <= 0:
                    continue
                spread.append(dict(modality=modality, model=model, variable=v, bag_id=grp,
                                   pred=int(sub[f"{v}_pred_label"].iloc[0]), true=int(sub[f"{v}_true_label"].iloc[0]),
                                   neff_frac=effective_num_cells(a / a.sum()) / len(a)))

        print(f"[{modality}] loading cohort and cell-level features …")
        df = load_cohort(modality)
        feats = feature_lists(df, modality)
        cell_df = pd.read_parquet(FEAT_DIR / FILES[modality]["parquet"])
        cell_df["database_id"] = cell_df["database_id"].astype(str)
        cell_df["album_id"] = cell_df["album_id"].astype(str)
        agg_fn = aggregate_bm_dataset if modality == "BM" else aggregate_pb_dataset

        print(f"[{modality}] aggregating all cells per sample …")
        samples = df[["database_id", "album_id"]].drop_duplicates()
        agg_all = preprocess(agg_fn(cell_df.merge(samples, on=["database_id", "album_id"])).reset_index(), modality)

        for var in SELECTED_VARIABLES:
            pos = df[df[var] == 1]
            neg = df[df[var] == 0]
            if len(pos) < 3:
                continue
            # association on aggregated CSV features (= notebook balloon plots)
            assoc = []
            for ftype, cols in feats.items():
                res = [compare(pos[c], neg[c]) for c in cols]
                padj = fdr([r[1] for r in res])
                assoc += [(ftype, c, *r, pa) for c, r, pa in zip(cols, res, padj)]

            # model use: top-attention cells within positive samples
            top = None
            if var in attn_vars:
                top_ids = select_top_attention_cell_ids(df_attn, var, ATTENTION_THRESHOLD)
                pos_keys = pos[["database_id", "album_id"]]
                top_cells = cell_df.merge(top_ids, on=["database_id", "album_id", "cell_id"]).merge(pos_keys)
                if len(top_cells):
                    top = preprocess(agg_fn(top_cells).reset_index(), modality)
            pos_all = agg_all.merge(pos[["database_id", "album_id"]])
            neg_all = agg_all.merge(neg[["database_id", "album_id"]])

            use = {}
            for ftype, cols in feats.items():
                res_tp, res_tn = [], []
                for c in cols:
                    ok = top is not None and c in top.columns and c in pos_all.columns
                    res_tp.append(compare(top[c], pos_all[c]) if ok else (np.nan, np.nan, 0, 0))
                    res_tn.append(compare(top[c], neg_all[c]) if ok else (np.nan, np.nan, 0, 0))
                padj = fdr([r[1] for r in res_tp])
                for c, rtp, rtn, pa in zip(cols, res_tp, res_tn, padj):
                    use[c] = (rtp[0], rtp[1], pa, rtn[0], rtp[2])

            for ftype, c, d, p, n1, n0, pa in assoc:
                d_tp, p_tp, pa_tp, d_tn, n_top = use.get(c, (np.nan,) * 5)
                ct, feat, fam = parse_feature(c, ftype)
                rows.append(dict(
                    variable=var, modality=modality, feature_type=ftype, feature_col=c, cell_type=ct,
                    feature=feat, family=fam, n_pos=n1, n_neg=n0,
                    delta_pos_vs_neg=d, p_pos_vs_neg=p, padj_pos_vs_neg=pa,
                    n_pos_top=n_top, delta_top_vs_pos=d_tp, p_top_vs_pos=p_tp, padj_top_vs_pos=pa_tp,
                    delta_top_vs_neg=d_tn,
                ))
            print(f"[{modality}] {var}: n_pos={len(pos)}  ({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows)
    res["associated"] = res["padj_pos_vs_neg"] < ALPHA
    res["model_use"] = [
        classify_status(r.delta_pos_vs_neg, r.delta_top_vs_pos, r.padj_top_vs_pos, r.delta_top_vs_neg)
        if r.associated else "" for r in res.itertuples()
    ]
    res.to_csv(OUT_DIR / "feature_effects.csv", index=False)
    pd.concat(enrich).to_csv(OUT_DIR / "attention_enrichment.csv", index=False)
    pd.DataFrame(spread).to_csv(OUT_DIR / "attention_spread.csv", index=False)

    tags = res[res["associated"]].copy()
    tags["abs_delta"] = tags["delta_pos_vs_neg"].abs()
    tags = tags.sort_values(["variable", "modality", "cell_type", "family", "abs_delta"], ascending=[True, True, True, True, False])
    tags["strongest_in_family"] = ~tags.duplicated(["variable", "modality", "cell_type", "family"])
    tags.drop(columns="abs_delta").to_csv(OUT_DIR / "fingerprint_tags.csv", index=False)

    info = dict(
        created=datetime.now().isoformat(timespec="seconds"), data_version=DATA_VERSION_IDENTIFIER,
        attention_model=ATTENTION_MODEL,
        attention_runs={m: RUN_PATHS[m if ATTENTION_MODEL == "per_modality" else ATTENTION_MODEL] for m in ["BM", "PB"]},
        attention_threshold=ATTENTION_THRESHOLD, alpha=ALPHA, not_used_ratio=NOT_USED_RATIO,
        min_samples_per_group=MIN_SAMPLES,
        attention_enrichment="core.attention.compute_logR_summary, subset=pred_pos (as notebook heatmap)",
        attention_spread="N_eff / N per bag (core.attention.effective_num_cells)",
        variables=SELECTED_VARIABLES, inputs={m: FILES[m] for m in FILES},
        association="Pos vs Neg on aggregated features; Wilcoxon rank-sum; BH-FDR per variable x modality x feature type; Cliff's delta",
        model_use="top-attention cells (per-sample attention mass threshold, per modality) vs all cells of positive samples, "
                  "both re-aggregated from the cell-level parquet; rank-sum; BH-FDR per variable x modality x feature type",
        status_rules=classify_status.__doc__.strip(),
        n_rows=len(res), n_associated=int(res["associated"].sum()), runtime_s=round(time.time() - t0),
    )
    json.dump(info, open(OUT_DIR / "run_info.json", "w"), indent=2, default=str)
    print(f"Saved to {OUT_DIR}  ({info['n_associated']} associated features, {info['runtime_s']} s)")
    print(tags.groupby(["modality", "model_use"]).size())


if __name__ == "__main__":
    main()
