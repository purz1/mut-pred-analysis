"""FAB class and the targets: composition, coverage, prevalence, model performance and errors per FAB group.

Input: FAB_DATA_PATH (config.py: data/<version>/fab/aml_mut_pred_fab_<version>.csv) with the columns subject_id and fab (M0 … M7), one row per patient; and the predictions of the three
model runs (config.py). FAB is attached to every model by subject_id (the same patient in BM and PB).
FAB classes are pooled into GROUPS; M3 (APL) only serves as a positive control and is left out of the group contrasts below.
Per modality × variable × FAB group: n, positives, prevalence, AUROC / AP (patient bootstrap CI), miss rate (FN / true positives) and
false-alarm rate (FP / true negatives) with exact Clopper–Pearson CIs. Contrast of the two main groups (CONTRAST): Fisher exact test of the
miss rate and of the false-alarm rate, BH-FDR across the FAB_TARGETS within model and error type (exploratory, small cells).
Coverage: samples with and without FAB compared by database (year) and by target prevalence.

    .venv/bin/python scripts/extract_fab_analysis.py
Writes data/<DATA_VERSION_IDENTIFIER>/derived/fab/: fab_composition.csv, fab_target_prevalence.csv, fab_coverage.csv, fab_performance.csv,
fab_errors.csv, fab_contrast.csv, fab_outcomes.csv (TP / FN / FP / TN per FAB class), fab_scores.csv (model probability per sample of the FAB_TARGETS, no identifiers)
and fab_score_contrast.csv (Mann–Whitney M0–M2 vs M4–M5 of the probability among true positives and true negatives, BH-FDR across the FAB_TARGETS).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, fisher_exact, mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from config import BM_RUN_PATH, FAB_DATA_PATH, FUSION_RUN_PATH, PB_RUN_PATH, SELECTED_VARIABLES  # noqa: E402
from core.data_loading import DERIVED_DIR  # noqa: E402

OUT = DERIVED_DIR / "fab"
GROUPS = {"M0–M2": ["M0", "M1", "M2"], "M3": ["M3"], "M4–M5": ["M4", "M5"]}
CONTRAST = ("M0–M2", "M4–M5")
RUNS = {"BM": BM_RUN_PATH, "PB": PB_RUN_PATH, "FUSION": FUSION_RUN_PATH}
FAB_TARGETS = ["PML-RARA", "NPM1"]  # targets of the per-target supplements (S15, S16); the fingerprint by FAB group is only computed for these
MIN_CLASS = 5  # positives and negatives needed for an AUROC
N_BOOT = 2000


def exact_ci(k, n):
    """Clopper–Pearson 95% CI of a proportion."""
    if n == 0:
        return np.nan, np.nan
    return (beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


def boot_metric(y, p, f, rng):
    b = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            b.append(f(y[i], p[i]))
    return tuple(np.percentile(b, [2.5, 97.5])) if b else (np.nan, np.nan)


def group_of(fab):
    return fab.map({c: g for g, cs in GROUPS.items() for c in cs})


def main():
    fab = pd.read_csv(FAB_DATA_PATH, dtype={"fab": str})[["subject_id", "fab"]].dropna()
    assert fab.subject_id.is_unique, "a patient has several FAB values"
    rng = np.random.default_rng(0)
    comp, prev, cov, perf, err, con, outc, scores = [], [], [], [], [], [], [], []

    for model, run in RUNS.items():
        pred = pd.read_csv(REPO / run / "predictions.csv")
        d = pred.merge(fab.assign(subject_id=fab.subject_id.astype(pred.subject_id.dtype)), on="subject_id", how="left")
        d["group"] = group_of(d.fab)
        print(f"{model}: {len(d)} samples, FAB for {d.fab.notna().sum()}")

        comp += [dict(model=model, fab=c, n=int((d.fab == c).sum())) for c in sorted(d.fab.dropna().unique())]
        comp.append(dict(model=model, fab="missing", n=int(d.fab.isna().sum())))
        for db, g in d.groupby("database_id"):
            cov.append(dict(model=model, kind="database", name=db, n=len(g), with_fab=int(g.fab.notna().sum())))

        for v in SELECTED_VARIABLES:
            y, yh, p = f"{v}_true_label", f"{v}_pred_label", f"{v}_pred_prob"
            if y not in d:
                continue
            x = d.dropna(subset=[y])
            has = x.fab.notna()
            if has.any() and (~has).any():  # coverage: prevalence with vs without FAB
                tab = [[(x[has][y] == 1).sum(), (x[has][y] == 0).sum()], [(x[~has][y] == 1).sum(), (x[~has][y] == 0).sum()]]
                cov.append(dict(model=model, kind="target", name=v, n=len(x), with_fab=int(has.sum()),
                                prev_with=x[has][y].mean(), prev_without=x[~has][y].mean(), p=fisher_exact(tab)[1]))
            for gname, g in list(x.dropna(subset=["group"]).groupby("group")) + [("all with FAB", x[has])]:
                yy, pp = g[y].to_numpy(int), g[p].to_numpy(float)
                tp, fn = int(((g[y] == 1) & (g[yh] == 1)).sum()), int(((g[y] == 1) & (g[yh] == 0)).sum())
                fp, tn = int(((g[y] == 0) & (g[yh] == 1)).sum()), int(((g[y] == 0) & (g[yh] == 0)).sum())
                prev.append(dict(model=model, variable=v, level="group", group=gname, n=len(g), n_pos=int(yy.sum()), prevalence=yy.mean()))
                row = dict(model=model, variable=v, group=gname, n=len(g), n_pos=int(yy.sum()))
                if min(yy.sum(), len(yy) - yy.sum()) >= MIN_CLASS:
                    for name, fn_ in (("auroc", roc_auc_score), ("ap", average_precision_score)):
                        row[name] = fn_(yy, pp)
                        row[f"{name}_low"], row[f"{name}_high"] = boot_metric(yy, pp, fn_, rng)
                perf.append(row)
                for kind, k, n in (("miss", fn, tp + fn), ("false_alarm", fp, fp + tn)):
                    lo, hi = exact_ci(k, n)
                    err.append(dict(model=model, variable=v, group=gname, kind=kind, k=k, n=n,
                                    rate=k / n if n else np.nan, low=lo, high=hi))
            for c, g in x.dropna(subset=["fab"]).groupby("fab"):
                oc = g[f"{v}_true_label"].astype(int).astype(str) + g[f"{v}_pred_label"].astype(int).astype(str)
                outc.append(dict(model=model, variable=v, fab=c, TP=int((oc == "11").sum()), FN=int((oc == "10").sum()),
                                 FP=int((oc == "01").sum()), TN=int((oc == "00").sum())))
            if v in FAB_TARGETS:  # model score per FAB group (no identifiers)
                sc = x.dropna(subset=["group"])[["group", y, p]].rename(columns={y: "true", p: "prob"})
                scores.append(sc.assign(model=model, variable=v, true=sc.true.astype(int)))
            prev += [dict(model=model, variable=v, level="class", group=c, n=len(g), n_pos=int((g[y] == 1).sum()), prevalence=(g[y] == 1).mean())
                     for c, g in x.dropna(subset=["fab"]).groupby("fab")]
            # contrast of the two main groups
            for kind, pos in (("miss", 1), ("false_alarm", 0)):
                cells = []
                for gname in CONTRAST:
                    g = x[(x.group == gname) & (x[y] == pos)]
                    wrong = int((g[yh] != pos).sum())
                    cells.append([wrong, len(g) - wrong])
                row = dict(model=model, variable=v, kind=kind, group_a=CONTRAST[0], group_b=CONTRAST[1],
                           k_a=cells[0][0], n_a=sum(cells[0]), k_b=cells[1][0], n_b=sum(cells[1]))
                if min(row["n_a"], row["n_b"]) > 0 and sum(c[0] for c in cells) > 0:
                    row["p"] = fisher_exact(cells)[1]
                con.append(row)

    con = pd.DataFrame(con)
    con["padj"] = np.nan  # BH across the FAB_TARGETS only (pre-specified); the other targets keep the raw p
    for _, idx in con[con.variable.isin(FAB_TARGETS)].groupby(["model", "kind"]).groups.items():
        ok = con.loc[idx, "p"].dropna().index
        if len(ok):
            con.loc[ok, "padj"] = multipletests(con.loc[ok, "p"], method="fdr_bh")[1]

    scores = pd.concat(scores, ignore_index=True)
    sc_con = []
    for (model, v, t), g in scores.groupby(["model", "variable", "true"]):
        a, b = (g.loc[g.group == c, "prob"] for c in CONTRAST)
        sc_con.append(dict(model=model, variable=v, true=t, n_a=len(a), n_b=len(b),
                           p=mannwhitneyu(a, b).pvalue if min(len(a), len(b)) >= 3 else np.nan))
    sc_con = pd.DataFrame(sc_con)
    sc_con["padj"] = np.nan  # BH across FAB_TARGETS within model and class
    for _, idx in sc_con.groupby(["model", "true"]).groups.items():
        ok = sc_con.loc[idx, "p"].dropna().index
        if len(ok):
            sc_con.loc[ok, "padj"] = multipletests(sc_con.loc[ok, "p"], method="fdr_bh")[1]

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("fab_composition", comp), ("fab_target_prevalence", prev), ("fab_coverage", cov),
                       ("fab_performance", perf), ("fab_errors", err), ("fab_contrast", con), ("fab_outcomes", outc), ("fab_scores", scores), ("fab_score_contrast", sc_con)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    print(f"Written to {OUT}")


if __name__ == "__main__":
    main()
