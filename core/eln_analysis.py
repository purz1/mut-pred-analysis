"""ELN risk groups from label-replacement modes: agreement with the true-label groups and C-index with paired bootstrap CIs.

Used by 8_eln_risk_group_analysis.ipynb. The R output holds one ELN1 group per patient for each mode
(TARGETS = true labels, TP, TP_FP, FN ...); every mode is compared with TARGETS on the patients that have a valid ELN1 in both.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import cohen_kappa_score

ELN_RISK = {"Favorable": 0, "Intermediate": 1, "Adverse": 2}
KEYS = ["album_id", "database_id", "henkilotunnus"]


def survival_frame(df_survival, horizon=None):
    """time (months) and event per patient. There are no follow-up dates, so alive patients (time missing) are censored at
    the longest observed time. With a horizon, follow-up is cut there (events later than the horizon become censored)."""
    t = df_survival["time_dg_to_death"]
    out = df_survival[KEYS + ["died"] + [c for c in ("age",) if c in df_survival]].copy()
    out["time"] = t.fillna(t.max()) / 30
    out["event"] = out["died"].astype(bool)
    if horizon is not None:
        out["event"] &= out["time"] <= horizon
        out["time"] = out["time"].clip(upper=horizon)
    return out.drop(columns="died")


def load_eln(df_survival, r_output_dir, mode, variable_set, model_type, slide_type, run_id):
    """Survival table merged with the ELN1 group of a mode; None if the R output of the mode does not exist."""
    path = Path(r_output_dir) / f"aml_mut_pred_{mode}_{variable_set}_{model_type}_eln_{slide_type}_{run_id}.csv"
    if not path.exists():
        return None
    df = pd.merge(df_survival, pd.read_csv(path), on=KEYS, how="inner")
    return df.dropna(subset="ELN1").assign(score=lambda d: d["ELN1"].map(ELN_RISK))


def c_index(time, score, event):
    """Harrell's C for integer risk scores (higher = higher risk): pairs where the earlier time is an event; risk ties count 0.5.
    O(n log n) with category counts instead of all pairs."""
    t, s, e = np.asarray(time, float), np.asarray(score, int), np.asarray(event, bool)
    order = np.argsort(t, kind="stable")
    t, s, e = t[order], s[order], e[order]
    k = s.max() + 1
    suffix = np.vstack([np.cumsum(np.eye(k, dtype=np.int64)[s][::-1], axis=0)[::-1], np.zeros((1, k), np.int64)])
    ev = np.flatnonzero(e)
    later = suffix[np.searchsorted(t, t, side="right")][ev]  # [events, k]: category counts among patients with a later time
    si, rows = s[ev], np.arange(len(ev))
    tied = later[rows, si]
    lower = np.cumsum(later, axis=1)[rows, si] - tied  # later patients with a lower score: the earlier event had the higher risk
    higher = later.sum(1) - lower - tied
    n = lower.sum() + higher.sum() + tied.sum()
    return np.nan if n == 0 else (lower.sum() + 0.5 * tied.sum()) / n


def bc_ci(boot, point, alpha=0.05):
    """Bias-corrected percentile CI (as in the performance figures)."""
    boot = boot[~np.isnan(boot)]
    p = np.clip(((boot < point).sum() + 0.5 * (boot == point).sum()) / len(boot), 1 / (len(boot) + 1), 1 - 1 / (len(boot) + 1))
    z0 = norm.ppf(p)
    lo, hi = norm.cdf(2 * z0 + norm.ppf([alpha / 2, 1 - alpha / 2]))
    return tuple(np.quantile(boot, [lo, hi]))


def compare_mode(base, mode, n_boot=2000, seed=0):
    """Mode versus TARGETS on the patients with a valid ELN1 in both (merge on KEYS).

    base, mode: tables from load_eln. C-index of both groupings and their difference (mode − TARGETS) with 95% bias-corrected
    percentile CIs from one patient bootstrap (the same resample scores both, so the difference is paired), plus agreement."""
    d = base.merge(mode[KEYS + ["score"]], on=KEYS, suffixes=("_base", "_mode"))
    t, e, a, b = d["time"].to_numpy(), d["event"].to_numpy(), d["score_base"].to_numpy(), d["score_mode"].to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.array([[c_index(t[i], a[i], e[i]), c_index(t[i], b[i], e[i])]
                     for i in (rng.integers(0, len(d), len(d)) for _ in range(n_boot))])
    point = np.array([c_index(t, a, e), c_index(t, b, e)])
    delta = point[1] - point[0]
    dci = bc_ci(boot[:, 1] - boot[:, 0], delta)
    return dict(
        n=len(d), n_lost=len(base) - len(d), events=int(e.sum()),
        c_targets=point[0], c_targets_low=bc_ci(boot[:, 0], point[0])[0], c_targets_high=bc_ci(boot[:, 0], point[0])[1],
        c_mode=point[1], c_mode_low=bc_ci(boot[:, 1], point[1])[0], c_mode_high=bc_ci(boot[:, 1], point[1])[1],
        delta=delta, delta_low=dci[0], delta_high=dci[1],
        same_group=(a == b).mean(), kappa_w=cohen_kappa_score(a, b, weights="quadratic") if len(set(a) | set(b)) > 1 else np.nan,
        lower_risk=int((b < a).sum()), higher_risk=int((b > a).sum()),
    )


def survival_at(df, times=(12, 24), group_col="ELN1", order=tuple(ELN_RISK)):
    """Kaplan-Meier survival (Greenwood 95% CI, lifelines) at the given months per risk group."""
    from lifelines import KaplanMeierFitter
    rows = []
    for g in order:
        s = df[df[group_col] == g]
        if len(s) == 0:
            continue
        km = KaplanMeierFitter().fit(s["time"], s["event"])
        ci = km.confidence_interval_survival_function_
        for t in times:
            rows.append(dict(group=g, n=len(s), month=t, survival=float(km.predict(t)),
                             low=float(ci.iloc[:, 0].asof(t)), high=float(ci.iloc[:, 1].asof(t))))
    return pd.DataFrame(rows)


def cox_with_age(base, mode, age_col="age"):
    """Cox models of overall survival: age alone, age + ELN from TARGETS, age + ELN from the mode (ELN as 0/1/2 score).
    Returns None if the survival table has no age column. Likelihood-ratio test of each ELN model against age alone."""
    if age_col not in base:
        return None
    from lifelines import CoxPHFitter
    from scipy.stats import chi2
    d = base.merge(mode[KEYS + ["score"]], on=KEYS, suffixes=("_targets", "_mode")).dropna(subset=[age_col])
    d = d.rename(columns={age_col: "age"})
    rows, ll0 = [], None
    for name, cols in (("age", []), ("age + ELN (TARGETS)", ["score_targets"]), ("age + ELN (mode)", ["score_mode"])):
        m = CoxPHFitter().fit(d[["time", "event", "age"] + cols], "time", "event")
        ll0 = m.log_likelihood_ if ll0 is None else ll0
        row = dict(model=name, n=len(d), c_index=m.concordance_index_,
                   p_vs_age=np.nan if not cols else chi2.sf(2 * (m.log_likelihood_ - ll0), 1))
        if cols:
            row.update(hr_per_step=float(np.exp(m.params_[cols[0]])), hr_low=float(np.exp(m.confidence_intervals_.iloc[-1, 0])),
                       hr_high=float(np.exp(m.confidence_intervals_.iloc[-1, 1])))
        rows.append(row)
    return pd.DataFrame(rows)
