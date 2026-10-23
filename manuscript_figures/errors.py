"""Supplementary error analysis: where errors occur and whether misclassified samples look like the other class.

errors_in_context()   outcome oncoprint (TP / FN / FP / TN) + co-occurrence of FP / FN with other abnormalities
shift_grid()          composition · dysplasia · QC: shift of the FP / FN median toward the other class, BM and PB
shift_grid_morph(mod) the same for morphometry, one modality per figure
variable_examples()   four-group distributions for a chosen list of (variable, modality, feature)

Shift per feature with TP vs TN FDR < 0.05: FP = (med FP − med TN) / (med TP − med TN), FN = (med TP − med FN) /
(med TP − med TN); means replace medians where TN and TP medians tie; ≥ 3 errors required.
Inputs: derived/error_analysis (scripts/extract_error_attention_cache.py → extract_error_outcome_stats.py) and
derived/fig4_fingerprints/feature_effects.csv.
"""
from functools import cache, wraps

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from core.data_loading import load_cohort
from . import config as C
from .style import MM, VARS, OUTS, OUT_COL, vname, vcol

ROW = 3.0  # mm per feature row
SMALL_N = 5  # error groups with fewer samples are hatched (still shown)
SHIFT_CMAP = plt.get_cmap("RdBu_r")  # signed: red = higher than own class, blue = lower
ONC_COL = {"TP": "#174E6D", "FN": "#7fb3d5", "FP": "#D69455", "TN": "#ffffff", "NaN": "#E7E7E7"}
FAMILY = ["proportion", "dysplasia", "size", "N:C", "nuclear shape", "cell shape", "qc"]  # row order
F_FAMS = ["proportion", "dysplasia", "qc"]
MORPH = ["size", "N:C", "nuclear shape", "cell shape"]
FEAT_SHORT = {"nucleus_area": "nucl. area", "nucleus_solidity": "nucl. solidity",
              "nucleus_eccentricity": "nucl. eccentricity", "nucleus_compactness": "nucl. compactness",
              "entire_area": "cell area", "entire_compactness": "cell compactness", "entire_N:C-ratio": "N:C"}
# targets that are subsets of / defined by each other — co-occurrence is trivial
NESTED = [{"IDH2", "IDH_mutations"}, {"IDH2", "CpG_mutations"}, {"IDH_mutations", "CpG_mutations"},
          {"FLT3_ITD", "FLT3_mutations"}, {"FLT3_ITD", "RTK_RAS_mutations"}, {"FLT3_mutations", "RTK_RAS_mutations"},
          {"KIT", "RTK_RAS_mutations"}, {"NPM1", "NPM1_AND_FLT3_ITD"}, {"FLT3_ITD", "NPM1_AND_FLT3_ITD"},
          {"FLT3_mutations", "NPM1_AND_FLT3_ITD"}, {"RTK_RAS_mutations", "NPM1_AND_FLT3_ITD"},
          {"complex_karyotype", "monosomal_karyotype"}, {"monosomal_karyotype", "monosomy5_del_5q"}]


def _rc(f):
    @wraps(f)
    def g(*a, **k):
        with plt.rc_context({"font.size": 5.5}):
            return f(*a, **k)
    return g


@cache
def stats():
    """Per modality × variable × feature: group sizes, medians, means, Cliff's δ and BH-FDR per contrast."""
    st = pd.read_csv(C.ERRORS / "morph_outcome_stats.csv")
    eff = pd.read_csv(C.FINGERPRINTS / "feature_effects.csv")
    fam = st.feature_type.map({"cell_differential": "proportion", "dysplasia": "dysplasia", "qc": "qc"})
    st["family"] = fam.fillna(st.feature_col.map(eff.drop_duplicates("feature_col").set_index("feature_col").family))
    return st


@cache
def samples():
    """Per modality × variable × labelled sample: outcome, n cells, low-quality share, top-attention cell summaries."""
    return pd.read_parquet(C.ERRORS / "sample_attention_summary.parquet")


def _long(r):
    if r.family == "qc":
        return r.feature
    ct = r.cell_type.replace("_", " ").replace("-", " ")
    if r.family == "proportion":
        return f"{ct} (%)"
    if r.family == "dysplasia":
        f = r.feature.replace("apl_", "").replace("neut_gran_normal", "normal granulation").replace("neut_gran_", "") \
            .replace("hyper", "hypergranular").replace("hypo", "hypogranular").replace("granulargranulated", "granular")
        return f"{ct} – {f}"
    return f"{ct} – {FEAT_SHORT.get(r.feature, r.feature)}"


def _rows(d, fams):
    """Phenotype features (TP vs TN FDR < 0.05) of the given families, one row per feature, in family order."""
    r = d[(d.padj_TP_vs_TN < 0.05) & d.family.isin(fams)].drop_duplicates("feature_col")
    return r.assign(fo=r.family.map({f: i for i, f in enumerate(FAMILY)})).sort_values(["fo", "cell_type", "feature"])


def _panel_label(fig, x, y, lab, text=""):
    fig.text(x, y, lab, fontsize=10, fontweight="bold", va="center")
    if text:
        fig.text(x + 0.035, y, text, fontsize=5.6, va="center")


# ── errors in context ────────────────────────────────────────────────────────
def outcome_matrix(mod):
    S = samples()
    return S[S.modality == mod].pivot_table(index=["database_id", "album_id"], columns="variable",
                                            values="outcome", aggfunc="first").reindex(columns=VARS)


def cooccurrence(mod, mode):
    """Fisher exact per (error target a, co-occurring target b): FP vs TN (or FN vs TP) × b present; BH within row."""
    m = outcome_matrix(mod)
    truth = m.isin(["TP", "FN"]).astype(float).where(m.notna())
    rows = []
    for a in VARS:
        sub = m[a].isin(["TN", "FP"] if mode == "FP" else ["TP", "FN"])
        err = (m.loc[sub, a] == mode).astype(int)
        for b in VARS:
            if a == b or {a, b} in NESTED:
                continue
            t = truth.loc[sub, b]
            ok = t.notna()
            e, tb = err[ok], t[ok]
            tab = [[((e == 1) & (tb == 1)).sum(), ((e == 1) & (tb == 0)).sum()],
                   [((e == 0) & (tb == 1)).sum(), ((e == 0) & (tb == 0)).sum()]]
            if sum(tab[0]) == 0 or sum(tab[1]) == 0 or tab[0][0] + tab[1][0] == 0:
                continue
            _, p = fisher_exact(tab)
            lor = np.log10(((tab[0][0] + .5) * (tab[1][1] + .5)) / ((tab[0][1] + .5) * (tab[1][0] + .5)))
            rows.append(dict(a=a, b=b, lor=lor, p=p, k=tab[0][0], nb=tab[0][0] + tab[1][0]))
    r = pd.DataFrame(rows)
    r["padj"] = r.groupby("a").p.transform(lambda p: multipletests(p, method="fdr_bh")[1])
    return r


def _oncoprint(fig, box, order):
    l, b, r, t = box
    gs = fig.add_gridspec(1, 4, left=l, right=r, top=t, bottom=b, width_ratios=[4, 1, 4, 1], wspace=0.06)
    for j, mod in enumerate(("BM", "PB")):
        m = outcome_matrix(mod)
        pos = m.isin(["TP", "FN"])
        m = m.loc[pos[order].sort_values(by=order, ascending=False).index, order]
        img = np.ones((len(order), len(m), 3))
        for g, c in ONC_COL.items():
            img[m.T.fillna("NaN").values == g] = matplotlib.colors.to_rgb(c)
        ax = fig.add_subplot(gs[2 * j]); ax.imshow(img, aspect="auto", interpolation="none")
        ax.set_yticks(range(len(order))); ax.set_yticklabels([vname(v) for v in order] if j == 0 else [], fontsize=4.6)
        for tl, v in zip(ax.get_yticklabels(), order):
            tl.set_color(vcol(v))
        ax.set_xticks([]); ax.set_xlabel(f"{mod} samples (n = {len(m)})", fontsize=5)
        ax.set_title(f"{mod} model", fontsize=6, fontweight="bold")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        axb = fig.add_subplot(gs[2 * j + 1])
        left = np.zeros(len(order))
        for g in ("TP", "FN", "FP"):
            n = (m == g).sum().values
            axb.barh(range(len(order)), n, left=left, color=ONC_COL[g], height=0.8)
            left += n
        axb.set_ylim(len(order) - 0.5, -0.5)
        axb.tick_params(labelleft=False, length=0, labelsize=4.3)
        axb.spines[["top", "right", "left"]].set_visible(False)
        axb.set_xlabel("count", fontsize=5)
    h = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="0.5", lw=0.3, label=k if k != "NaN" else "missing") for k, c in ONC_COL.items()]
    fig.legend(handles=h, loc="center", bbox_to_anchor=((l + r) / 2, b - 0.035), ncol=5, frameon=False, fontsize=5)


def _cooc_panel(ax, mod, mode, xlab, ylab, norm):
    idx = {v: i for i, v in enumerate(VARS)}
    r = cooccurrence(mod, mode)
    mat = np.full((len(VARS), len(VARS)), np.nan); nom = mat.copy()
    for t in r[(r.p < 0.05) & (r.padj >= 0.05)].itertuples():
        nom[idx[t.a], idx[t.b]] = t.lor
    ax.imshow(nom, cmap="RdBu_r", norm=norm, aspect="auto", alpha=0.35)
    for t in r[r.padj < 0.05].itertuples():
        mat[idx[t.a], idx[t.b]] = t.lor
        ax.text(idx[t.b], idx[t.a], f"{t.k}\n/{t.nb}", ha="center", va="center", fontsize=3.3,
                color="w" if abs(t.lor) > 0.9 else "k", linespacing=0.9)
    ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xlim(-.5, len(VARS) - .5); ax.set_ylim(len(VARS) - .5, -.5)
    for a, b in (tuple(p) for p in NESTED):
        for x, y in ((idx[a], idx[b]), (idx[b], idx[a])):
            ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, fill=False, hatch="////", ec="0.85", lw=0))
    for k in range(len(VARS)):
        ax.add_patch(plt.Rectangle((k - .5, k - .5), 1, 1, color="0.85", lw=0))
    ax.set_xticks(range(len(VARS))); ax.set_yticks(range(len(VARS)))
    ax.set_xticklabels([vname(v) for v in VARS] if xlab else [], rotation=60, ha="right", fontsize=4.4)
    ax.set_yticklabels([vname(v) for v in VARS] if ylab else [], fontsize=4.4)
    for tl, v in zip(ax.get_xticklabels() + ax.get_yticklabels(), VARS + VARS):
        tl.set_color(vcol(v))
    ax.set_xticks(np.arange(len(VARS) + 1) - .5, minor=True); ax.set_yticks(np.arange(len(VARS) + 1) - .5, minor=True)
    ax.grid(which="minor", color="0.93", lw=0.4); ax.tick_params(which="both", length=0)
    ax.set_title(f"{mod}  ·  {'false positives' if mode == 'FP' else 'false negatives'}", fontweight="bold", fontsize=6)
    if ylab:
        ax.set_ylabel(f"{mode} for target", fontsize=5.3)
    if xlab:
        ax.set_xlabel("co-occurring abnormality present", fontsize=5.3)


@_rc
def errors_in_context():
    fig = plt.figure(figsize=(177 * MM, 235 * MM))
    order = outcome_matrix("BM").isin(["TP", "FN"]).sum().sort_values(ascending=False).index.tolist()
    _panel_label(fig, 0.01, 0.975, "A", "Outcome per sample and target (columns = samples, sorted by positives of the most frequent target)")
    _oncoprint(fig, (0.15, 0.75, 0.99, 0.945), order)
    _panel_label(fig, 0.01, 0.675, "B", "Co-occurrence of errors with other abnormalities (Fisher exact, log10 OR; "
                 "text + solid = FDR < 0.05 within row, faded = p < 0.05; hatched = nested pairs)")
    gs = fig.add_gridspec(2, 2, left=0.17, right=0.92, top=0.645, bottom=0.10, wspace=0.06, hspace=0.10)
    norm = TwoSlopeNorm(0, -1.5, 1.5)
    for i, mode in enumerate(("FP", "FN")):
        for j, mod in enumerate(("BM", "PB")):
            _cooc_panel(fig.add_subplot(gs[i, j]), mod, mode, xlab=i == 1, ylab=j == 0, norm=norm)
    cax = fig.add_axes([0.93, 0.33, 0.012, 0.12])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax)
    cb.set_label("log10 OR", fontsize=5); cb.ax.tick_params(labelsize=4.5)
    return fig


# ── shift grid ───────────────────────────────────────────────────────────────
def shift_table(mod, fams=F_FAMS):
    """Shift of the error-group median toward the other class per phenotype feature (TP vs TN FDR < 0.05):
    0 = at its own class, 1 = at the other class; dir_* = +1 if the error group is higher than its own class."""
    st = stats()
    d = st[(st.modality == mod) & (st.padj_TP_vs_TN < 0.05) & st.family.isin(fams)].copy()
    tie = (d.median_TP - d.median_TN).abs() < 1e-9  # mostly-zero features: use means
    lo, hi = d.median_TN.where(~tie, d.mean_TN), d.median_TP.where(~tie, d.mean_TP)
    d["dir_FP"], d["dir_FN"] = np.sign(hi - lo), -np.sign(hi - lo)
    for g in ("FP", "FN"):
        e = d[f"median_{g}"].where(~tie, d[f"mean_{g}"])
        pos = ((e - lo) / (hi - lo)).where((hi - lo).abs() > 1e-9)  # position on the TN (0) → TP (1) axis
        pos[d[f"n_{g}"] < 3] = np.nan
        d[f"shift_{g}"] = pos if g == "FP" else 1 - pos
    return d


def _shift_heat(ax, d, rows, g, labels):
    ri, ci = {f: i for i, f in enumerate(rows.feature_col)}, {v: i for i, v in enumerate(VARS)}
    m = np.full((len(rows), len(VARS)), np.nan)
    sgn = np.ones_like(m)
    for r in d.itertuples():
        v = getattr(r, f"shift_{g}")
        if r.feature_col in ri and not np.isnan(v) and round(v, 1) >= 0:  # only shifts toward the other class
            m[ri[r.feature_col], ci[r.variable]] = round(v, 1) + 0.0
            sgn[ri[r.feature_col], ci[r.variable]] = getattr(r, f"dir_{g}")
    ax.imshow(np.clip(m, 0, 1) * sgn, cmap=SHIFT_CMAP, vmin=-1, vmax=1, aspect="auto")
    for i, j in zip(*np.where(~np.isnan(m))):
        v = m[i, j]
        ax.text(j, i, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=3.0,
                fontweight="bold" if v > 1 else "normal", color="w" if v > 0.6 else "k")
    few = d[d.feature_col.isin(ri) & (d[f"n_{g}"] < 3)]
    ax.scatter(few.variable.map(ci), few.feature_col.map(ri), marker="x", s=6, color="0.6", lw=0.5)
    for i, j in zip(*np.where(~np.isnan(m))):  # small groups (3–4 errors): hatched, colour kept
        if d.loc[d.variable == VARS[j], f"n_{g}"].max() < SMALL_N:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="//////", ec="0.45", lw=0, alpha=0.6))
    for v in VARS:  # number of errors per variable
        n = d.loc[d.variable == v, f"n_{g}"]
        ax.text(ci[v], 1.004, int(n.max()) if len(n) else 0, ha="center", va="bottom", fontsize=3.8,
                color="firebrick" if len(n) and n.max() < SMALL_N else "0.45", transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(VARS))); ax.set_xticklabels([vname(v) for v in VARS], rotation=60, ha="right", fontsize=4.3)
    for tl, v in zip(ax.get_xticklabels(), VARS):
        tl.set_color(vcol(v))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([_long(r) for r in rows.itertuples()] if labels else [], fontsize=4.3)
    ax.set_xticks(np.arange(len(VARS) + 1) - .5, minor=True); ax.set_yticks(np.arange(len(rows) + 1) - .5, minor=True)
    ax.grid(which="minor", color="0.92", lw=0.4); ax.tick_params(which="both", length=0, pad=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    prev = None
    for i, f in enumerate(rows.family):
        if prev is not None and f != prev:
            ax.axhline(i - 0.5, color="0.4", lw=0.6)
        prev = f


def _shift_colorbar(fig, rect):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap=SHIFT_CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([-1, -0.5, 0, 0.5, 1]); cb.set_ticklabels(["1", "0.5", "0", "0.5", "1"])
    cb.ax.tick_params(labelsize=4.5, length=1.5); cb.outline.set_visible(False)
    cb.ax.set_title("shift toward the other class", fontsize=4.8, pad=2)
    cb.ax.text(-0.04, 0.5, "lower than\nown class", transform=cb.ax.transAxes, fontsize=4.3, color="#2166ac",
               va="center", ha="right")
    cb.ax.text(1.04, 0.5, "higher than\nown class", transform=cb.ax.transAxes, fontsize=4.3, color="#b2182b",
               va="center", ha="left")


KEY_CASES = {  # group medians TN, FN, FP, TP per schematic example
    "FP": [("rises toward TP", dict(TN=2, FN=7, FP=4.7, TP=10)),
           ("falls toward TP", dict(TN=10, FN=4, FP=5.2, TP=2)),
           ("moves away from TP", dict(TN=5, FN=8, FP=3, TP=9))],
    "FN": [("falls to TN level", dict(TN=2, FN=2, FP=4, TP=10)),
           ("falls beyond TN", dict(TN=5, FN=2, FP=6, TP=9)),
           ("rises toward TN", dict(TN=10, FN=7, FP=6, TP=2))],
}


def _key_example(fig, x, y, w, h, title, med, g):
    own, other = ("TN", "TP") if g == "FP" else ("TP", "TN")
    ax = fig.add_axes([x, y, w * 0.56, h])
    for i, q in enumerate(OUTS):
        on = q in (own, other, g)
        ax.add_patch(plt.Rectangle((i - 0.3, med[q] - 0.9), 0.6, 1.8, fc=OUT_COL[q], alpha=0.55 if on else 0.12, lw=0))
        ax.plot([i - 0.3, i + 0.3], [med[q]] * 2, color="k" if on else "0.75", lw=0.7)
    for q in (own, other):
        ax.axhline(med[q], color=OUT_COL[q], lw=0.5, ls=":", zorder=0)
    gi = OUTS.index(g)
    ax.annotate("", xy=(gi + 0.42, med[g]), xytext=(gi + 0.42, med[own]),
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.2", shrinkA=0, shrinkB=0))
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 12)
    ax.set_xticks(range(4)); ax.set_xticklabels(OUTS, fontsize=4)
    for tl, q in zip(ax.get_xticklabels(), OUTS):
        tl.set_color(OUT_COL[q] if q in (own, other, g) else "0.7")
    ax.set_yticks([]); ax.tick_params(length=0, pad=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(f"{g} {title}", fontsize=4.6, loc="left", x=-0.02, pad=2)
    shift = (med[g] - med[own]) / (med[other] - med[own])
    sign, v = np.sign(med[g] - med[own]), round(shift, 1)
    cx = fig.add_axes([x + w * 0.66, y + h * 0.52, w * 0.22, h * 0.36])
    if v >= 0:
        cx.imshow([[np.clip(v, 0, 1) * sign]], cmap=SHIFT_CMAP, vmin=-1, vmax=1, aspect="auto")
        cx.text(0, 0, f"{v:.1f}", ha="center", va="center", fontsize=4.8, fontweight="bold" if v > 1 else "normal",
                color="w" if v > 0.6 else "k")
    cx.set_xticks([]); cx.set_yticks([])
    for sp in cx.spines.values():
        sp.set_color("0.6"); sp.set_linewidth(0.4)
    note = (f"{'higher' if sign > 0 else 'lower'} than {own}\n→ {'red' if sign > 0 else 'blue'}"
            + (", bold (> 1)" if v > 1 else "") if v >= 0 else "away from\n" + other + " → hidden")
    num = f"{med[g]:g} − {med[own]:g}" if g == "FP" else f"{med[own]:g} − {med[g]:g}"
    den = f"{med[other]:g} − {med[own]:g}" if g == "FP" else f"{med[own]:g} − {med[other]:g}"
    fig.text(x + w * 0.77, y + h * 0.46, f"({num})\n/ ({den})\n= {shift:.2f}\n{note}", fontsize=3.6, color="0.25",
             ha="center", va="top", linespacing=1.2)


def _shift_key(fig, bottom, height, blocks=((0.20, 0.38), (0.60, 0.38))):
    """Schematic how-to-read key, one block above each grid column: FP (TN → TP) and FN (TP → TN)."""
    for (x0, wd), g in zip(blocks, ("FP", "FN")):
        fig.patches.append(plt.Rectangle((x0 - 0.005, bottom - height * 0.30), wd + 0.01, height * 1.67,
                                         transform=fig.transFigure, fc=OUT_COL[g], alpha=0.05, ec=OUT_COL[g], lw=0.5))
        fig.text(x0, bottom + height * 1.28, "false positives (left grids): shift from TN toward TP" if g == "FP" else
                 "false negatives (right grids): shift from TP toward TN", fontsize=5.2, fontweight="bold",
                 color=OUT_COL[g])
        w = wd / len(KEY_CASES[g])
        for k, (t, med) in enumerate(KEY_CASES[g]):
            _key_example(fig, x0 + k * w + 0.004, bottom, w - 0.006, height, t, med, g)


@_rc
def shift_grid():
    st = {m: shift_table(m) for m in ("BM", "PB")}
    rows = {m: _rows(d, F_FAMS) for m, d in st.items()}
    H = 54 + sum(ROW * len(r) + 34 for r in rows.values())
    fig = plt.figure(figsize=(177 * MM, H * MM))
    f = lambda mm: 1 - mm / H  # noqa: E731
    fig.text(0.01, f(1), "Per feature with TP vs TN FDR < 0.05. False positives: (median FP − median TN) / (median TP − median TN); "
             "false negatives: (median TP − median FN) / (median TP − median TN).\n0 = like its own class, 1 = like the other "
             "class; red = error group higher than its own class (FP vs TN, FN vs TP), blue = lower; bold = beyond the other class (> 1).\n"
             "Cells < 0 (no shift) hidden; × = < 3 errors; hatched = 3–4 errors (n above columns); "
             "means replace medians where TN and TP medians tie.", fontsize=5, color="0.3", va="top")
    _shift_colorbar(fig, [0.74, f(15), 0.16, 0.006])
    fig.text(0.01, f(26), "How to read\na cell", fontsize=5.6, fontweight="bold", va="top")
    _shift_key(fig, f(48), 18 / H)
    y = 56
    for k, m in enumerate(("BM", "PB")):
        _panel_label(fig, 0.01, f(y), "AB"[k], f"{m} model")
        hh = ROW * len(rows[m])
        for j, (g, t) in enumerate((("FP", "false positives: shift from TN toward TP  (n FP above columns)"),
                                    ("FN", "false negatives: shift from TP toward TN  (n FN above columns)"))):
            ax = fig.add_axes([(0.20, 0.60)[j], f(y + 8 + hh), 0.38, hh / H])
            _shift_heat(ax, st[m], rows[m], g, labels=j == 0)
            ax.set_title(t, fontsize=5.6, fontweight="bold", color=OUT_COL[g], pad=9)
        y += 8 + hh + 34
    return fig


@_rc
def shift_grid_morph(mod):
    """Morphometry counterpart of shift_grid (one modality per figure); rows = features drawn in either grid."""
    d = shift_table(mod, MORPH)
    vis = lambda g: (d[f"shift_{g}"].round(1) >= 0) & (d[f"n_{g}"] >= 3)  # noqa: E731
    rows = _rows(d[d.feature_col.isin(d[vis("FP") | vis("FN")].feature_col)], MORPH)
    H = 26 + ROW * len(rows) + 30
    fig = plt.figure(figsize=(177 * MM, H * MM))
    f = lambda mm: 1 - mm / H  # noqa: E731
    fig.text(0.01, f(1), f"{mod} model. Same measure and encoding as the composition shift grid (see its key): per morphometric "
             "feature with TP vs TN FDR < 0.05, shift of the error-group median toward the other class.\n0 = like its own "
             "class, 1 = like the other class; red = higher than own class, blue = lower; bold = > 1; cells < 0 hidden; "
             "× = < 3 errors; hatched = 3–4 errors.", fontsize=5, color="0.3", va="top")
    _shift_colorbar(fig, [0.74, f(12), 0.16, 0.006])
    hh = ROW * len(rows)
    for j, (g, t) in enumerate((("FP", "false positives: shift from TN toward TP\n(n FP above columns)"),
                                ("FN", "false negatives: shift from TP toward TN\n(n FN above columns)"))):
        ax = fig.add_axes([(0.20, 0.60)[j], f(26 + hh), 0.38, hh / H])
        _shift_heat(ax, d, rows, g, labels=j == 0)
        ax.set_title(t, fontsize=5.6, fontweight="bold", color=OUT_COL[g], pad=9)
    return fig


# ── variable-specific examples ───────────────────────────────────────────────
PCT = {"low_quality_all": 100, "top_apl_bilobed": 100, "top_vacuolated": 100, "top_apl_hypergranulated": 100}
CONTRASTS = (("TP", "TN"), ("FN", "TP"), ("FP", "TN"))


def feature_label(c):
    if c in ("low_quality_all", "n_cells"):
        return {"low_quality_all": "low-quality cells (%)", "n_cells": "cells / sample"}[c]
    if c.startswith("top_"):
        return f"top-attention cells: {c[4:].replace('apl_', '').replace('_', ' ')}" + (" (%)" if c in PCT else "")
    c = c.replace("_proportion", " (%)").replace("_median", "").replace("-", " – ").replace("entire_", "cell ")
    return c.replace("Myeloid_precursors", "Myeloid precursors").replace("apl_", "").replace("_", " ")


@_rc
def variable_examples(panels):
    """panels: [(variable, modality, feature)]. Feature = a sample-level column of load_cohort(modality)
    (e.g. "Promyelocytes_proportion", "Blasts_nucleus_area_median"), "low_quality_all" / "n_cells", or a
    top-attention summary (e.g. "top_apl_bilobed"). p values are unadjusted (illustrative)."""
    S = samples()
    cohort = {m: load_cohort(m) for m in {mod for _, mod, _ in panels}}
    nc = 4; nr = -(-len(panels) // nc)
    H = 14 + 62 * nr + 14
    fig = plt.figure(figsize=(177 * MM, H * MM))
    fig.text(0.01, 1 - 1 / H, "Boxes = IQR, whiskers = 1.5 × IQR, dots = samples (groups < 60). "
             "Cliff's δ and unadjusted Mann–Whitney p for TP vs TN, FN vs TP and FP vs TN below each panel (illustrative selection).",
             fontsize=5, color="0.35", va="top")
    gs = fig.add_gridspec(nr, nc, left=0.07, right=0.98, top=1 - 14 / H, bottom=24 / H, wspace=0.45, hspace=0.75)
    for k, (v, mod, c) in enumerate(panels):
        ax = fig.add_subplot(gs[k // nc, k % nc])
        o = S[(S.modality == mod) & (S.variable == v)]
        x = o if c in o else o.merge(cohort[mod][["database_id", "album_id", c]], on=["database_id", "album_id"])
        g = {grp: x.loc[x.outcome == grp, c].dropna() * PCT.get(c, 1) for grp in OUTS}
        bp = ax.boxplot([g[o_] for o_ in OUTS], widths=0.6, showfliers=False, patch_artist=True,
                        medianprops=dict(color="k", lw=0.7))
        for bx, o_ in zip(bp["boxes"], OUTS):
            bx.set(facecolor=OUT_COL[o_], alpha=0.5, lw=0.4)
        for i, o_ in enumerate(OUTS, 1):
            if len(g[o_]) < 60:
                ax.scatter(i + np.random.default_rng(0).uniform(-0.15, 0.15, len(g[o_])), g[o_], s=2.5,
                           color=OUT_COL[o_], lw=0, zorder=3)
        ax.set_xticks(range(1, 5)); ax.set_xticklabels([f"{o_}\n{len(g[o_])}" for o_ in OUTS], fontsize=4.3)
        ax.set_title(f"{vname(v)} · {mod}\n{feature_label(c)}", fontsize=5, color=vcol(v))
        ax.tick_params(labelsize=4.3, length=2, pad=1); ax.spines[["top", "right"]].set_visible(False)
        lines = []
        for a_, b_ in CONTRASTS:
            if min(len(g[a_]), len(g[b_])) < 3:
                lines.append(f"{a_} vs {b_}: n < 3")
                continue
            u, p = mannwhitneyu(g[a_], g[b_])
            lines.append(f"{a_} vs {b_}: δ = {2 * u / (len(g[a_]) * len(g[b_])) - 1:+.2f}, p = {p:.2g}")
        ax.text(0.5, -0.17, "\n".join(lines), transform=ax.transAxes, ha="center", va="top", fontsize=4.2,
                color="0.25", linespacing=1.3)
    return fig
