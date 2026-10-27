"""FAB supplements: composition and positive controls, model performance and errors per FAB group, fingerprint per FAB group.

fab_composition()                 FAB class counts (with missing) and prevalence of every target per class (positive controls: PML-RARA in M3 …)
fab_variable(variable)            one target: prevalence and outcomes per FAB class, the model probability of true positives and true negatives per FAB
                                  group and, where both groups have enough positives, the target's fingerprint within M0–M2 and M4–M5
Inputs: derived/fab (scripts/extract_fab_analysis.py, extract_fab_features.py).
"""
from functools import cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config as C
from .style import MM, W, VARS, vname, vcol

GROUPS = ["M0–M2", "M3", "M4–M5"]
GROUP_COL = {"M0–M2": "#3a6fb0", "M3": "#8c8c8c", "M4–M5": "#e08a1e"}
CLASSES = ["M0", "M1", "M2", "M3", "M4", "M5"]
FAMILY_COL = {"proportion": "#455fa2", "dysplasia": "#a92b27", "size": "#4e867a", "N:C": "#9b9231",
              "nuclear shape": "#5f4480", "cell shape": "#b95a1e", "qc": "#8c8c8c"}
CONTRAST = ("M0–M2", "M4–M5")


@cache
def _t(name):
    return pd.read_csv(C.FAB / f"{name}.csv")


def _p(p):
    return r"$\it{P}$ < 0.001" if p < 0.001 else rf"$\it{{P}}$ = {p:.3f}"


def fab_composition(model="BM"):
    comp = _t("fab_composition")
    comp = comp[comp.model == model].set_index("fab").n
    prev = _t("fab_target_prevalence")
    prev = prev[(prev.model == model) & (prev.level == "class")]
    variables = [v for v in VARS if v in set(prev.variable)]
    with plt.rc_context({"font.size": 5.5}):
        fig = plt.figure(figsize=(W * MM, (30 + 4.2 * len(variables)) * MM))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], left=0.06, right=0.97, top=1 - 8 / (30 + 4.2 * len(variables)), bottom=14 / (30 + 4.2 * len(variables)), wspace=0.5)
        ax = fig.add_subplot(gs[0])
        order = CLASSES + ["missing"]
        vals = [int(comp.get(c, 0)) for c in order]
        ax.bar(range(len(order)), vals, color=["#3a6fb0"] * 3 + ["#8c8c8c"] + ["#e08a1e"] * 2 + ["#d9d9d9"], width=0.7)
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=5)
        ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
        ax.set_ylabel("Samples")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("FAB class", fontsize=6, fontweight="bold")
        ax = fig.add_subplot(gs[1])
        pv = prev.pivot(index="variable", columns="group", values="prevalence").reindex(index=variables, columns=CLASSES) * 100
        n = prev.pivot(index="variable", columns="group", values="n_pos").reindex(index=variables, columns=CLASSES)
        im = ax.imshow(pv.to_numpy(float), cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(len(variables)):
            for j in range(len(CLASSES)):
                if pd.notna(pv.iat[i, j]):
                    ax.text(j, i, f"{pv.iat[i, j]:.0f}\n({int(n.iat[i, j])})", ha="center", va="center", fontsize=4,
                            color="white" if pv.iat[i, j] > 55 else "k")
        ax.set_xticks(range(len(CLASSES)), CLASSES)
        ax.set_yticks(range(len(variables)), [vname(v) for v in variables])
        for t, v in zip(ax.get_yticklabels(), variables):
            t.set_color(vcol(v))
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        ax.set_title("Positive samples, % of the class (n)", fontsize=6, fontweight="bold", pad=14)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=4.5)
    return fig


OUTCOME_COL = {"TP": "#8b1a1a", "FN": "#f39c34", "FP": "#d62728", "TN": "#3a6fb0"}
MIN_FEATURES = 20  # fingerprint panels need this many features with a δ in both groups


def fab_variable(variable, n_top=10, modality="BM"):
    prev, outc = (_t(n) for n in ("fab_target_prevalence", "fab_outcomes"))
    prev = prev[(prev.variable == variable) & (prev.level == "class") & (prev.model == modality)].set_index("group").reindex(CLASSES)
    eff = _t("fab_feature_effects")
    a, b = (f"delta_{g}" for g in CONTRAST)
    eff = eff[(eff.variable == variable) & (eff.modality == modality)].dropna(subset=[a, b]).copy()
    fingerprint = len(eff) >= MIN_FEATURES
    H = 128 if fingerprint else 62
    with plt.rc_context({"font.size": 5.5}):
        fig = plt.figure(figsize=(W * MM, H * MM))
        top, bot = (1 - 10 / H, 1 - 54 / H) if fingerprint else (1 - 10 / H, 12 / H)
        gs = fig.add_gridspec(1, 4, left=0.06, right=0.98, top=top, bottom=bot, wspace=0.55, width_ratios=[1, 1, 1, 1])
        ax = fig.add_subplot(gs[0])
        ax.bar(range(6), prev.prevalence.fillna(0) * 100, color=[GROUP_COL[g] for g in ("M0–M2",) * 3 + ("M3",) + ("M4–M5",) * 2], width=0.7)
        for i, (pr, n) in enumerate(zip(prev.prevalence, prev.n_pos)):
            ax.text(i, (0 if pd.isna(pr) else pr * 100) + 1, f"{int(n) if pd.notna(n) else 0}", ha="center", va="bottom", fontsize=4.5)
        ax.set_xticks(range(6), CLASSES); ax.set_ylabel("Positive samples (%)"); ax.set_title("Prevalence (n positive)", fontsize=5.5)
        ax = fig.add_subplot(gs[1])
        o = outc[(outc.variable == variable) & (outc.model == modality)].set_index("fab").reindex(CLASSES).fillna(0)
        bottom = np.zeros(6)
        for k in ("TP", "FN", "FP", "TN"):
            ax.bar(range(6), o[k], bottom=bottom, color=OUTCOME_COL[k], width=0.7, label=k, lw=0)
            bottom += o[k].to_numpy()
        ax.set_xticks(range(6), CLASSES); ax.set_ylabel("Samples"); ax.set_title(f"Outcomes ({modality} model)", fontsize=5.5)
        ax.set_ylim(0, max(bottom.max(), 1) * 1.25)
        ax.legend(fontsize=4.5, frameon=False, ncol=4, loc="upper left", columnspacing=0.8, handlelength=1)
        sc, scc = _t("fab_scores"), _t("fab_score_contrast")
        sc = sc[(sc.variable == variable) & (sc.model == modality)]
        scc = scc[(scc.variable == variable) & (scc.model == modality)].set_index("true")
        rng = np.random.default_rng(0)
        for k, (true, title) in enumerate(((1, "Model score, true positives"), (0, "Model score, true negatives"))):
            ax = fig.add_subplot(gs[2 + k])
            present = [g for g in GROUPS if ((sc.true == true) & (sc.group == g)).any()]
            ys = [sc.loc[(sc.true == true) & (sc.group == g), "prob"].to_numpy() for g in present]
            bp = ax.boxplot(ys, positions=range(len(present)), widths=0.6, whis=0, showfliers=False, showcaps=False, patch_artist=True,
                            medianprops=dict(color="k", lw=1.2), whiskerprops=dict(lw=0))
            for bx, g in zip(bp["boxes"], present):
                bx.set(facecolor=GROUP_COL[g], alpha=0.35, lw=0.4)
            for x, y in enumerate(ys):
                ax.scatter(x + rng.uniform(-0.2, 0.2, len(y)), y, s=3, color=".3", lw=0, zorder=3)
            ax.set_xticks(range(len(present)), [f"{g}\n(n={len(y)})" for g, y in zip(present, ys)], fontsize=4.8)
            ax.set_ylim(-0.02, 1.12); ax.set_title(title, fontsize=5.5); ax.spines[["top", "right"]].set_visible(False)
            if k == 0:
                ax.set_ylabel(f"Predicted probability ({modality})")
            if true in scc.index and pd.notna(scc.padj[true]) and all(g in present for g in CONTRAST):
                ax.text(0.5, 1.0, _p(scc.padj[true]), transform=ax.transAxes, ha="center", va="bottom", fontsize=5)
        fig.text(0.06, 1 - 2 / H, vname(variable), fontsize=7, fontweight="bold", color=vcol(variable), va="center")
        if fingerprint:
            eff["label"] = eff.cell_type.fillna("").str.replace("_", " ") + " – " + eff.feature.fillna(eff.feature_col).str.replace("_", " ")
            rest = eff[~eff.hypothesis]
            sel = pd.concat([eff[eff.hypothesis], rest.reindex(rest.delta_diff.abs().sort_values(ascending=False).index).head(n_top)]).iloc[::-1].reset_index(drop=True)
            gs2 = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], left=0.06, right=0.98, top=1 - 68 / H, bottom=14 / H, wspace=0.8)
            ax = fig.add_subplot(gs2[0])
            for fam, g in eff.groupby("family"):
                ax.scatter(g[a], g[b], s=4, color=FAMILY_COL.get(fam, "k"), alpha=0.7, lw=0, label=fam)
            lim = max(0.5, np.nanmax(np.abs(eff[[a, b]].to_numpy())) * 1.05)
            ax.plot([-lim, lim], [-lim, lim], color="0.5", lw=0.5, ls="--")
            ax.axhline(0, color="0.8", lw=0.3); ax.axvline(0, color="0.8", lw=0.3)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            ax.set_xlabel(f"Cliff's δ in {CONTRAST[0]}"); ax.set_ylabel(f"Cliff's δ in {CONTRAST[1]}")
            ax.set_title(f"Fingerprint ({modality}): {len(eff)} features, ρ = {eff[[a, b]].corr(method='spearman').iat[0, 1]:.2f}", fontsize=5.5)
            for _, row in eff[eff.hypothesis].iterrows():
                ax.scatter(row[a], row[b], s=14, facecolor="none", edgecolor="k", lw=0.7)
            ax.legend(fontsize=4.3, frameon=False, loc="lower right", markerscale=1.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax2 = fig.add_subplot(gs2[1])
            for i, row in sel.iterrows():
                ax2.plot([row[a], row[b]], [i, i], color="0.6", lw=0.8)
                ax2.scatter(row[a], i, s=9, color=GROUP_COL[CONTRAST[0]], zorder=3, lw=0)
                ax2.scatter(row[b], i, s=9, color=GROUP_COL[CONTRAST[1]], zorder=3, lw=0)
                if pd.notna(row.get("padj_interaction")) and row.padj_interaction < 0.05:
                    ax2.text(1.0, i, "*", transform=ax2.get_yaxis_transform(), ha="right", va="center", fontsize=8)
            ax2.axvline(0, color="0.8", lw=0.3)
            ax2.set_yticks(range(len(sel)), [("◆ " if h else "") + t for t, h in zip(sel.label, sel.hypothesis)], fontsize=4.8)
            ax2.set_xlabel("Cliff's δ"); ax2.spines[["top", "right"]].set_visible(False)
            ax2.set_title("Pre-specified (◆) and largest differences between the groups", fontsize=5.5, pad=6)
            ax2.legend(handles=[plt.Line2D([], [], marker="o", color=GROUP_COL[g], lw=0, ms=3, label=g) for g in CONTRAST],
                       fontsize=5, frameon=False, loc="lower right")
    return fig
