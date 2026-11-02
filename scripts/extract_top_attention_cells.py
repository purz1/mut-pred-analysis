"""Top-attention cells per sample for manual image review, filtered by target, outcome, cell type and modality.

Run from the repository root, e.g. NPM1 true positives, blasts only, top 20 per sample in BM and PB:
    .venv/bin/python scripts/extract_top_attention_cells.py --var NPM1 --outcome TP --cell-types Blasts --mod BM PB --top 20

Filters are applied in order: modality model (RUN_PATHS) → samples with the given outcome for --var → cells of
--cell-types → the --top cells per sample by attention for --var → the --samples samples per model with the highest
pred_prob (lowest for the predicted negatives TN and FN, i.e. the most confident predictions first). rank_all and share are computed over all cells of
the sample (before the cell-type filter), rank over the cells left after it. Samples without attention scores are
skipped (counted in the printout).

Outputs (git-ignored), data/<DATA_VERSION_IDENTIFIER>/derived/top_attention_cells/:
    <name>.csv          one row per cell: modality, model, fold, database_id, album_id, cell_id, cell_label, rank,
                        rank_all, n_cells, attention, share, true_label, pred_prob
Also writes derived/cell_crops/lists/<name>.csv; fetch the images with
    .venv/bin/python scripts/extract_cell_crops.py data/<DATA_VERSION_IDENTIFIER>/derived/cell_crops/lists/<name>.csv
<name> = <var>_<outcome>_<cell types>_<models>_top<n>[_s<samples>], or --name.
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from core.data_loading import RUN_PATHS  # noqa: E402
from manuscript_figures import cells, config as C  # noqa: E402

SAMPLE = ["database_id", "album_id"]
OUTCOMES = {"TP": (1, 1), "TN": (0, 0), "FP": (0, 1), "FN": (1, 0)}  # (true, pred)
OUT = C.DERIVED / "top_attention_cells"


def load(model, var):
    sc, t, p, pr = (f"cell_attention_score_{var}", f"{var}_true_label", f"{var}_pred_label", f"{var}_pred_prob")
    cols = ["modality", *SAMPLE, "cell_id", "cell_label", sc, t, p, pr]
    files = sorted(glob.glob(f"{REPO / RUN_PATHS[model]}/fold_*/cell_attention_scores.csv"))
    return pd.concat(pd.read_csv(f, usecols=cols).assign(fold=Path(f).parent.name) for f in files).astype(
        {"database_id": str, "album_id": str}).rename(
        columns={sc: "attention", t: "true_label", p: "pred_label", pr: "pred_prob"})


def select(model, var, outcome, cell_types, top, samples):
    a = load(model, var)
    if outcome != "all":
        true, pred = OUTCOMES[outcome]
        a = a[(a.true_label == true) & (a.pred_label == pred)]
    n = a.groupby(SAMPLE).ngroups
    a = a.dropna(subset=["attention"]).sort_values(SAMPLE + ["attention"], ascending=[True, True, False])
    g = a.groupby(SAMPLE).attention
    a = a.assign(rank_all=g.cumcount() + 1, n_cells=g.transform("size"), share=a.attention / g.transform("sum"))
    if cell_types:
        a = a[a.cell_label.isin(cell_types)]
    a = a.assign(rank=a.groupby(SAMPLE).cumcount() + 1)
    if top:
        a = a[a["rank"] <= top]
    n_sel = a.groupby(SAMPLE).ngroups
    if samples:
        p = a.groupby(SAMPLE).pred_prob.first()
        keep = (p.nsmallest(samples) if outcome in ("TN", "FN") else p.nlargest(samples)).index
        a = a[a.set_index(SAMPLE).index.isin(keep)]
    print(f"[{model}] {var} {outcome}: {n} samples, {n - n_sel} without attention scores or selected cells, "
          f"{a.groupby(SAMPLE).ngroups} kept, {len(a)} cells")
    return a.assign(model=model)[["modality", "model", "fold", *SAMPLE, "cell_id", "cell_label", "rank", "rank_all",
                                  "n_cells", "attention", "share", "true_label", "pred_prob"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--var", required=True, help="target variable, e.g. NPM1")
    parser.add_argument("--outcome", default="TP", choices=[*OUTCOMES, "all"])
    parser.add_argument("--cell-types", nargs="*", default=[], help="cell_label values, e.g. Blasts (default: all)")
    parser.add_argument("--mod", nargs="+", default=["BM", "PB"], choices=list(RUN_PATHS), help="models")
    parser.add_argument("--top", type=int, default=20, help="cells per sample after the filters (0: all)")
    parser.add_argument("--samples", type=int, default=0,
                        help="samples per model, most confident first: highest pred_prob, lowest for TN/FN (0: all)")
    parser.add_argument("--name", help="output name (default: built from the arguments)")
    args = parser.parse_args()

    ct = "-".join(args.cell_types) or "allcells"
    name = args.name or f"{args.var}_{args.outcome}_{ct}_{'-'.join(args.mod)}_top{args.top or 'all'}" + (f"_s{args.samples}" if args.samples else "")
    df = pd.concat(select(m, args.var, args.outcome, args.cell_types, args.top, args.samples) for m in args.mod)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    cells.write_list(list(df.groupby("modality")), name)  # FUSION cells go to their own modality's crops
    print(f"{len(df)} cells → {OUT / name}.csv\nfetch images:\n  .venv/bin/python scripts/extract_cell_crops.py "
          f"{(C.CROPS / 'lists' / f'{name}.csv').relative_to(REPO)}")


if __name__ == "__main__":
    main()
