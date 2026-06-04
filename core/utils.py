import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Configurable defaults — notebooks can override by setting these module attributes
# e.g.: import utils; utils.CELL_NAMES = CELL_NAMES
SAVE_FIGURES = True
FIGURE_SAVE_DIR = "figures"
FIGURE_DPI = 300
FIGURE_MAX_WIDTH = 6
FONTSIZE = 7
CELL_NAMES: dict = {}
CUSTOM_VARIABLES: dict = {}


def save_and_show(fig, title="", figure_type=None):
    """Cap width to FIGURE_MAX_WIDTH, optionally save at FIGURE_DPI, then display.
    figure_type: 'main' or 'supp' prefixes the filename; None skips saving."""
    w, h = fig.get_size_inches()
    if w > FIGURE_MAX_WIDTH:
        fig.set_size_inches(FIGURE_MAX_WIDTH, h * FIGURE_MAX_WIDTH / w)
    if SAVE_FIGURES and figure_type is not None:
        os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)
        name = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower() or "figure"
        fig.savefig(
            f"{FIGURE_SAVE_DIR}/{figure_type}_{name}.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    plt.show()


def get_other_variables(
    label_map_path: str,
    selected_variables: list[str],
    exclude: list[str] | None = None,
) -> list[str]:
    """Return variables in the label map that are not in selected_variables or exclude."""
    label_map = json.load(open(label_map_path))
    skip = set(selected_variables) | set(exclude or [])
    return sorted(k for k in label_map if k not in skip)


def load_predictions(
    path: str | Path,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Load a predictions.csv from a run directory or a direct file path.

    Expected columns: sample identifiers + {var}_true_label, {var}_pred_label, {var}_pred_prob.
    If variables is given, keep only those variables' columns plus identifiers.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "predictions.csv"
    df = pd.read_csv(p)
    if variables is not None:
        id_cols = [c for c in df.columns if c in ("subject_id", "album_id", "database_id")]
        var_cols = [c for c in df.columns if any(c.startswith(v + "_") for v in variables)]
        df = df[id_cols + var_cols]
    return df


def add_confusion_categories(
    df: pd.DataFrame,
    target_variables: list[str] | None = None,
    keep_original: bool = False,
) -> pd.DataFrame:
    """Add a {variable}_predicted column per variable with TP/TN/FP/FN or NaN.

    Looks for {var}_true_label and {var}_pred_label columns. If target_variables
    is None, auto-detects variables from columns ending in '_true_label'.
    """
    if target_variables is None:
        target_variables = [
            col[: -len("_true_label")]
            for col in df.columns
            if col.endswith("_true_label")
        ]
    df = df.copy()
    for variable in target_variables:
        true_col = f"{variable}_true_label"
        pred_col = f"{variable}_pred_label"
        true = df[true_col]
        pred = df[pred_col]
        has_label = ~true.isna()
        category = pd.Series(index=df.index, dtype=object)
        category[has_label & (true == 1) & (pred == 1)] = "TP"
        category[has_label & (true == 0) & (pred == 0)] = "TN"
        category[has_label & (true == 0) & (pred == 1)] = "FP"
        category[has_label & (true == 1) & (pred == 0)] = "FN"
        df[f"{variable}_predicted"] = category
    if not keep_original:
        drop_cols = [
            col
            for v in target_variables
            for col in (f"{v}_true_label", f"{v}_pred_label", f"{v}_pred_prob")
            if col in df.columns
        ]
        df = df.drop(columns=drop_cols)
    return df


def get_readable_variable_name(target: str) -> str:
    if target in CUSTOM_VARIABLES:
        return CUSTOM_VARIABLES[target]
    cell_key = target.removesuffix("_proportion")
    if cell_key in CELL_NAMES:
        return CELL_NAMES[cell_key]
    t = target.replace("_OR_", " / ").replace("_AND_", " + ")
    t = t.replace("_", " ")
    return t
