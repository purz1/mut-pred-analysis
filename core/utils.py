import json
import os
import re

import matplotlib.pyplot as plt

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


def get_readable_variable_name(target: str) -> str:
    if target in CUSTOM_VARIABLES:
        return CUSTOM_VARIABLES[target]
    cell_key = target.removesuffix("_proportion")
    if cell_key in CELL_NAMES:
        return CELL_NAMES[cell_key]
    t = target.replace("_OR_", " / ").replace("_AND_", " + ")
    t = t.replace("_", " ")
    return t
