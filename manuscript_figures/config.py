"""Inputs and outputs of the manuscript figures. Model runs and the data version are set in the repository config.py."""
from pathlib import Path

from config import DATA_VERSION_IDENTIFIER

REPO = Path(__file__).resolve().parents[1]
RUN = "AML_V15_OPTIMIZED_PARAMETERS"  # aml-mil run for Fig 2/3 and calibration (stand-in until MANUSCRIPT_V1)

DERIVED = REPO / f"data/{DATA_VERSION_IDENTIFIER}/derived"
FINGERPRINTS = DERIVED / "fig4_fingerprints"   # scripts/extract_fig4_fingerprint_results.py
ERRORS = DERIVED / "error_analysis"            # scripts/extract_error_attention_cache.py, extract_error_outcome_stats.py
AML_MIL_FIGS = REPO.parent / "aml-mil/results" / RUN / "figures"

# Pixel size of the morphometry / cell-crop scale (µm per pixel). Morphometry px = BM at the 1232 × 1028 field resolution
# (2464-px fields are halved; see CROP_SCALE) and PB crops. BM: 9.034925854 px/µm (MPP² = (1 / 9.034925854)², area × MPP² = µm²);
# checked: whole-cell diameters of all BM cell types fall in the expected ranges and both field resolutions agree.
# PB: the same value is ASSUMED (PB crops are cut at the morphometry scale), not yet confirmed. Areas only; ratios and
# Cliff's δ are unit-free.
UM_PER_PX = {"BM": 1 / 9.034925854, "PB": 1 / 9.034925854}
UM_PER_PX_UNCONFIRMED = {"PB"}
CROP_SCALE = {2464: 0.5, 1232: 1.0}  # BM field width (px) → factor to the morphometry scale

CROPS = DERIVED / "cell_crops"                  # scripts/extract_cell_crops.py
GALLERY = DERIVED / "gallery"                    # scripts/extract_gallery_cells.py (top cells, clusters, UMAP, tiles)
TOP_CELLS = GALLERY / "top_cells.parquet"

OUT = REPO / "figures/manuscript"
DPI = 300


def save(fig, name):
    """Save as PNG (viewing) and PDF (vector, submission) under figures/manuscript/."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=DPI)
    fig.savefig(OUT / f"{name}.pdf")
    return fig


def to_um(feature, mod):
    """(factor, unit) to convert a morphometry value from pixels: areas → µm²; unit-free measures unchanged."""
    if "area" in feature:
        return UM_PER_PX[mod] ** 2, "µm²"
    return 1.0, ""
