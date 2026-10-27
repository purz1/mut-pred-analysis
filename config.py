import os as _os
from pathlib import Path as _Path

# ── Global flags ───────────────────────────────────────────────────────────────
USE_DUMMY_DATA = False

# ── Data versioning ────────────────────────────────────────────────────────────
# Used in both folder names and filenames. Change this when regenerating data.
DATA_VERSION_IDENTIFIER = "03-06-2026"
SAVE_FILES = True

# ── Shared image-derived data ──────────────────────────────────────────────────
# Written by hematoscope-research into 2-mut-pred/data (next to this repository) and
# also read by aml-mil. Set MUT_PRED_DATA_DIR to use another location.
# FEATURE_VERSION is the cohort/label version the features were generated from; it is
# separate from DATA_VERSION_IDENTIFIER (this repository's analysis data version).
FEATURE_VERSION = "07-04-2026"
SHARED_DATA_DIR = _Path(
    _os.environ.get("MUT_PRED_DATA_DIR", _Path(__file__).resolve().parent.parent / "data")
)
TRAINING_FEATURES_DIR = SHARED_DATA_DIR / f"aml_dg_de-novo-and-secondary-{FEATURE_VERSION}"
VENEX_FEATURES_DIR    = SHARED_DATA_DIR / f"venex_bm-{FEATURE_VERSION}"
SWEDEN_FEATURES_DIR   = SHARED_DATA_DIR / f"sweden_bm-{FEATURE_VERSION}"
HEHR_FEATURES_DIR     = SHARED_DATA_DIR / "hehr_aml"


def resolve_data_path(path: str) -> str:
    """Return the real path, or the _DUMMY companion path if USE_DUMMY_DATA is True."""
    if not USE_DUMMY_DATA:
        return path
    p = _Path(path)
    return str(p.parent / f"{p.stem}_DUMMY{p.suffix}")


# ── Model prediction run paths ─────────────────────────────────────────────────
_V = DATA_VERSION_IDENTIFIER
BM_RUN_PATH     = f"data/{_V}/model_predictions/AML_V15_OPTIMIZED_PARAMETERS/multilabel/scemila/BM/All cells/run_2026-05-19_11-37"
PB_RUN_PATH     = f"data/{_V}/model_predictions/AML_V15_OPTIMIZED_PARAMETERS/multilabel/scemila/PB/All cells/run_2026-05-19_11-40"
FUSION_RUN_PATH = f"data/{_V}/model_predictions/AML_V15_OPTIMIZED_PARAMETERS/multilabel/scemila/FUSION/All cells/run_2026-05-05_17-01"

# ── Clinical data paths ────────────────────────────────────────────────────────
SURVIVAL_DATA_PATH_BM   = resolve_data_path(f"data/{_V}/survival/aml_mut_pred_survival_data_bm_{_V}.xlsx")
SURVIVAL_DATA_PATH_PB   = resolve_data_path(f"data/{_V}/survival/aml_mut_pred_survival_data_pb_{_V}.xlsx")
VAF_DATA_PATH_BM        = resolve_data_path(f"data/{_V}/vaf/aml_mut_pred_vaf_values_bm_{_V}.xlsx")
VAF_DATA_PATH_PB        = resolve_data_path(f"data/{_V}/vaf/aml_mut_pred_vaf_values_pb_{_V}.xlsx")
CLINICAL_DATA_PATH_BM   = resolve_data_path(f"data/{_V}/unprocessed_clinical/aml_mut_pred_unprocessed_clinical_data_bm_{_V}.csv")
CLINICAL_DATA_PATH_PB   = resolve_data_path(f"data/{_V}/unprocessed_clinical/aml_mut_pred_unprocessed_clinical_data_pb_{_V}.csv")
MUTATION_LABELS_PATH_BM = resolve_data_path(f"data/{_V}/mutation_labels/aml_mut_pred_sample_ids_bm_{_V}.xlsx")
MUTATION_LABELS_PATH_PB = resolve_data_path(f"data/{_V}/mutation_labels/aml_mut_pred_sample_ids_pb_{_V}.xlsx")

# ── Variables ──────────────────────────────────────────────────────────────────
SELECTED_VARIABLES = [
    # APL
    "PML-RARA",
    # CBF rearrangements
    "CBFB-MYH11",
    "RUNX1-RUNX1T1_OR_t_8_21",
    # Epigenetic mutations
    "IDH_mutations",
    "IDH2",
    "CpG_mutations",
    # NPM1
    "NPM1",
    "NPM1_AND_FLT3_ITD",
    # Signaling mutations
    "FLT3_mutations",
    "FLT3_ITD",
    "KIT",
    "RTK_RAS_mutations",
    # Secondary AML-related
    "complex_karyotype",
    "monosomal_karyotype",
    "monosomy5_del_5q",
    "TP53",
    "STAG2",
]

# ── Display settings ───────────────────────────────────────────────────────────
BM_COLOR     = "#581CA2"
PB_COLOR     = "#E35252"
FUSION_COLOR = "#9A9696"

FONTSIZE        = 7
SAVE_FIGURES    = True
FIGURE_SAVE_DIR = "figures"
FIGURE_DPI      = 300
