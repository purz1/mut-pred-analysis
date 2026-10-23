"""Inputs and outputs of the manuscript figures. Model runs and the data version are set in the repository config.py."""
from pathlib import Path

from config import DATA_VERSION_IDENTIFIER

REPO = Path(__file__).resolve().parents[1]
RUN = "AML_V15_OPTIMIZED_PARAMETERS"  # aml-mil run for Fig 2/3 and calibration (stand-in until MANUSCRIPT_V1)

DERIVED = REPO / f"data/{DATA_VERSION_IDENTIFIER}/derived"
FINGERPRINTS = DERIVED / "fig4_fingerprints"   # scripts/extract_fig4_fingerprint_results.py
ERRORS = DERIVED / "error_analysis"            # scripts/extract_error_attention_cache.py, extract_error_outcome_stats.py
AML_MIL_FIGS = REPO.parent / "aml-mil/results" / RUN / "figures"

OUT = REPO / "figures/manuscript"
DPI = 300


def save(fig, name):
    """Save as PNG (viewing) and PDF (vector, submission) under figures/manuscript/."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=DPI)
    fig.savefig(OUT / f"{name}.pdf")
    return fig
