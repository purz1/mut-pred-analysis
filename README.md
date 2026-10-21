# AML Mutation Prediction — Analysis

Analysis code for evaluating machine learning models that predict genetic mutations and cytogenetic aberrations in Acute Myeloid Leukemia (AML) from bone marrow (BM) and peripheral blood (PB) microscopy images.

The models themselves live in a separate `aml-mil` repository. This repo contains the downstream analysis: survival analyses, ELN risk group reclassification, VAF comparisons, and cytomorphology characterization.

---

## Repository structure

```
├── config.py                       # Central configuration — paths, variables, flags
│
├── Notebooks (numbered pipeline)
│   ├── 1_study_cohort_selection.ipynb
│   ├── 2_helsinki_venex_validation_cohort_selection.ipynb
│   ├── 3_sweden_venex_validation_cohort_selection.ipynb
│   ├── 5_genetic_aberration_based_survival_analysis.ipynb
│   ├── 6_eln_risk_group_generation.ipynb
│   ├── 7_modify_eln_risk_groups.R      # R script called by notebook 6 (run manually between 6_generation and 6_analysis)
│   ├── 6_eln_risk_group_analysis.ipynb
│   ├── 8_vaf_analysis.ipynb
│   ├── 10_FAB_analysis.ipynb
│   └── 11_cytomorphology_analysis.ipynb
│
├── core/                           # Shared Python utilities
│   ├── utils.py                    # load_predictions, add_confusion_categories,
│   │                               # load_or_cache, save_and_show, ...
│   ├── dummy.py                    # generate_dummy_df, save_with_dummy
│   ├── attention.py                # Attention score loading and plotting
│   ├── visualization/              # Kaplan-Meier, box plots, scatter plots
│   ├── statistics/                 # Wilcoxon, statistical analysis
│   └── cell_feature_aggregation/   # Cell-level feature aggregation constants
│
├── scripts/
│   ├── generate_dummy_data.py      # CLI: generate _DUMMY file from any data file
│   ├── zip_project.py              # CLI: export project zip (excludes real patient data)
│   └── export_predictions.py       # (aml-mil repo) Zip predictions for transfer
│
├── 4_import_predictions.py         # Unzip model predictions into data/{V}/model_predictions/
│
├── data/
│   └── {DATA_VERSION_IDENTIFIER}/  # Date-stamped folder (set in config.py)
│       ├── unprocessed_clinical/   # Raw clinical data dumps (source for ELN analysis)
│       ├── mutation_labels/        # Anonymized mutation label tables (study + validation cohorts)
│       ├── survival/               # Survival outcome data
│       ├── vaf/                    # Variant allele frequency data
│       ├── eln/                    # ELN pipeline folders (r_input/, r_output/)
│       └── model_predictions/      # Model run outputs (imported via 4_import_predictions.py)
│           └── AML_V15_OPTIMIZED_PARAMETERS/multilabel/scemila/
│               ├── BM/All cells/run_.../   # predictions.csv, metrics, fold_*/cell_attention_scores.csv
│               ├── PB/All cells/run_.../
│               └── FUSION/All cells/run_.../
│
├── pyproject.toml                  # Python dependencies (uv)
└── environment.yml                 # Conda environment (data environment)
```

---

## Notebook pipeline

| Notebook | Description |
|---|---|
| `1_study_cohort_selection` | Reads raw clinical/mutation data from the academic data lake. Selects diagnostic BM+PB pairs, derives treatment groups, outputs survival/VAF/clinical files to `data/{DATA_VERSION_IDENTIFIER}/`. |
| `2_helsinki_venex_validation_...` | Selects the Helsinki venetoclax validation cohort. |
| `3_sweden_venex_validation_...` | Selects the Sweden venetoclax validation cohort. |
| `5_genetic_aberration_based_survival_analysis` | Kaplan-Meier survival curves stratified by mutation status and model prediction accuracy (TP/FP/FN/TN). |
| `6_eln_risk_group_generation` | Merges model predictions with clinical data and writes input CSVs to `r_input/` for each label-replacement mode. Also writes `eln_run_config.R` at the project root. |
| `7_modify_eln_risk_groups.R` | **Run manually** after `6_generation`. Applies ELN 2022 classification logic and writes results to `r_output/`. |
| `6_eln_risk_group_analysis` | Reads R output from `r_output/` and produces Kaplan-Meier curves, Sankey patient-flow diagrams, and VAF plots. |
| `8_vaf_analysis` | Compares variant allele frequencies between true-positive and false-negative predictions. |
| `10_FAB_analysis` | Associates FAB morphotype classes with mutation predictions. |
| `11_cytomorphology_analysis` | Full cytomorphology analysis: cell differentials, morphometry, attention scores, UMAP, oncoprint. |

---

## Setup

### Local environment (no real patient data)

Requires [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run jupyter lab
```

Set `USE_DUMMY_DATA = True` in `config.py` (default) to use synthetic data.

### Data environment (real patient data, conda)

```bash
conda env create -f environment.yml
conda activate aml-mut-pred
```

Update `environment.yml`: 

```bash
conda env update -f environment.yml --prune
```

Set `USE_DUMMY_DATA = False` in `config.py` before running notebooks.

---

## Configuration (`config.py`)

All shared settings live in `config.py`. Notebooks import from it:

```python
from config import USE_DUMMY_DATA, BM_RUN_PATH, SELECTED_VARIABLES, SURVIVAL_DATA_PATH_BM
```

Key settings:

| Variable | Description |
|---|---|
| `USE_DUMMY_DATA` | `True` → all data paths resolve to `_DUMMY` companion files |
| `DATA_VERSION_IDENTIFIER` | Date string used in folder names and filenames (e.g. `"03-06-2026"`); change to save a new data version |
| `FEATURE_VERSION` | Version of the image-derived features (e.g. `"07-04-2026"`): the cohort/label selection hematoscope-research generated them from. Separate from `DATA_VERSION_IDENTIFIER` |
| `SHARED_DATA_DIR`, `TRAINING_FEATURES_DIR`, `VENEX_FEATURES_DIR`, `SWEDEN_FEATURES_DIR`, `HEHR_FEATURES_DIR` | Shared feature folders in `2-mut-pred/data/` (override with the `MUT_PRED_DATA_DIR` environment variable) |
| `SAVE_FILES` | `True` → notebook 1 writes output files; skips if file already exists |
| `BM_RUN_PATH` / `PB_RUN_PATH` / `FUSION_RUN_PATH` | Paths to model prediction run directories under `data/{V}/model_predictions/` |
| `SELECTED_VARIABLES` | Canonical list of 17 mutation/cytogenetic targets |
| `SURVIVAL_DATA_PATH_BM/PB` | Survival outcome data |
| `VAF_DATA_PATH_BM/PB` | Variant allele frequency data |
| `CLINICAL_DATA_PATH_BM/PB` | Raw clinical data in `unprocessed_clinical/` (used by ELN analysis) |
| `MUTATION_LABELS_PATH_BM/PB` | Anonymized mutation label tables in `mutation_labels/` |
| `BM_COLOR`, `PB_COLOR`, `FUSION_COLOR` | Modality colours for plots |

---

## Data folder

The data folder is named after `DATA_VERSION_IDENTIFIER` set in `config.py` (e.g. `data/03-06-2026/`). To create a new data version, update the identifier and re-run notebook 1 — existing files are never overwritten.

```
data/{DATA_VERSION_IDENTIFIER}/
├── unprocessed_clinical/   # Raw clinical dumps (all variables; source for ELN analysis)
├── mutation_labels/        # Anonymized label tables (study cohort + validation cohorts)
├── survival/               # Survival outcomes (time to death, treatment groups)
├── vaf/                    # Variant allele frequency values per gene
├── eln/                    # ELN pipeline input/output (r_input/, r_output/)
└── model_predictions/      # Model run outputs (imported via 4_import_predictions.py)
```

### Shared feature data (`2-mut-pred/data/`)

The image-derived files are no longer copied into this repository. They live in the project's shared data folder next to the repositories, where hematoscope-research writes them (`scripts/mut_pred/`) and aml-mil also reads them:

```
2-mut-pred/
├── aml-mil/
├── mut-pred-analysis/       this repository
└── data/
    ├── aml_dg_de-novo-and-secondary-{FEATURE_VERSION}/   study cohort: H5, labels, label map, aggregated CSVs, cell-level parquets
    ├── venex_bm-{FEATURE_VERSION}/                         Helsinki venetoclax cohort: H5, sample sheet, aggregated CSV
    ├── sweden_bm-{FEATURE_VERSION}/                        Sweden cohort: H5, sample sheet, aggregated CSV
    └── hehr_aml/                                           Hehr AML PB: H5, labels, aggregated CSV
```

Each data subfolder of `data/{DATA_VERSION_IDENTIFIER}/` contains:
- **Real files** — patient data, present only in the data environment
- **`_DUMMY` files** — synthetic 30-patient versions committed to the repo

---

## Model predictions workflow

```bash
# In the aml-mil environment — run export_predictions.py (copy it there first):
python export_predictions.py          # zips predictions.csv, metrics, attention scores

# In this repo (place predictions.zip in the project root first):
python 4_import_predictions.py        # extracts zip into data/{V}/model_predictions/ and deletes the zip
```

---

## Generating dummy data

Any DataFrame saved with `save_with_dummy()` automatically gets a `_DUMMY` companion in the same folder.
To generate one from an existing file:

```bash
python scripts/generate_dummy_data.py data/03-06-2026/survival/aml_mut_pred_survival_data_bm_03-06-2026.xlsx
# → writes aml_mut_pred_survival_data_bm_03-06-2026_DUMMY.xlsx in the same folder
```

---

## Exporting the project (without patient data)

```bash
python scripts/zip_project.py                          # → project_zips/aml_mut_pred_analysis_safe.zip
python scripts/zip_project.py --mode code              # → project_zips/aml_mut_pred_analysis_code.zip
python scripts/zip_project.py --mode full              # → project_zips/aml_mut_pred_analysis_full.zip
python scripts/zip_project.py --mode full -n my.zip   # → project_zips/my.zip
```

| Mode | Data files included |
|---|---|
| `safe` (default) | Only `_DUMMY` files — recipient can run with `USE_DUMMY_DATA = True` |
| `code` | No data files — code and environment only |
| `full` | All data files (real patient data + model predictions) — for backup or data-environment transfer |

`data/source_data_from_academic/` and `eln_run_config.R` are always excluded.
