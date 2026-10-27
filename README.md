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
│   ├── dummy.py                    # generate_dummy_df, save_with_dummy, make_toy_eln_data
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
| `8_eln_risk_group_analysis` | Reads R output from `r_output/`. Tabulates the label-replacement modes (C-index with paired bootstrap CIs, agreement with the true-label groups), then Kaplan-Meier curves, Sankey diagrams, driver bar plots and VAF plots for the chosen configuration. `USE_TOY_DATA` tests it without patient data. |
| `9_vaf_analysis` | Exploratory comparison of variant allele frequencies between true-positive and false-negative predictions (the manuscript version is `scripts/extract_vaf_by_outcome.py` + S13 of the V2 notebook). |
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
conda activate aml_mut
```

Update `environment.yml`: 

```bash
conda env update -f environment.yml --prune
```

Set `USE_DUMMY_DATA = False` in `config.py` before running notebooks.

`8_eln_risk_group_analysis.ipynb` can be tested without the patient data: set `USE_TOY_DATA = True` in its config cell. It writes a random cohort to `data/_temp_test/` (`core.dummy.make_toy_eln_data`) and saves the figures to `figures/_temp_test/`, so the real figures are not touched. The results are meaningless; set it back to `False` for real runs.

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

## After a new model run: regenerating the manuscript figures

`11_cytomorphology_analysis_V2.ipynb` only reads derived tables; it computes nothing heavy. After new aml-mil training runs, regenerate the tables below in this order (from the repository root, data environment). All outputs are git-ignored and go to `data/{DATA_VERSION_IDENTIFIER}/derived/`.

1. **Model outputs from aml-mil:** run its steps after training (see [aml-mil README](../aml-mil/README.md#after-training-completing-the-manuscript-figures)), then `export_predictions.py` there, put `predictions.zip` in the project root and run `python 4_import_predictions.py`. Set the aml-mil run name in `manuscript_figures/config.py`. The notebook shows Fig 2, Fig 3 and the performance supplements from `aml-mil/results/<RUN>/figures`.
2. **Extraction scripts**:

| Order | Script | Writes (`derived/…`) | Used by |
|---|---|---|---|
| 1 | `scripts/extract_fig4_fingerprint_results.py` | `fig4_fingerprints/` (feature effects, fingerprint tags, attention enrichment, per-sample features) | Fig 4/5, variable cards, gallery |
| 2 | `scripts/extract_error_attention_cache.py` (minutes) | `error_analysis/sample_attention_summary.parquet` | error-analysis supplements |
| 3 | `scripts/extract_error_outcome_stats.py` | `error_analysis/morph_outcome_stats.csv` (needs 2) | error-analysis supplements |
| 4 | `scripts/extract_morphometry_dysplasia_correlation.py` | `morph_dysplasia/` | morphometry of dysplastic cells (supplement) |
| 5 | `scripts/extract_gallery_cells.py` (~3 min; `--skip-top` reuses the top cells) | `gallery/`: top cells of all card targets, clusters + 2D embedding, Fig 6 tiles; also the cell lists for step 6 | Fig 6, variable cards (S7), TP/FP pages (S8), embedding scatter (S6) |
| 6 | `scripts/extract_cell_crops.py data/<version>/derived/cell_crops/lists/*.csv` | `cell_crops/` (cached cell images) | everything that shows cell images |
| 7 | `scripts/extract_vaf_by_outcome.py` (needs the VAF tables of `config.py`) | `error_analysis/vaf_by_outcome.csv`, `vaf_outcome_stats.csv` | VAF of TP vs FN (S13) |
| 8 | `scripts/extract_fab_analysis.py` (needs `FAB_DATA_PATH`: subject_id, fab) | `fab/`: composition, coverage, prevalence, performance and error rates per FAB group | FAB supplements (S14–S16) |
| 9 | `scripts/extract_fab_features.py` (needs 1 and `FAB_DATA_PATH`) | `fab/fab_feature_effects.csv`: fingerprint per FAB group, interaction and FAB-stratified tests, only for `FAB_TARGETS` of `extract_fab_analysis.py` (PML-RARA, NPM1) | FAB supplements (S16) |
| 10 | `scripts/extract_cooccurrence.py` | `cooccurrence/`: cell counts and φ per pair of targets, model coupling | co-occurrence overview (S17a) |
| 11 | `scripts/extract_cooccurrence_features.py` (needs 1) | `cooccurrence/cooc_feature_effects.csv`, `cooc_summary.csv`: fingerprints adjusted for a co-occurring target | S17b–c |

3. **Cell images (step 6)** need the image server on `localhost:5432` and the NAS. The figure code writes the cell lists, so the first time: run the notebook once, fetch the crops, run the notebook again. Cells without an image are drawn as stand-ins. Cells whose window leaves the field and samples with unreadable images are skipped, so the selection is stable; failed fetches are listed in `cell_crops/missing.csv`.
4. **Run the notebook** top to bottom. Sections with missing inputs print a message instead of a figure.

> **Pixel size:** BM = 1 / 9.034925854 µm/px (`UM_PER_PX` in `manuscript_figures/config.py`; area × MPP² = µm²). PB uses the same value but this is not yet confirmed (`UM_PER_PX_UNCONFIRMED`); once it is, update `UM_PER_PX["PB"]`, empty `UM_PER_PX_UNCONFIRMED` and rerun the notebook.

---

## Analyses waiting for the real data

These are built and tested on synthetic data only (the patient-level files exist only in the academic environment). Nothing below has produced a real number yet. Each analysis needs its input, the commands, and a re-run of the notebook named.

### 1. ELN risk groups per label-replacement mode (`8_eln_risk_group_analysis.ipynb`)

Question: how much of the ELN stratification do the model's labels preserve (reference = true labels, TARGETS)? Not "better than ELN".
- **Inputs (academic environment):** the R output for the modes `TARGETS`, `TP`, `TP_FP`, `FN` (`6_eln_risk_group_generation.ipynb` → `7_modify_eln_risk_groups.R`; `TP_FP_FN` is no longer generated, it equals `TP_FP`); the survival tables. An `age` column in the survival tables (optional) enables the Cox models.
- **Run:** set `USE_TOY_DATA = False`; run the notebook top to bottom. The section *Label-replacement modes compared* writes `data/<version>/eln/mode_comparison.csv` and shows the table (C-index of TARGETS and the mode with 95% bias-corrected CIs, ΔC from a paired patient bootstrap, share in the same group, weighted kappa, patients moving up or down, patients lost to ELN1 = NA).
- **Follow-up:** there are no follow-up dates, so alive patients are censored at the longest observed time, which inflates follow-up for recent patients. Every row is therefore also computed with follow-up cut at 36 months (`HORIZONS`); treat the cut version as primary. Report how censoring was set in the Methods.
- **Choosing the configuration:** the best mode and modality are picked after seeing the table, so say so. To pre-specify, edit `PRIMARY` (default FUSION, `selected_variables`, `TP_FP`, 36 months) *before* looking. `PRIMARY` also drives the survival-at-12/24-months table and the age-adjusted Cox models.
- **Then** use the Kaplan-Meier, Sankey and driver bar plots further down the notebook for the chosen configuration only. Code: `core/eln_analysis.py`.
- **Fig 7 (clinical impact)** is not designed yet; this table is its likely basis.

### 2. VAF of detected and missed positives (S13)

- **Inputs:** the VAF tables of `config.py` (`VAF_DATA_PATH_BM`, `VAF_DATA_PATH_PB`; columns `album_id`, `database_id` and one VAF column per target). Targets without a VAF column are skipped.
- **Run (where the VAF tables are):** `.venv/bin/python scripts/extract_vaf_by_outcome.py` → `derived/error_analysis/vaf_by_outcome.csv`, `vaf_outcome_stats.csv` (no patient identifiers). Copy them here if run elsewhere, then re-run the V2 notebook (S13a–c: BM, PB, FUSION).
- Only mutated patients have a VAF, so the contrast is FN against TP. P values are BH-FDR adjusted across the targets of the model.

### 3. FAB classes (S14–S16)

- **Input:** `data/<version>/fab/aml_mut_pred_fab_<version>.csv` (`FAB_DATA_PATH`) with the columns `subject_id` (the anonymised integer of `predictions.csv`, not `henkilotunnus`) and `fab` (`M0` … `M7`), one row per patient.
- **Run:** `.venv/bin/python scripts/extract_fab_analysis.py`, then `.venv/bin/python scripts/extract_fab_features.py`; re-run the V2 notebook.
- **Groups:** M0–M2, M3, M4–M5 (counts in the cohort: M2 124, M5 89, M1 88, M3 24, M4 11, M0 9; no M6/M7, M4eo not coded). M3 is a control only. Targets of the per-target supplements: `FAB_TARGETS` in `scripts/extract_fab_analysis.py` (PML-RARA → S15, NPM1 → S16); add a target there and a notebook section to extend.
- **Check first (S14):** the FAB × target prevalence heatmap should show PML-RARA in M3, and `fab_coverage.csv` shows whether patients with a FAB class differ from those without (39% have none).
- **Pre-specified hypothesis:** NPM1-positive blasts in M0–M2 (mostly M1) have more indented nuclei than in M4–M5 (Rastogi et al.; cup-shaped nuclei). Proxy features: solidity, compactness and eccentricity of blast nuclei (flagged ◆ in S16). Everything else is exploratory; interaction FDR is across the features of one target.
- **Not covered by the data:** M6/M7 (TP53, complex karyotype) cannot be tested; M4 alone (n = 11) is too small.
- Write the results as a short paragraph; figures stay in the supplement.

### 4. After the MANUSCRIPT_V1 run

The attention files of the stand-in run (`AML_V15_OPTIMIZED_PARAMETERS`) lost attention scores for arbitrary labelled bags (masking bug fixed in aml-mil `e97172f`). With the new run: check that no labelled bag has an empty `cell_attention_score_<target>` (CBFB-MYH11 should have 12 true positives in BM, not 8), then re-run all `scripts/extract_*.py` in the order of the table above and the V2 notebook.

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
