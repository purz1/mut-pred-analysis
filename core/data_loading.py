"""Shared loading of the discovery cohort (labels + sample-level morphology features).

Mirrors the "Load data" section of 11_cytomorphology_analysis.ipynb so that notebooks and
scripts use one implementation. Paths are relative to the repository root.
"""
import json
from pathlib import Path

import h5py
import pandas as pd

from config import DATA_VERSION_IDENTIFIER, BM_RUN_PATH, PB_RUN_PATH, FUSION_RUN_PATH, TRAINING_FEATURES_DIR

REPO = Path(__file__).resolve().parents[1]
FEAT_DIR = TRAINING_FEATURES_DIR
DERIVED_DIR = REPO / f"data/{DATA_VERSION_IDENTIFIER}/derived"
RUN_PATHS = {"BM": BM_RUN_PATH, "PB": PB_RUN_PATH, "FUSION": FUSION_RUN_PATH}

SELECTED_VARIABLES = [
    "PML-RARA", "CBFB-MYH11", "RUNX1-RUNX1T1_OR_t_8_21", "IDH_mutations", "IDH2", "CpG_mutations",
    "NPM1", "NPM1_AND_FLT3_ITD", "FLT3_mutations", "FLT3_ITD", "KIT", "RTK_RAS_mutations",
    "complex_karyotype", "monosomal_karyotype", "monosomy5_del_5q", "TP53", "STAG2",
]

FILES = {
    "BM": dict(labels="labels_aml_dg_de-novo-and-secondary_bm_07-04-2026.xlsx",
               features="aggregated_features_aml_dg_de-novo-and-secondary_bm_07-04-2026.csv",
               h5="features_no-quality-filter_aml_dg_de-novo-and-secondary_bm_07-04-2026.h5",
               parquet="cell_level_features_aml_dg_de-novo-and-secondary_bm_07-04-2026.parquet",
               cup="cup_blasts_features_no-quality-filter_aml_dg_de-novo-and-secondary_bm_07-04-2026.csv",
               cup_cells="cup_blasts_cells_no-quality-filter_aml_dg_de-novo-and-secondary_bm_07-04-2026.parquet",
               run=BM_RUN_PATH),
    "PB": dict(labels="labels_aml_dg_de-novo-and-secondary_pb_07-04-2026.xlsx",
               features="aggregated_features_aml_dg_de-novo-and-secondary_pb_07-04-2026.csv",
               h5="features_no-quality-filter_aml_dg_de-novo-and-secondary_pb_07-04-2026.h5",
               parquet="cell_level_features_aml_dg_de-novo-and-secondary_pb_07-04-2026.parquet",
               cup="cup_blasts_features_no-quality-filter_aml_dg_de-novo-and-secondary_pb_07-04-2026.csv",
               cup_cells="cup_blasts_cells_no-quality-filter_aml_dg_de-novo-and-secondary_pb_07-04-2026.parquet",
               run=PB_RUN_PATH),
}
LABEL_MAP_FILE = "label-map_aml_dg_de-novo-and-secondary_07-04-2026.json"
CUP_THRESHOLD = 0.44  # as in hematoscope-research v3_score_cup_blasts.py

# ── feature lists (as in the notebook constants) ────────────────────────────
CELL_DIFF_BM = [
    "Blasts_proportion", "Proerythroblasts_proportion", "Erythroblasts_proportion", "Promyelocytes_proportion",
    "Myelocytes_proportion", "Metamyelocytes_proportion", "Eosinophils_immature_proportion", "Eosinophils_proportion",
    "Basophils_proportion", "Neutrophils_proportion", "Promonocytes_proportion", "Monocytes_proportion",
    "Lymphocytes_proportion", "Plasma_cells_proportion",
]  # Megakaryocytes_proportion removed, as in get_bm_cell_differential_variables()
CELL_DIFF_PB = [
    "Blasts_proportion", "Erythroblasts_proportion", "Myelocytes_proportion", "Metamyelocytes_proportion",
    "Basophils_proportion", "Band_neutrophils_proportion", "Segmented_neutrophils_proportion",
    "Eosinophils_proportion", "Monocytes_proportion", "Lymphocytes_proportion", "Plasma_cells_proportion",
]
DYSPLASIA_BM = [
    "Myeloid_precursors_hypergranulated_proportion", "Myeloid_precursors_bilobed_proportion",
    "Myeloid_precursors_auer_proportion", "Erythroblasts_asynchrony_proportion", "Erythroblasts_dysmorphic_proportion",
    "Erythroblasts_multinucleated_proportion", "Megakaryocytes_multinucleated_proportion",
    "Megakaryocytes_hypolobated_proportion", "Megakaryocytes-vacuolated_proportion", "Blasts-vacuolated_proportion",
    "Erythroblasts-vacuolated_proportion", "Promyelocytes-vacuolated_proportion", "Myelocytes-vacuolated_proportion",
    "Basophils-vacuolated_proportion", "Neutrophils-vacuolated_proportion", "Eosinophils-vacuolated_proportion",
    "Lymphocytes-vacuolated_proportion", "Plasma_cells-vacuolated_proportion", "Blasts-cup_proportion",
]
DYSPLASIA_PB = [
    "APL-apl_hypergranulated_proportion", "Blasts-apl_hypergranulated_proportion",
    "Myelocytes-apl_hypergranulated_proportion", "Neutrophils-neut_gran_normal_proportion",
    "Neutrophils-neut_gran_hypo_proportion", "Neutrophils-neut_gran_hyper_proportion", "Blasts-cup_proportion",
]
CELL_STAT_REGEX = ("entire_area_median|N:C-ratio_median|entire_compactness_median|nucleus_area_median"
                   "|nucleus_solidity_median|nucleus_eccentricity_median|nucleus_compactness_median")
CELL_STAT_EXCLUDE = "Lymphocytes|Macrophages|Plasma_cells"
STAT_SUFFIXES = ["entire_area_median", "nucleus_area_median", "entire_N:C-ratio_median", "nucleus_solidity_median",
                 "nucleus_eccentricity_median", "nucleus_compactness_median", "entire_compactness_median"]
FAMILY = {"entire_area_median": "size", "nucleus_area_median": "size", "entire_N:C-ratio_median": "N:C",
          "nucleus_solidity_median": "nuclear shape", "nucleus_eccentricity_median": "nuclear shape",
          "nucleus_compactness_median": "nuclear shape", "entire_compactness_median": "cell shape"}


def get_target_labels(df, label_map, include_variables):
    label_map = {v: k for k, v in label_map.items()}
    df = df.rename(columns=label_map)
    df = df[["subject_id", "album_id", "database_id"] + include_variables]
    df.album_id = df.album_id.astype(str)
    return df


def rename_dysplasia_variables(df):
    df = df.rename(columns={"APL-overall_dysplasia_proportion": "Myeloid_precursors_overall_dysplasia_proportion"})
    df = df.rename(columns=lambda c: c.replace("-apl", ""))
    df = df.rename(columns=lambda c: c.replace("APL-", "Myeloid_precursors"))
    df = df.rename(columns=lambda c: c.replace("APL", "Myeloid_precursors"))
    df = df.rename(columns=lambda c: c.replace("-Sufficient-mgk", ""))
    df = df.rename(columns=lambda c: c.replace("-mds", ""))
    return df


def cell_statistic_columns(df):
    cols = df.filter(regex=CELL_STAT_REGEX).columns
    return [c for c in cols if not pd.Series([c]).str.contains(CELL_STAT_EXCLUDE).iloc[0]]


def preprocess(df, modality):
    """Aggregated features → notebook conventions (percentages, renamed dysplasia, no Living_cells- prefix)."""
    df = df.copy()
    if modality == "BM":
        df = df.rename(columns={"year": "database_id"})
    df.album_id = df.album_id.astype(str)
    df["database_id"] = df["database_id"].astype(str)
    df.loc[:, df.columns.str.contains("proportion")] *= 100
    if modality == "BM":
        df = rename_dysplasia_variables(df)
    df.columns = df.columns.str.replace(r"^Living_cells-", "", regex=True)
    return df


def load_obs_from_h5(path):
    """Samples (album_id, database_id) present in the modelling h5 file."""
    rows = []
    with h5py.File(path, "r") as f:
        for database_id in f:
            for album_id in f[database_id]:
                rows.append((album_id, database_id))
    return pd.DataFrame(rows, columns=["album_id", "database_id"]).drop_duplicates()


def load_labels(modality, variables=SELECTED_VARIABLES):
    labels = pd.read_excel(FEAT_DIR / FILES[modality]["labels"])
    labels = get_target_labels(labels, json.load(open(FEAT_DIR / LABEL_MAP_FILE)), variables)
    labels["database_id"] = labels["database_id"].astype(str)
    return labels


def load_cup_features(modality):
    """Blasts-cup_proportion per sample (hematoscope-research v3_score_cup_blasts.py, threshold 0.44)."""
    cup = pd.read_csv(FEAT_DIR / FILES[modality]["cup"], dtype={"database_id": str, "album_id": str})
    return cup[["database_id", "album_id", "Blasts-cup_proportion"]]


def add_cup_flag(cell_df, modality):
    """Cell-level parquet + `cup` (blast scored > CUP_THRESHOLD); the low_quality gate is applied in the aggregation."""
    cup = pd.read_parquet(FEAT_DIR / FILES[modality]["cup_cells"], columns=["database_id", "album_id", "cell_id", "cup_blasts_prob"])
    cell_df = cell_df.merge(cup, on=["database_id", "album_id", "cell_id"], how="left")
    cell_df["cup"] = cell_df.pop("cup_blasts_prob") > CUP_THRESHOLD
    return cell_df


def load_cohort(modality, variables=SELECTED_VARIABLES):
    """Labels + sample-level features for samples used in modelling (= notebook df_bm / df_pb)."""
    fl = FILES[modality]
    labels = load_labels(modality, variables)
    preds = pd.read_csv(REPO / fl["run"] / "predictions.csv", usecols=["album_id", "database_id"])
    features = pd.read_csv(FEAT_DIR / fl["features"], dtype={"database_id": str, "album_id": str})
    features = preprocess(features.merge(load_cup_features(modality), how="left"), modality)
    in_model = load_obs_from_h5(FEAT_DIR / fl["h5"])
    for d in (labels, preds, features, in_model):
        d["database_id"] = d["database_id"].astype(str)
        d["album_id"] = d["album_id"].astype(str)
    df = labels.merge(preds.drop_duplicates(), on=["album_id", "database_id"])
    df = df.merge(features, on=["album_id", "database_id"])
    df = in_model.merge(df, on=["album_id", "database_id"])
    return df


def feature_lists(df, modality):
    """Feature columns per type: cell_differential, dysplasia, cell_statistic."""
    diff = CELL_DIFF_BM if modality == "BM" else CELL_DIFF_PB
    dys = DYSPLASIA_BM if modality == "BM" else DYSPLASIA_PB
    return {"cell_differential": [c for c in diff if c in df.columns],
            "dysplasia": [c for c in dys if c in df.columns],
            "cell_statistic": cell_statistic_columns(df)}
