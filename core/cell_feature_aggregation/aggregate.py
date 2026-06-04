import pandas as pd
import numpy as np
from pandas import DataFrame, Series

from core.cell_feature_aggregation.constants.bm import (
    CELL_LABEL_MAP as BM_CELL_LABEL_MAP,
    CELL_LINEAGES as BM_CELL_LINEAGES,
    CELL_STATISTICS_CLASSES as BM_CELL_STATISTICS_CLASSES,
    APL_INCLUDED_CLASSES as BM_APL_INCLUDED_CLASSES,
    APL_RESULTS as BM_APL_RESULTS,
    MDS_INCLUDED_CLASSES as BM_MDS_INCLUDED_CLASSES,
    MDS_RESULTS as BM_MDS_RESULTS,
    MYELOID_DYSPLASIA,
    ERYTHROID_DYSPLASIA,
)
from core.cell_feature_aggregation.constants.pb import (
    CELL_LABEL_MAP as PB_CELL_LABEL_MAP,
    CELL_LINEAGES as PB_CELL_LINEAGES,
    CELL_STATISTICS_CLASSES as PB_CELL_STATISTICS_CLASSES,
    APL_INCLUDED_CLASSES as PB_APL_INCLUDED_CLASSES,
    APL_RESULTS as PB_APL_RESULTS,
    MDS_INCLUDED_CLASSES as PB_MDS_INCLUDED_CLASSES,
    MDS_RESULTS as PB_MDS_RESULTS,
    MACROP_INCLUDED_CLASSES as PB_MACROP_INCLUDED_CLASSES,
    MACROP_RESULTS as PB_MACROP_RESULTS,
    NEUT_INCLUDED_CLASSES as PB_NEUT_INCLUDED_CLASSES,
    NEUT_RESULTS as PB_NEUT_RESULTS,
)

SHAPE_FEATURES = [
    "nucleus_area",
    "nucleus_solidity",
    "nucleus_eccentricity",
    "nucleus_compactness",
    "entire_area",
    "entire_solidity",
    "entire_eccentricity",
    "entire_compactness",
    "entire_N:C-ratio",
]


def _min_division(numerator: int, denominator: int, minimum: int) -> float | None:
    if denominator >= minimum:
        return numerator / denominator
    return None


def _to_str_cts(cts: int | list, cell_label_map: dict) -> str | list[str]:
    """Convert integer class ID(s) to string label(s) using the label map."""
    if isinstance(cts, list):
        return [cell_label_map[ct] for ct in cts]
    return cell_label_map[cts]


def _cls_mask(cell_df: DataFrame, ct_str: str | list[str]):
    if isinstance(ct_str, list):
        return cell_df["class_label"].isin(ct_str)
    return cell_df["class_label"] == ct_str


def _aggregate_shape_stats(
    cell_df: DataFrame,
    cell_statistics_classes: list[int],
    cell_label_map: dict,
    minimum: int,
    minimum_megas: int | None = None,
) -> Series:
    """Compute per-cell-type median and 95% range for each shape feature."""
    result = {}
    for ct_int in cell_statistics_classes:
        ct_name = cell_label_map[ct_int]
        ct_df = cell_df.loc[cell_df["class_label"] == ct_name, SHAPE_FEATURES]

        min_count = minimum_megas if (minimum_megas is not None and ct_name == "Megakaryocytes") else minimum

        median_keys = [f"{ct_name}_{feat}_median" for feat in SHAPE_FEATURES]
        range_keys = [f"{ct_name}_{feat}_95_range" for feat in SHAPE_FEATURES]

        if ct_df.shape[0] >= min_count:
            vals = ct_df.values.astype(float)
            medians = np.nanmedian(vals, axis=0)
            p975 = np.nanpercentile(vals, 97.5, axis=0)
            p025 = np.nanpercentile(vals, 2.5, axis=0)
            for key, val in zip(median_keys, medians):
                result[key] = val
            for key, val in zip(range_keys, p975 - p025):
                result[key] = val
        else:
            for key in median_keys + range_keys:
                result[key] = np.nan

    return pd.Series(result)


def aggregate_bm_sample(
    cell_df: DataFrame,
    filter_type: str = "cell_differential",
    minimum: int = 20,
    minimum_megas: int = 10,
) -> Series:
    """
    Aggregate a cell-level BM DataFrame (one sample from the parquet) into
    sample-level features matching get_aggregated_data(..., data_source="BM", robust=True).

    Parameters
    ----------
    cell_df : DataFrame
        Rows are cells for a single BM sample from the cell-level parquet.
    filter_type : str
        Which feature group to compute and which cell filter to apply:
        - "cell_differential": no extra filter; returns differential counts,
          lineage proportions, mitotic proportions, vacuolated proportions.
        - "dysplasia": low_quality < 0.5 filter for APL/MDS/overall dysplasia;
          quality_model_filtered=False for MGK dysplasia (Sufficient megas only).
        - "cell_statistics": quality_model_filtered=False filter; returns shape
          feature statistics (median and 95% range per cell type).
        - "all": concatenates results from all three filter types.
    minimum : int
        Minimum cell count required for any proportion or statistic to be computed.
    minimum_megas : int
        Same as minimum but applied to Megakaryocytes shape statistics and MGK
        dysplasia proportions.

    Returns
    -------
    Series
        Feature values for the requested filter_type.
    """
    result = {}

    if filter_type == "cell_statistics":
        stats_df = cell_df[~cell_df["quality_model_filtered"]]
        shape_ser = _aggregate_shape_stats(
            stats_df, BM_CELL_STATISTICS_CLASSES, BM_CELL_LABEL_MAP, minimum, minimum_megas
        )
        result.update(shape_ser.to_dict())

    elif filter_type == "cell_differential":
        living_names = [BM_CELL_LABEL_MAP[ct] for ct in BM_CELL_LINEAGES["Living_cells"]]
        n_living = int(cell_df["class_label"].isin(living_names).sum())

        cell_counts: dict[str, int] = {}
        for ct in BM_CELL_LINEAGES["Living_cells"]:
            ct_name = BM_CELL_LABEL_MAP[ct]
            cell_counts[ct_name] = int((cell_df["class_label"] == ct_name).sum())

        lineage_counts: dict[str, int] = {}
        for lineage, cts in BM_CELL_LINEAGES.items():
            ct_names = [BM_CELL_LABEL_MAP[ct] for ct in cts]
            lineage_counts[lineage] = int(cell_df["class_label"].isin(ct_names).sum())

        for ct in BM_CELL_LINEAGES["Living_cells"]:
            ct_name = BM_CELL_LABEL_MAP[ct]
            result[f"Living_cells-{ct_name}_proportion"] = _min_division(cell_counts[ct_name], n_living, minimum)

        for lineage in BM_CELL_LINEAGES.keys():
            result[f"Living_cells-{lineage}_proportion"] = _min_division(lineage_counts[lineage], n_living, minimum)

        for ct in BM_CELL_LINEAGES["Living_cells"]:
            ct_name = BM_CELL_LABEL_MAP[ct]
            n_ct = cell_counts[ct_name]
            n_mitotic = int(((cell_df["class_label"] == ct_name) & cell_df["mitotic"]).sum())
            result[f"{ct_name}-mitotic_proportion"] = _min_division(n_mitotic, n_ct, minimum)

        for ct in BM_CELL_LINEAGES["Living_cells"]:
            ct_name = BM_CELL_LABEL_MAP[ct]
            n_ct = cell_counts[ct_name]
            n_vacuolated = int(((cell_df["class_label"] == ct_name) & cell_df["vacuolated"]).sum())
            result[f"{ct_name}-vacuolated_proportion"] = _min_division(n_vacuolated, n_ct, minimum)

    elif filter_type == "dysplasia":
        lq = cell_df["low_quality"] < 0.5

        # APL dysplasia — denominator: all cells of class; numerator: lq-filtered
        for cls_name, cts in BM_APL_INCLUDED_CLASSES.items():
            ct_str = _to_str_cts(cts, BM_CELL_LABEL_MAP)
            mask = _cls_mask(cell_df, ct_str)
            n_cls = int(mask.sum())
            for res in BM_APL_RESULTS:
                n_res = int((mask & lq & cell_df[res]).sum())
                result[f"{cls_name}-{res}_proportion"] = _min_division(n_res, n_cls, minimum)

        # MDS dysplasia — denominator: all cells of class; numerator: lq & ~mitotic filtered
        for cls_name, cts in BM_MDS_INCLUDED_CLASSES.items():
            ct_str = _to_str_cts(cts, BM_CELL_LABEL_MAP)
            mask = _cls_mask(cell_df, ct_str)
            lq_non_mitotic_mask = mask & lq & ~cell_df["mitotic"]
            n_cls = int(mask.sum())
            for res in BM_MDS_RESULTS:
                n_res = int((lq_non_mitotic_mask & cell_df[res]).sum())
                result[f"{cls_name}-{res}_proportion"] = _min_division(n_res, n_cls, minimum)

        # MGK dysplasia — Sufficient megas only (quality_model_filtered=False captures
        # both Insufficient-size and ONNX-failing megas)
        mega_mask = cell_df["class_label"] == "Megakaryocytes"
        sufficient_mega_mask = mega_mask & ~cell_df["quality_model_filtered"]
        n_mega = int(sufficient_mega_mask.sum())
        for mgk_res in ["mgk_hypolobated", "mgk_multinucleated"]:
            n_res = int((sufficient_mega_mask & cell_df[mgk_res]).sum())
            result[f"Megakaryocytes-Sufficient-{mgk_res}_proportion"] = _min_division(n_res, n_mega, minimum_megas)

        # Myeloid overall dysplasia — both numerator and denominator lq-filtered
        n_apl_dysplasia = 0
        n_apl_total = 0
        for ct_int, dysplasia_rules in MYELOID_DYSPLASIA.items():
            ct_name = BM_CELL_LABEL_MAP[ct_int]
            mask = cell_df["class_label"] == ct_name
            lq_mask = mask & lq
            n_ct = int(lq_mask.sum())
            is_dysplastic = cell_df[dysplasia_rules[0][0]]
            for (res, _) in dysplasia_rules[1:]:
                is_dysplastic = is_dysplastic | cell_df[res]
            n_dysplastic = int((lq_mask & is_dysplastic).sum())
            result[f"{ct_name}-overall_dysplasia_proportion"] = _min_division(n_dysplastic, n_ct, minimum)
            n_apl_dysplasia += n_dysplastic
            n_apl_total += n_ct
        result["APL-overall_dysplasia_proportion"] = _min_division(n_apl_dysplasia, n_apl_total, minimum)

        # Erythroid overall dysplasia — both numerator and denominator lq & ~mitotic filtered
        erythro_mask = cell_df["class_label"] == "Erythroblasts"
        lq_non_mitotic_erythro = erythro_mask & lq & ~cell_df["mitotic"]
        n_denom = int(lq_non_mitotic_erythro.sum())
        erythro_rules = ERYTHROID_DYSPLASIA[5]
        is_dysplastic = cell_df[erythro_rules[0][0]]
        for (res, _) in erythro_rules[1:]:
            is_dysplastic = is_dysplastic | cell_df[res]
        n_dysplastic = int((lq_non_mitotic_erythro & is_dysplastic).sum())
        result["Erythroblasts-overall_dysplasia_proportion"] = _min_division(n_dysplastic, n_denom, minimum)

    elif filter_type == "all":
        parts = [
            aggregate_bm_sample(cell_df, "cell_differential", minimum, minimum_megas),
            aggregate_bm_sample(cell_df, "dysplasia", minimum, minimum_megas),
            aggregate_bm_sample(cell_df, "cell_statistics", minimum, minimum_megas),
        ]
        return pd.concat(parts)

    else:
        raise ValueError(f"Unknown filter_type '{filter_type}'. Use 'cell_differential', 'dysplasia', 'cell_statistics', or 'all'.")

    return pd.Series(result)


def aggregate_pb_sample(
    cell_df: DataFrame,
    filter_type: str = "cell_differential",
    minimum: int = 10,
) -> Series:
    """
    Aggregate a cell-level PB DataFrame (one sample from the parquet) into
    sample-level features matching get_aggregated_data(..., data_source="PB", robust=True).

    Parameters
    ----------
    cell_df : DataFrame
        Rows are cells for a single PB sample from the cell-level parquet.
    filter_type : str
        Which feature group to compute and which cell filter to apply:
        - "cell_differential": no extra filter; returns differential counts,
          lineage proportions, mitotic proportions, vacuolated proportions,
          and neutrophil granulation/segmentation proportions.
        - "dysplasia": low_quality < 0.5 filter for APL/MDS/macropolycyte dysplasia.
        - "cell_statistics": quality_model_filtered=False filter; returns shape
          feature statistics.
        - "all": concatenates results from all three filter types.
    minimum : int
        Minimum cell count for any proportion or statistic to be computed.

    Returns
    -------
    Series
        Feature values for the requested filter_type.
    """
    result = {}

    if filter_type == "cell_statistics":
        stats_df = cell_df[~cell_df["quality_model_filtered"]]
        shape_ser = _aggregate_shape_stats(
            stats_df, PB_CELL_STATISTICS_CLASSES, PB_CELL_LABEL_MAP, minimum
        )
        result.update(shape_ser.to_dict())

    elif filter_type == "cell_differential":
        living_names = [PB_CELL_LABEL_MAP[ct] for ct in PB_CELL_LINEAGES["Living_cells"]]
        n_living = int(cell_df["class_label"].isin(living_names).sum())

        cell_counts: dict[str, int] = {}
        for ct in PB_CELL_LINEAGES["All_detections"]:
            ct_name = PB_CELL_LABEL_MAP[ct]
            cell_counts[ct_name] = int((cell_df["class_label"] == ct_name).sum())

        lineage_counts: dict[str, int] = {}
        for lineage, cts in PB_CELL_LINEAGES.items():
            ct_names = [PB_CELL_LABEL_MAP[ct] for ct in cts]
            lineage_counts[lineage] = int(cell_df["class_label"].isin(ct_names).sum())

        for ct in PB_CELL_LINEAGES["Living_cells"]:
            ct_name = PB_CELL_LABEL_MAP[ct]
            result[f"Living_cells-{ct_name}_proportion"] = _min_division(cell_counts[ct_name], n_living, minimum)

        for lineage in PB_CELL_LINEAGES.keys():
            if lineage in ("All_detections", "Abnormal_platelets"):
                continue
            result[f"Living_cells-{lineage}_proportion"] = _min_division(lineage_counts[lineage], n_living, minimum)

        for ct in PB_CELL_LINEAGES["Living_cells"]:
            ct_name = PB_CELL_LABEL_MAP[ct]
            n_ct = cell_counts[ct_name]
            n_mitotic = int(((cell_df["class_label"] == ct_name) & cell_df["mitotic"]).sum())
            result[f"{ct_name}-mitotic_proportion"] = _min_division(n_mitotic, n_ct, minimum)

        for ct in PB_CELL_LINEAGES["Living_cells"]:
            ct_name = PB_CELL_LABEL_MAP[ct]
            n_ct = cell_counts[ct_name]
            n_vacuolated = int(((cell_df["class_label"] == ct_name) & cell_df["vacuolated"]).sum())
            result[f"{ct_name}-vacuolated_proportion"] = _min_division(n_vacuolated, n_ct, minimum)

        # Neutrophil granulation and segmentation proportions
        for cls_name, cts in PB_NEUT_INCLUDED_CLASSES.items():
            ct_str = _to_str_cts(cts, PB_CELL_LABEL_MAP)
            mask = _cls_mask(cell_df, ct_str)
            n_cls = int(mask.sum())
            for res_group in PB_NEUT_RESULTS:
                for res in res_group:
                    n_res = int((mask & cell_df[res]).sum())
                    result[f"{cls_name}-{res}_proportion"] = _min_division(n_res, n_cls, minimum)

    elif filter_type == "dysplasia":
        lq = cell_df["low_quality"] < 0.5

        # APL dysplasia — denominator: all cells of class; numerator: lq-filtered
        # Note: PB APL_INCLUDED_CLASSES has "Myelocytes": 12 → "Platelet_aggregations"
        # — preserved intentionally to match the upstream pipeline behaviour.
        for cls_name, cts in PB_APL_INCLUDED_CLASSES.items():
            ct_str = _to_str_cts(cts, PB_CELL_LABEL_MAP)
            mask = _cls_mask(cell_df, ct_str)
            n_cls = int(mask.sum())
            for res in PB_APL_RESULTS:
                n_res = int((mask & lq & cell_df[res]).sum())
                result[f"{cls_name}-{res}_proportion"] = _min_division(n_res, n_cls, minimum)

        # MDS dysplasia — denominator: all cells of class; numerator: lq & ~mitotic filtered
        for cls_name, cts in PB_MDS_INCLUDED_CLASSES.items():
            ct_str = _to_str_cts(cts, PB_CELL_LABEL_MAP)
            mask = _cls_mask(cell_df, ct_str)
            lq_non_mitotic_mask = mask & lq & ~cell_df["mitotic"]
            n_cls = int(mask.sum())
            for res in PB_MDS_RESULTS:
                n_res = int((lq_non_mitotic_mask & cell_df[res]).sum())
                result[f"{cls_name}-{res}_proportion"] = _min_division(n_res, n_cls, minimum)

        # Macropolycyte proportions — lq-filtered numerator
        for cls_name, cts in PB_MACROP_INCLUDED_CLASSES.items():
            ct_str = _to_str_cts(cts, PB_CELL_LABEL_MAP)
            mask = _cls_mask(cell_df, ct_str)
            n_cls = int(mask.sum())
            for res in PB_MACROP_RESULTS:
                n_res = int((mask & lq & cell_df[res]).sum())
                result[f"{cls_name}-{res}_proportion"] = _min_division(n_res, n_cls, minimum)

    elif filter_type == "all":
        parts = [
            aggregate_pb_sample(cell_df, "cell_differential", minimum),
            aggregate_pb_sample(cell_df, "dysplasia", minimum),
            aggregate_pb_sample(cell_df, "cell_statistics", minimum),
        ]
        return pd.concat(parts)

    else:
        raise ValueError(f"Unknown filter_type '{filter_type}'. Use 'cell_differential', 'dysplasia', 'cell_statistics', or 'all'.")

    return pd.Series(result)


def aggregate_bm_dataset(
    cell_df: DataFrame,
    filter_type: str = "all",
    minimum: int = 20,
    minimum_megas: int = 10,
) -> DataFrame:
    """
    Apply aggregate_bm_sample to every (database_id, album_id) group in cell_df.

    Returns a DataFrame with a MultiIndex of (database_id, album_id).
    """
    rows = {}
    for (db_id, album_id), group in cell_df.groupby(["database_id", "album_id"], sort=False):
        rows[(db_id, album_id)] = aggregate_bm_sample(
            group,
            filter_type=filter_type,
            minimum=minimum,
            minimum_megas=minimum_megas,
        )
    result = pd.DataFrame(rows).T
    result.index.names = ["database_id", "album_id"]
    return result


def aggregate_pb_dataset(
    cell_df: DataFrame,
    filter_type: str = "all",
    minimum: int = 10,
) -> DataFrame:
    """
    Apply aggregate_pb_sample to every (database_id, album_id) group in cell_df.

    Returns a DataFrame with a MultiIndex of (database_id, album_id).
    """
    rows = {}
    for (db_id, album_id), group in cell_df.groupby(["database_id", "album_id"], sort=False):
        rows[(db_id, album_id)] = aggregate_pb_sample(group, filter_type=filter_type, minimum=minimum)
    result = pd.DataFrame(rows).T
    result.index.names = ["database_id", "album_id"]
    return result
