"""Synthetic data generation for local testing without real patient files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_dummy_df(
    df: pd.DataFrame,
    n: int = 30,
    seed: int = 42,
    id_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return a synthetic DataFrame with the same schema as *df*.

    Per-column strategy:
    - Columns listed in *id_columns* (e.g. ``album_id``, ``henkilotunnus``):
      sequential synthetic IDs (``DUMMY_0001``, ``DUMMY_0002``, …).
    - ``bool`` dtype: sample with the same ``True`` prevalence.
    - Numeric (int / float): sample with replacement from real non-null values.
    - ``object`` / categorical: sample with replacement from real non-null values.
    - ``datetime``: sample uniformly between real min and max.
    - NaN pattern: each column's observed null rate is preserved.
    """
    rng = np.random.default_rng(seed)
    id_cols = set(id_columns or [])
    rows: dict[str, list] = {}

    for col in df.columns:
        series = df[col]
        null_mask = rng.random(n) < series.isna().mean()

        if col in id_cols:
            values = [f"DUMMY_{i:04d}" for i in range(n)]

        elif pd.api.types.is_bool_dtype(series):
            p_true = series.dropna().mean()
            values = (rng.random(n) < p_true).tolist()

        elif pd.api.types.is_datetime64_any_dtype(series):
            non_null = series.dropna()
            if len(non_null) == 0:
                values = [pd.NaT] * n
            else:
                lo = non_null.min().value
                hi = non_null.max().value
                sampled = rng.integers(lo, max(hi, lo + 1), size=n)
                values = pd.to_datetime(sampled).tolist()

        elif pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna().values
            if len(non_null) == 0:
                values = [np.nan] * n
            else:
                sampled = rng.choice(non_null, size=n, replace=True)
                if pd.api.types.is_integer_dtype(series):
                    values = sampled.astype(int).tolist()
                else:
                    values = sampled.tolist()

        else:
            non_null = series.dropna().values
            if len(non_null) == 0:
                values = [None] * n
            else:
                values = rng.choice(non_null, size=n, replace=True).tolist()

        if col not in id_cols:
            for i in np.where(null_mask)[0]:
                values[i] = None if not pd.api.types.is_numeric_dtype(series) else np.nan

        rows[col] = values

    return pd.DataFrame(rows)


def _dummy_path(path: Path) -> Path:
    return path.parent / f"{path.stem}_DUMMY{path.suffix}"


def _save(df: pd.DataFrame, path: Path, **kwargs) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False, **kwargs)
    elif suffix in (".xlsx", ".xls"):
        df.to_excel(path, index=False, **kwargs)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def save_with_dummy(
    df: pd.DataFrame,
    path: str | Path,
    id_columns: list[str] | None = None,
    n: int = 30,
    seed: int = 42,
    **kwargs,
) -> None:
    """Save *df* to *path* and write a ``_DUMMY`` companion in the same directory.

    The dummy file has the same name as the real file but with ``_DUMMY`` inserted
    before the extension, e.g.::

        data/03-06-2026/survival/aml_mut_pred_survival_data_bm_03-06-2026.xlsx
        data/03-06-2026/survival/aml_mut_pred_survival_data_bm_03-06-2026_DUMMY.xlsx

    If the real file already exists, saving is skipped — update
    DATA_VERSION_IDENTIFIER in config.py to save a new version of the data.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists():
        print(
            f"Skipped (already exists): {p}\n"
            "  → Update DATA_VERSION_IDENTIFIER in config.py to save a new version."
        )
        return

    _save(df, p, **kwargs)
    print(f"Saved: {p}")

    dummy = generate_dummy_df(df, n=n, seed=seed, id_columns=id_columns)
    dummy_p = _dummy_path(p)
    _save(dummy, dummy_p, **kwargs)
    print(f"Saved: {dummy_p}")
