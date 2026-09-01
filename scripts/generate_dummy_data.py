#!/usr/bin/env python3
"""Generate a synthetic _DUMMY companion for any CSV, Excel, or Parquet file.

The dummy file is written to the same directory as the input, with ``_DUMMY``
inserted before the extension:

    data/07_04_2026/survival/aml_mut_pred_survival_data_bm_07-04-2026.xlsx
    → data/07_04_2026/survival/aml_mut_pred_survival_data_bm_07-04-2026_DUMMY.xlsx

Usage:
    python scripts/generate_dummy_data.py <input_file> [--n N] [--id-cols col1,col2]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from core.dummy import generate_dummy_df


def _read(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def _write(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path, help="CSV, Excel, or Parquet file to generate dummy data from")
    parser.add_argument("--n", type=int, default=500, help="Number of synthetic rows (default: 30)")
    parser.add_argument("--id-cols", type=str, default="", help="Comma-separated list of ID columns to replace with DUMMY_NNNN values")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = args.input_file
    if not src.exists():
        raise FileNotFoundError(src)

    id_columns = [c.strip() for c in args.id_cols.split(",") if c.strip()]

    df = _read(src)
    dummy = generate_dummy_df(df, n=args.n, seed=args.seed, id_columns=id_columns or None)

    out = src.parent / f"{src.stem}_DUMMY{src.suffix}"
    _write(dummy, out)
    print(f"Written: {out}  ({len(dummy)} rows × {len(dummy.columns)} columns)")


if __name__ == "__main__":
    main()
