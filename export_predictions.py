#!/usr/bin/env python3
"""Zip predictions.csv files from one or more SCEMILA run directories.

Intended to be moved to the modelling repo (aml-mil) once integration is done.

Usage (run from the modelling repo):
    python export_predictions.py \\
        results/.../BM/All\\ cells/run_2026-05-19 \\
        results/.../PB/All\\ cells/run_2026-05-19 \\
        results/.../FUSION/All\\ cells/run_2026-05-05 \\
        -o predictions_v15.zip
"""
import argparse
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Run directories containing predictions.csv",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("predictions.zip"),
        help="Output zip file (default: predictions.zip)",
    )
    args = parser.parse_args()

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for run_dir in args.run_dirs:
            pred = run_dir / "predictions.csv"
            if not pred.exists():
                raise FileNotFoundError(f"No predictions.csv in {run_dir}")
            zf.write(pred, arcname=f"{run_dir.name}/predictions.csv")
            print(f"  added {run_dir.name}/predictions.csv")

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
