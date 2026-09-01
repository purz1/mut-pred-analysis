#!/usr/bin/env python3
"""Zip the project for transfer.

Three modes:
  safe (default)  Include only _DUMMY data files; exclude all real patient data.
                  Recipients can run notebooks immediately with USE_DUMMY_DATA = True.
  code            No data files at all — just code, notebooks, config, and env files.
  full            Include all data files (real + model predictions).
                  Use for backup or transfer to an environment with real data.

Build artifacts (.venv/, __pycache__/, .git/) are always excluded.
data/source_data_from_academic/ is always excluded.
eln_run_config.R is always excluded (machine-specific absolute paths).

Usage:
    python scripts/zip_project.py [--mode {safe,code,full}] [--name OUTPUT]
"""

import argparse
import zipfile
from pathlib import Path

ALWAYS_EXCLUDE = {
    ".venv", "__pycache__", ".git", ".mypy_cache", ".ruff_cache",
    ".vscode", "figures", "project_zips",
}
ALWAYS_EXCLUDE_FILES = {"eln_run_config.R"}
ALWAYS_EXCLUDE_DATA_SUBDIRS = {"source_data_from_academic"}


def should_include(rel: Path, mode: str) -> bool:
    parts = rel.parts

    for part in parts:
        if part in ALWAYS_EXCLUDE:
            return False
    if rel.name in ALWAYS_EXCLUDE_FILES:
        return False
    if any(str(rel).endswith(ext) for ext in (".pyc", ".pyo")):
        return False

    if parts[0] == "data":
        if len(parts) >= 2 and parts[1] in ALWAYS_EXCLUDE_DATA_SUBDIRS:
            return False
        if mode == "full":
            return True
        if mode == "code":
            return False
        if mode == "safe":
            return "_DUMMY" in rel.stem

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode", choices=["safe", "code", "full"], default="safe",
        help="safe: only _DUMMY data files (default). "
             "code: no data files. "
             "full: all data files.",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Output filename inside project_zips/ (default: aml_mut_pred_analysis_{mode}.zip)",
    )
    args = parser.parse_args()

    mode = args.mode
    root = Path(".").resolve()
    zip_dir = Path("project_zips")
    zip_dir.mkdir(parents=True, exist_ok=True)
    output = (zip_dir / (args.name or f"aml_mut_pred_analysis_{mode}.zip")).resolve()

    included = []
    excluded = []

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root)
        if output == path:
            continue
        if should_include(rel, mode):
            included.append((path, rel))
        else:
            excluded.append(rel)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, rel in included:
            zf.write(path, arcname=rel)

    size_mb = output.stat().st_size / 1024 / 1024
    print(f"Mode:     {mode}")
    print(f"Included: {len(included)} files")
    print(f"Excluded: {len(excluded)} files")
    print(f"Size:     {size_mb:.1f} MB")
    print(f"\nWritten to: {output}")


if __name__ == "__main__":
    main()
