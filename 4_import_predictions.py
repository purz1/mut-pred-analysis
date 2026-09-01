#!/usr/bin/env python3
"""Unzip a predictions archive into data/{DATA_VERSION_IDENTIFIER}/model_predictions/.

Usage:
    python 4_import_predictions.py
"""

import zipfile
from pathlib import Path

from config import DATA_VERSION_IDENTIFIER

# ── Configure here ────────────────────────────────────────────────────────────
INPUT_ZIP  = Path("predictions.zip")
OUTPUT_DIR = Path(f"data/{DATA_VERSION_IDENTIFIER}/model_predictions")
# ─────────────────────────────────────────────────────────────────────────────


def main():
    if not INPUT_ZIP.exists():
        raise FileNotFoundError(f"Zip not found: {INPUT_ZIP}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(INPUT_ZIP, "r") as zf:
        for member in zf.namelist():
            dest = OUTPUT_DIR / member
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(member))
            print(f"  extracted {member}")

    INPUT_ZIP.unlink()
    print(f"\nDone — files in {OUTPUT_DIR}/")
    print(f"Deleted {INPUT_ZIP}")


if __name__ == "__main__":
    main()
