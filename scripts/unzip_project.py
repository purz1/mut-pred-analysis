#!/usr/bin/env python3
"""Unzip a project archive with per-file overwrite confirmation.

Usage:
    python scripts/unzip_project.py <zipfile> [--output-dir DIR]
"""
import argparse
import sys
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zipfile", type=Path, help="Path to the zip archive")
    parser.add_argument("--output-dir", "-d", type=Path, default=Path("."), help="Destination directory (default: .)")
    args = parser.parse_args()

    if not args.zipfile.exists():
        sys.exit(f"Error: {args.zipfile} not found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    new_count = replaced_count = skipped_count = 0
    overwrite_all = False

    with zipfile.ZipFile(args.zipfile, "r") as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            dest = args.output_dir / member
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(member))
                print(f"  new      {member}")
                new_count += 1
            elif overwrite_all:
                dest.write_bytes(zf.read(member))
                print(f"  replaced {member}")
                replaced_count += 1
            else:
                answer = input(f"overwrite? {member}  [y/N/a/q] : ").strip().lower()
                if answer == "q":
                    print("Aborted.")
                    break
                elif answer in ("a", "y"):
                    if answer == "a":
                        overwrite_all = True
                    dest.write_bytes(zf.read(member))
                    print(f"  replaced {member}")
                    replaced_count += 1
                else:
                    print(f"  skipped  {member}")
                    skipped_count += 1

    print(f"\nDone — {new_count} new, {replaced_count} replaced, {skipped_count} skipped")


if __name__ == "__main__":
    main()
