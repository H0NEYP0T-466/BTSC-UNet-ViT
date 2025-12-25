#!/usr/bin/env python3
"""
Find exact (byte-identical) duplicate image files by SHA-256.

- Scans a directory tree for image files
- Computes SHA-256 for each file
- Groups files with identical digests (exact duplicates)
- Writes results to CSV and prints a summary

Usage (Windows CMD):
  python SHA-256.py --dir "X:\file\FAST_API\BTSC-UNet-ViT\backend\dataset\Vit_Dataset" --csv "X:\file\FAST_API\BTSC-UNet-ViT\backend\resources\artifacts\exact_duplicates.csv"

Options:
  --dir    Root directory to scan (required)
  --csv    Output CSV file path (optional; default: exact_duplicates.csv next to script)
  --exts   Comma-separated extensions to include (default: .jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp)
  --top    Print up to N example groups in console (default: 10)
"""

import argparse
import hashlib
from pathlib import Path
from collections import defaultdict
import csv
import sys

DEFAULT_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_args():
    ap = argparse.ArgumentParser(description="Find exact duplicate image files by SHA-256.")
    ap.add_argument("--dir", required=True, help="Root directory to scan")
    ap.add_argument("--csv", default=None, help="Output CSV path (optional)")
    ap.add_argument("--exts", default=",".join(sorted(DEFAULT_EXTS)),
                    help="Comma-separated extensions to include (default: common image types)")
    ap.add_argument("--top", type=int, default=10, help="Print up to N example duplicate groups (default: 10)")
    return ap.parse_args()

def main():
    args = parse_args()
    root = Path(args.dir)
    if not root.exists():
        print(f"Directory not found: {root}")
        sys.exit(1)

    exts = {e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower()
            for e in args.exts.split(",") if e.strip()}
    print(f"Scanning under: {root}")
    print(f"Extensions: {sorted(exts)}")

    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    print(f"Found {len(files)} candidate files")

    buckets = defaultdict(list)
    errors = 0
    for i, p in enumerate(files, start=1):
        try:
            digest = sha256(p)
            buckets[digest].append(p)
        except Exception as e:
            errors += 1
            # Optionally print errors: print(f"Error hashing {p}: {e}")
        if i % 5000 == 0:
            print(f"Hashed {i}/{len(files)}...")

    duplicate_groups = [paths for paths in buckets.values() if len(paths) > 1]
    total_dupe_files = sum(len(g) for g in duplicate_groups)

    print("\n=== Exact Duplicate Summary ===")
    print(f"Total files scanned: {len(files)}")
    print(f"Hashing errors: {errors}")
    print(f"Duplicate groups (size >= 2): {len(duplicate_groups)}")
    print(f"Files involved in duplicates: {total_dupe_files}")

    # Print some example groups
    for idx, group in enumerate(sorted(duplicate_groups, key=len, reverse=True)[:args.top], start=1):
        print(f"\nGroup {idx} (size={len(group)}):")
        for p in group:
            print(f"  {p}")

    # Write CSV if requested (or default next to script)
    out_csv = Path(args.csv) if args.csv else Path(__file__).with_name("exact_duplicates.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sha256", "path"])
        for digest, paths in buckets.items():
            if len(paths) > 1:
                for p in paths:
                    w.writerow([digest, str(p)])
    print(f"\nWrote duplicate listing to: {out_csv}")

if __name__ == "__main__":
    main()