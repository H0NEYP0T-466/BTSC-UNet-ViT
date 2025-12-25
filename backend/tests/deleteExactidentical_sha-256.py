#!/usr/bin/env python3
"""
Delete exact (byte-identical) duplicate image files by SHA-256.

- Scans a directory tree for image files
- Computes SHA-256 for each file
- Keeps ONE canonical file per hash (configurable strategy)
- Deletes the rest (exact duplicates only; near-duplicates are ignored)

Safety:
- By default runs in DRY-RUN mode (no deletions), showing what would be deleted.
- Pass --do-delete to actually delete the files.

Usage (Windows CMD):
  python deleteExactidentical_sha-256.py --dir "X:\file\FAST_API\BTSC-UNet-ViT\backend\dataset\Vit_Dataset" --keep newest --do-delete
  python deleteExactidentical_sha-256.py --dir "X:\file\...\Vit_Dataset" --keep newest --do-delete
    
Options:
  --dir        Root directory to scan (required)
  --exts       Comma-separated extensions to include (default: .jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp)
  --keep       Which file to keep per duplicate group: first | oldest | newest | smallest | largest (default: first)
  --do-delete  Actually delete duplicates (omit to run as dry-run)
  --progress   Print hashing progress every N files (default: 5000)
"""

import argparse
import hashlib
from pathlib import Path
from collections import defaultdict
import sys
import os
import time

DEFAULT_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 digest of a file in streaming mode."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def choose_keep(paths, strategy: str) -> Path:
    """Pick the canonical file to keep within a duplicate group."""
    if strategy == "first":
        return paths[0]
    stats = []
    for p in paths:
        try:
            st = p.stat()
            stats.append((p, st.st_mtime, st.st_size))
        except Exception:
            stats.append((p, 0, 0))
    if strategy == "oldest":
        return sorted(stats, key=lambda x: x[1])[0][0]
    if strategy == "newest":
        return sorted(stats, key=lambda x: x[1], reverse=True)[0][0]
    if strategy == "smallest":
        return sorted(stats, key=lambda x: x[2])[0][0]
    if strategy == "largest":
        return sorted(stats, key=lambda x: x[2], reverse=True)[0][0]
    return paths[0]

def parse_args():
    ap = argparse.ArgumentParser(description="Delete exact duplicate image files by SHA-256.")
    ap.add_argument("--dir", required=True, help="Root directory to scan")
    ap.add_argument("--exts", default=",".join(sorted(DEFAULT_EXTS)), help="Comma-separated extensions to include")
    ap.add_argument("--keep", choices=["first", "oldest", "newest", "smallest", "largest"], default="first",
                    help="Which file to keep in each duplicate group")
    ap.add_argument("--do-delete", action="store_true", help="Actually delete duplicate files (otherwise dry-run)")
    ap.add_argument("--progress", type=int, default=5000, help="Print hashing progress every N files")
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
    print(f"Mode: {'DELETE' if args.do_delete else 'DRY-RUN (no deletions)'}")
    print(f"Keep strategy: {args.keep}")

    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    print(f"Found {len(files)} candidate files")

    buckets = defaultdict(list)
    errors = 0
    start = time.time()
    for i, p in enumerate(files, start=1):
        try:
            digest = sha256(p)
            buckets[digest].append(p)
        except Exception as e:
            errors += 1
            # Uncomment to debug: print(f"[ERR] Hashing {p}: {e}")
        if args.progress and i % args.progress == 0:
            print(f"Hashed {i}/{len(files)}...")

    elapsed = time.time() - start
    duplicate_groups = [paths for paths in buckets.values() if len(paths) > 1]
    files_in_groups = sum(len(g) for g in duplicate_groups)

    print("\n=== Exact Duplicate Scan Summary ===")
    print(f"Elapsed hashing: {elapsed:.1f}s")
    print(f"Total files scanned: {len(files)}")
    print(f"Hashing errors: {errors}")
    print(f"Duplicate groups (size >= 2): {len(duplicate_groups)}")
    print(f"Files in duplicate groups: {files_in_groups}")

    planned_deletions = 0
    for group in duplicate_groups:
        keep = choose_keep(group, args.keep)
        # Delete all except the chosen keep
        for dup in group:
            if dup == keep:
                continue
            planned_deletions += 1
            if args.do_delete:
                try:
                    os.remove(dup)
                    print(f"[DEL] {dup}")
                except Exception as e:
                    print(f"[ERR] Failed to delete {dup}: {e}")
            else:
                print(f"[DRY] Would delete: {dup} (keeping: {keep})")

    print(f"\nTotal {'deleted' if args.do_delete else 'planned'} duplicates: {planned_deletions}")
    if not args.do_delete:
        print("Dry-run complete. Re-run with --do-delete to actually remove exact duplicates.")

if __name__ == "__main__":
    main()