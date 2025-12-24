#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import csv

from PIL import Image, UnidentifiedImageError
import imagehash  # perceptual hash (pHash)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Find exact/near-duplicate images using perceptual hash (pHash).")
    p.add_argument("--dir", type=str, default=None,
                   help="Scan a single directory tree (reports duplicates within it).")
    p.add_argument("--train", type=str, default=None,
                   help="Compare this directory (train) against --test (cross-compare only).")
    p.add_argument("--test", type=str, default=None,
                   help="Compare this directory (test) against --train (cross-compare only).")
    p.add_argument("--hash-size", type=int, default=16,
                   help="pHash size (default 16 => 64-bit hash). Larger = slower but more robust.")
    p.add_argument("--threshold", type=int, default=5,
                   help="Hamming distance threshold for near-duplicates (0 = exact match). Common values: 5-10.")
    p.add_argument("--bucket-prefix-hex", type=int, default=4,
                   help="Bucket by first N hex chars to reduce comparisons (default 4). Increase to speed up, may miss some near matches.")
    p.add_argument("--csv", type=str, default="phash_duplicates.csv",
                   help="Output CSV file path.")
    p.add_argument("--max-images", type=int, default=None,
                   help="Optional: limit images for quick run.")
    return p.parse_args()


def collect_images(root: Path, max_images: int | None) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
            if max_images and len(files) >= max_images:
                break
    return files


def safe_open(path: Path) -> Image.Image | None:
    try:
        img = Image.open(path)
        img = img.convert("RGB")  # pHash works on luminance; RGB is fine here
        return img
    except (UnidentifiedImageError, OSError):
        return None


def compute_phash(img: Image.Image, hash_size: int) -> imagehash.ImageHash:
    return imagehash.phash(img, hash_size=hash_size)


def hamming_distance(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> int:
    return h1 - h2  # imagehash defines subtraction as Hamming distance


def hex_prefix(h: imagehash.ImageHash, n_hex: int) -> str:
    # ImageHash.__str__ returns hex string
    return str(h)[:max(0, n_hex)]


def scan_single_dir(root: Path, hash_size: int, threshold: int, bucket_prefix_hex: int,
                    max_images: int | None, out_csv: Path):
    files = collect_images(root, max_images)
    print(f"Scanning {len(files)} images under: {root}")

    # Hash all images
    hashes: Dict[Path, imagehash.ImageHash] = {}
    for p in tqdm(files, desc="Hashing"):
        img = safe_open(p)
        if img is None:
            continue
        hashes[p] = compute_phash(img, hash_size)

    # Build buckets by hex prefix to reduce pairwise checks
    buckets: Dict[str, List[Tuple[Path, imagehash.ImageHash]]] = {}
    for path, h in hashes.items():
        pref = hex_prefix(h, bucket_prefix_hex)
        buckets.setdefault(pref, []).append((path, h))

    # Find exact and near duplicates within each bucket
    duplicates: List[Tuple[str, str, int]] = []
    for pref, items in tqdm(buckets.items(), desc="Comparing within buckets"):
        n = len(items)
        for i in range(n):
            p1, h1 = items[i]
            for j in range(i + 1, n):
                p2, h2 = items[j]
                dist = hamming_distance(h1, h2)
                if dist <= threshold:
                    duplicates.append((str(p1), str(p2), dist))

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path_a", "path_b", "hamming_distance"])
        w.writerows(duplicates)
    print(f"Done. Found {len(duplicates)} near-duplicate pairs (<= {threshold}). CSV: {out_csv}")


def cross_compare(train_root: Path, test_root: Path, hash_size: int, threshold: int,
                  bucket_prefix_hex: int, max_images: int | None, out_csv: Path):
    train_files = collect_images(train_root, max_images)
    test_files = collect_images(test_root, max_images)
    print(f"Hashing train ({len(train_files)}) and test ({len(test_files)})")

    # Hash train set and bucket
    train_hashes: Dict[Path, imagehash.ImageHash] = {}
    for p in tqdm(train_files, desc="Hashing train"):
        img = safe_open(p)
        if img is None:
            continue
        train_hashes[p] = compute_phash(img, hash_size)

    train_buckets: Dict[str, List[Tuple[Path, imagehash.ImageHash]]] = {}
    for path, h in train_hashes.items():
        pref = hex_prefix(h, bucket_prefix_hex)
        train_buckets.setdefault(pref, []).append((path, h))

    # Hash test set and compare only against matching buckets
    duplicates: List[Tuple[str, str, int]] = []
    for p in tqdm(test_files, desc="Hashing & comparing test"):
        img = safe_open(p)
        if img is None:
            continue
        h = compute_phash(img, hash_size)
        pref = hex_prefix(h, bucket_prefix_hex)
        candidates = train_buckets.get(pref, [])
        for t_path, t_hash in candidates:
            dist = hamming_distance(h, t_hash)
            if dist <= threshold:
                duplicates.append((str(t_path), str(p), dist))

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_path", "test_path", "hamming_distance"])
        w.writerows(duplicates)
    print(f"Done. Found {len(duplicates)} train-vs-test near-duplicate pairs (<= {threshold}). CSV: {out_csv}")


def main():
    args = parse_args()
    out_csv = Path(args.csv)

    if args.train and args.test:
        cross_compare(
            Path(args.train), Path(args.test),
            hash_size=args.hash_size,
            threshold=args.threshold,
            bucket_prefix_hex=args.bucket_prefix_hex,
            max_images=args.max_images,
            out_csv=out_csv
        )
    elif args.dir:
        scan_single_dir(
            Path(args.dir),
            hash_size=args.hash_size,
            threshold=args.threshold,
            bucket_prefix_hex=args.bucket_prefix_hex,
            max_images=args.max_images,
            out_csv=out_csv
        )
    else:
        print("Provide either --dir or both --train and --test. Use --help for options.")


if __name__ == "__main__":
    main()