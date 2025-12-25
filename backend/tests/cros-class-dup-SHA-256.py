#!/usr/bin/env python3
"""
Detect cross-class exact duplicates using SHA-256.

- Scans dataset directory
- Computes SHA-256 for each image
- Finds cases where SAME hash appears in MULTIPLE class folders
- Does NOT delete anything (safe)

Usage:
  python cros-class-dup-SHA-256.py --dir "X:\file\FAST_API\BTSC-UNet-ViT\backend\dataset\Vit_Dataset"
"""

import hashlib
from pathlib import Path
from collections import defaultdict
import argparse

DEFAULT_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def sha256(path: Path, chunk_size=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_args():
    ap = argparse.ArgumentParser(description="Detect cross-class exact duplicates using SHA-256.")
    ap.add_argument("--dir", required=True, help="Root dataset directory (class folders inside)")
    ap.add_argument("--exts", default=",".join(DEFAULT_EXTS), help="Comma-separated image extensions")
    return ap.parse_args()

def main():
    args = parse_args()
    root = Path(args.dir)

    if not root.exists():
        print(f"[ERROR] Directory not found: {root}")
        return

    exts = {e if e.startswith(".") else "." + e for e in args.exts.lower().split(",")}

    print(f"🔍 Scanning dataset: {root}")
    print(f"📁 Extensions: {sorted(exts)}")

    hash_map = defaultdict(list)

    for img in root.rglob("*"):
        if img.is_file() and img.suffix.lower() in exts:
            try:
                h = sha256(img)
                class_name = img.parent.name
                hash_map[h].append((class_name, img))
            except Exception as e:
                print(f"[ERR] Failed hashing {img}: {e}")

    cross_class_cases = []

    for h, items in hash_map.items():
        classes = {cls for cls, _ in items}
        if len(classes) > 1:
            cross_class_cases.append((h, items))

    print("\n================ CROSS-CLASS DUPLICATE REPORT ================")
    print(f"Total images scanned: {sum(len(v) for v in hash_map.values())}")
    print(f"Cross-class duplicate hashes found: {len(cross_class_cases)}")

    if not cross_class_cases:
        print("✅ No cross-class duplicates found. Dataset labels look clean.")
        return

    for idx, (h, items) in enumerate(cross_class_cases, start=1):
        print(f"\n🚨 Case {idx}")
        print(f"SHA-256: {h}")
        for cls, path in items:
            print(f"  - Class: {cls:12s} | File: {path}")

    print("\n⚠️ These images have IDENTICAL pixels but DIFFERENT class labels.")
    print("You MUST remove or manually resolve them before training.")

if __name__ == "__main__":
    main()
