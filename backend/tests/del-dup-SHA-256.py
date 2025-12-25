#!/usr/bin/env python3
"""
Delete exact (SHA-256) and near-duplicate (pHash) image files.

- Exact duplicates: byte-identical files (SHA-256). Keep ONE per hash (strategy), delete the rest.
- Near duplicates: perceptually similar files (pHash Hamming distance <= threshold).
  Group similar images and keep ONE per group (strategy), delete the rest.

Defaults are safe:
- Dry-run by default (no deletions). Use --do-delete to apply changes.
- Near-duplicate deletion is limited to within the same class folder (use --near-scope global to cross classes).

Requirements:
  pip install pillow imagehash

Usage examples (Windows CMD):
  python delete_exact_dupes.py --dir "X:\\file\\FAST_API\\BTSC-UNet-ViT\\backend\\dataset\\Vit_Dataset"
  python delete_exact_dupes.py --dir "X:\\...\\Vit_Dataset" --keep newest --phash-threshold 6 --do-delete
  python delete_exact_dupes.py --dir "X:\\...\\Vit_Dataset" --near-scope global --phash-threshold 6 --do-delete

  python del-dup-SHA-256.py --dir "X:\file\FAST_API\BTSC-UNet-ViT\backend\dataset\Vit_Dataset" --keep newest --phash-threshold 6 --near-scope within-class --do-delete

Options:
  --dir            Root directory to scan (required)
  --exts           Comma-separated extensions to include (default: .jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp)
  --keep           Keep strategy in duplicate groups: first | oldest | newest | smallest | largest (default: first)
  --do-delete      Actually delete files (omit for dry-run)
  --progress       Print hashing progress every N files (default: 5000)
  --mode           What to delete: all | exact-only | near-only (default: all)
  --phash-threshold  Hamming distance threshold for near-duplicates (default: 6; typical 5–8)
  --phash-hash-size  pHash size (default: 16; larger is slower but more robust)
  --phash-bucket-prefix-hex  Bucket by first N hex chars to reduce comparisons (default: 6)
  --near-scope     Limit near-duplicate deletion to within-class or global (default: within-class)
"""

import argparse
import hashlib
from pathlib import Path
from collections import defaultdict
import sys
import os
import time

from PIL import Image, UnidentifiedImageError
import imagehash

DEFAULT_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 digest of a file in streaming mode."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_open_image(path: Path) -> Image.Image | None:
    try:
        img = Image.open(path)
        # Convert to RGB; imagehash handles luminance internally, but RGB is fine
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None

def compute_phash(img: Image.Image, hash_size: int) -> imagehash.ImageHash:
    return imagehash.phash(img, hash_size=hash_size)

def hamming_distance(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> int:
    return h1 - h2

def phash_hex_prefix(h: imagehash.ImageHash, n_hex: int) -> str:
    return str(h)[:max(0, n_hex)]

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

def parent_class(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
        parts = rel.parts
        return parts[0] if len(parts) > 1 else "(root)"
    except Exception:
        return path.parent.name

class DSU:
    """Disjoint-set union for grouping near-duplicates."""
    def __init__(self):
        self.parent = {}
        self.size = {}

    def find(self, x):
        px = self.parent.get(x, x)
        if px != x:
            self.parent[x] = self.find(px)
        else:
            self.parent.setdefault(x, x)
            self.size.setdefault(x, 1)
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.size[ra] < self.size[rb]: ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

def parse_args():
    ap = argparse.ArgumentParser(description="Delete exact and near-duplicate image files.")
    ap.add_argument("--dir", required=True, help="Root directory to scan")
    ap.add_argument("--exts", default=",".join(sorted(DEFAULT_EXTS)), help="Comma-separated extensions to include")
    ap.add_argument("--keep", choices=["first", "oldest", "newest", "smallest", "largest"], default="first",
                    help="Which file to keep in each duplicate group")
    ap.add_argument("--do-delete", action="store_true", help="Actually delete files (otherwise dry-run)")
    ap.add_argument("--progress", type=int, default=5000, help="Print hashing progress every N files")
    ap.add_argument("--mode", choices=["all", "exact-only", "near-only"], default="all",
                    help="Choose deletion type")
    ap.add_argument("--phash-threshold", type=int, default=6, help="pHash Hamming distance threshold (near-duplicate)")
    ap.add_argument("--phash-hash-size", type=int, default=16, help="pHash size")
    ap.add_argument("--phash-bucket-prefix-hex", type=int, default=6, help="Bucket prefix length (hex) for pHash")
    ap.add_argument("--near-scope", choices=["within-class", "global"], default="within-class",
                    help="Limit near-duplicate deletion to within the same class folder or across all classes")
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
    print(f"Deletion type: {args.mode}")
    if args.mode in ("all", "near-only"):
        print(f"Near-duplicate threshold: {args.phash_threshold} (hash_size={args.phash_hash_size}, bucket_prefix={args.phash_bucket_prefix_hex}, scope={args.near_scope})")

    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    print(f"Found {len(files)} candidate files")

    # Collect class names for files
    file_class = {p: parent_class(root, p) for p in files}

    to_delete = set()

    # 1) Exact duplicates (SHA-256)
    if args.mode in ("all", "exact-only"):
        print("\n[Exact] Computing SHA-256 for exact duplicates...")
        sha_buckets = defaultdict(list)
        errors_sha = 0
        start = time.time()
        for i, p in enumerate(files, start=1):
            try:
                digest = sha256(p)
                sha_buckets[digest].append(p)
            except Exception:
                errors_sha += 1
            if args.progress and i % args.progress == 0:
                print(f"[Exact] Hashed {i}/{len(files)}...")
        elapsed_sha = time.time() - start
        exact_groups = [paths for paths in sha_buckets.values() if len(paths) > 1]
        files_in_exact_groups = sum(len(g) for g in exact_groups)
        print(f"[Exact] Elapsed: {elapsed_sha:.1f}s | Errors: {errors_sha}")
        print(f"[Exact] Groups: {len(exact_groups)} | Files in groups: {files_in_exact_groups}")

        planned_exact = 0
        for group in exact_groups:
            keep = choose_keep(group, args.keep)
            for dup in group:
                if dup == keep:
                    continue
                if dup in to_delete:
                    continue
                to_delete.add(dup)
                planned_exact += 1
                if args.do_delete:
                    try:
                        os.remove(dup)
                        print(f"[DEL-EXACT] {dup}")
                    except Exception as e:
                        print(f"[ERR] Exact delete failed {dup}: {e}")
                else:
                    print(f"[DRY-EXACT] Would delete: {dup} (keeping: {keep})")
        print(f"[Exact] {'Deleted' if args.do_delete else 'Planned'}: {planned_exact}")

    # 2) Near duplicates (pHash)
    if args.mode in ("all", "near-only"):
        remaining_files = [p for p in files if p not in to_delete]
        print("\n[Near] Computing pHash for remaining files...")
        phashes = {}
        errors_ph = 0
        start = time.time()
        for i, p in enumerate(remaining_files, start=1):
            img = safe_open_image(p)
            if img is None:
                errors_ph += 1
                continue
            try:
                ph = compute_phash(img, args.phash_hash_size)
                phashes[p] = ph
            except Exception:
                errors_ph += 1
            if args.progress and i % args.progress == 0:
                print(f"[Near] Hashed {i}/{len(remaining_files)}...")
        elapsed_ph = time.time() - start
        print(f"[Near] Elapsed: {elapsed_ph:.1f}s | Errors: {errors_ph} | Hashed: {len(phashes)}")

        # Bucket by prefix to reduce pairwise comparisons
        buckets = defaultdict(list)  # prefix -> [(path, phash)]
        for p, h in phashes.items():
            pref = phash_hex_prefix(h, args.phash_bucket_prefix_hex)
            buckets[pref].append((p, h))

        # Group near duplicates via DSU
        dsu = DSU()
        for p in phashes.keys():
            dsu.find(p)  # initialize

        comps_edges = 0
        for pref, items in buckets.items():
            n = len(items)
            if n < 2:
                continue
            for i in range(n):
                p1, h1 = items[i]
                for j in range(i + 1, n):
                    p2, h2 = items[j]
                    # Scope check: within-class vs global
                    if args.near_scope == "within-class" and file_class[p1] != file_class[p2]:
                        continue
                    dist = hamming_distance(h1, h2)
                    if dist <= args.phash_threshold:
                        dsu.union(p1, p2)
                        comps_edges += 1

        # Build components
        comps = defaultdict(list)
        for p in phashes.keys():
            root_id = dsu.find(p)
            comps[root_id].append(p)

        near_groups = [members for members in comps.values() if len(members) > 1]
        print(f"[Near] Near-duplicate groups: {len(near_groups)} | edges: {comps_edges}")

        planned_near = 0
        for members in near_groups:
            keep = choose_keep(members, args.keep)
            for dup in members:
                if dup == keep:
                    continue
                if dup in to_delete:
                    continue  # already scheduled/deleted by exact step
                to_delete.add(dup)
                planned_near += 1
                if args.do_delete:
                    try:
                        os.remove(dup)
                        print(f"[DEL-NEAR] {dup}")
                    except Exception as e:
                        print(f"[ERR] Near delete failed {dup}: {e}")
                else:
                    print(f"[DRY-NEAR] Would delete: {dup} (keeping: {keep})")

        print(f"[Near] {'Deleted' if args.do_delete else 'Planned'}: {planned_near}")

    total_actions = len(to_delete)
    print(f"\n=== Summary ===")
    print(f"Total {'deleted' if args.do_delete else 'planned'} files: {total_actions}")
    if not args.do_delete:
        print("Dry-run complete. Re-run with --do-delete to apply deletions.")

if __name__ == "__main__":
    main()