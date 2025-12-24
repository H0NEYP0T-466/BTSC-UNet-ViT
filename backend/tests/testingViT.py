#!/usr/bin/env python3
"""
Evaluate the ViT model using the same inference pipeline as the repo (ViTInference).

- Loads the model via app.models.vit.infer_vit.ViTInference (uses get_vit_model and repo transforms).
- Evaluates a 4-class ImageFolder dataset:
    data_dir/
      notumor/
      glioma/
      meningioma/
      pituitary/
- Processes images one-by-one, computes CE loss and accuracy, per-class metrics, and confusion matrix.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import cv2

# Ensure we can import the repo's app package when running from backend/tests
REPO_BACKEND_DIR = Path(__file__).resolve().parents[1]  # points to backend/
if str(REPO_BACKEND_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_BACKEND_DIR))

from app.models.vit.infer_vit import ViTInference  # repo's inference wrapper
from app.config import settings  # repo settings (class names, image size, defaults)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ViT with repo-consistent inference.")
    p.add_argument("--data-dir", type=str, required=True, help="Path to ImageFolder dataset with 4 class subfolders.")
    p.add_argument("--checkpoint", type=str, default=None, help="Optional path to checkpoint (.pth). Defaults to settings.")
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"], help="Force device (default: auto).")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (use 1 for one-by-one).")
    p.add_argument("--limit", type=int, default=None, help="Optional: evaluate only first N images.")
    p.add_argument("--save-csv", type=str, default=None, help="Optional: path to save misclassifications CSV.")
    return p.parse_args()


def walk_image_folder(data_dir: Path, class_names: List[str]) -> List[Tuple[Path, int]]:
    """
    Walk class folders and collect image paths with labels according to class_names order.
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    samples: List[Tuple[Path, int]] = []
    for idx, cname in enumerate(class_names):
        cdir = data_dir / cname
        if not cdir.exists():
            print(f"WARNING: Class directory not found: {cdir}")
            continue
        for path in cdir.rglob("*"):
            if path.suffix.lower() in exts:
                samples.append((path, idx))
    return samples


def load_image(path: Path) -> np.ndarray:
    """
    Load image with OpenCV and convert BGR->RGB for 3-channel inputs. Grayscale kept as-is.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    # Convert BGR->RGB for 3-channel images
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def evaluate(vit: ViTInference,
             samples: List[Tuple[Path, int]],
             device: torch.device,
             batch_size: int = 1) -> Dict:
    """
    Evaluate using repo's ViTInference preprocessing and model.
    """
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0

    num_classes = len(vit.class_names)
    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_counts = np.zeros(num_classes, dtype=np.int64)
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    misclassified: List[Tuple[str, int, int, float]] = []

    vit.model.eval()
    with torch.no_grad():
        # Process one-by-one to mirror manual checks
        for i, (path, label_idx) in enumerate(samples):
            image = load_image(path)

            # Repo-consistent preprocessing: returns [1,3,H,W] already on device
            inp: torch.Tensor = vit.preprocess_image(image)  # DO NOT unsqueeze again

            target = torch.tensor([label_idx], dtype=torch.long, device=device)

            logits = vit.model(inp)
            loss = criterion(logits, target)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            pred_idx = int(pred.item())
            confidence = float(conf.item())

            total += 1
            if pred_idx == label_idx:
                correct += 1
                class_correct[label_idx] += 1
            else:
                misclassified.append((str(path), label_idx, pred_idx, confidence))
            class_counts[label_idx] += 1
            conf_mat[label_idx, pred_idx] += 1

            if (i + 1) % 500 == 0:
                run_acc = 100.0 * correct / max(1, total)
                print(f"Processed {i + 1} / {len(samples)} | Running Acc: {run_acc:.2f}%")

    avg_loss = total_loss / max(1, total)
    acc = 100.0 * correct / max(1, total)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "total_samples": total,
        "total_correct": correct,
        "class_correct": class_correct,
        "class_counts": class_counts,
        "confusion_matrix": conf_mat,
        "misclassified": misclassified,
    }


def maybe_save_misclassifications(csv_path: str,
                                  miscls: List[Tuple[str, int, int, float]],
                                  idx_to_class: Dict[int, str]):
    import csv
    out_dir = Path(csv_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_idx", "true_label", "pred_idx", "pred_label", "pred_confidence"])
        for path, t, p, conf in miscls:
            w.writerow([path, t, idx_to_class.get(t, str(t)), p, idx_to_class.get(p, str(p)), f"{conf:.6f}"])
    print(f"Saved misclassifications to: {csv_path} (count={len(miscls)})")


def main():
    args = parse_args()

    # Device selection
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    data_dir = Path(args.data_dir)
    checkpoint = args.checkpoint or str(settings.CHECKPOINTS_VIT / settings.VIT_CHECKPOINT_NAME)

    print("Configuration")
    print(f"- Data dir     : {data_dir}")
    print(f"- Checkpoint   : {checkpoint}")
    print(f"- Device       : {device.type}")
    print(f"- Batch size   : {args.batch_size}")

    # Init repo's inference (loads model via get_vit_model and applies ImageNet norm)
    vit = ViTInference(checkpoint_path=checkpoint, device=device.type)
    class_names = vit.class_names  # from settings.VIT_CLASS_NAMES
    print(f"Detected classes (repo settings): {class_names}")

    # Collect samples
    samples = walk_image_folder(data_dir, class_names)
    if args.limit is not None and args.limit > 0:
        samples = samples[:args.limit]
        print(f"Limiting evaluation to first {len(samples)} images.")

    # Evaluate
    results = evaluate(vit, samples, device, batch_size=args.batch_size)

    # Report
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    print("\n=== Evaluation Summary (repo-consistent) ===")
    print(f"Total samples : {results['total_samples']}")
    print(f"Total correct : {results['total_correct']}")
    print(f"Loss          : {results['loss']:.6f}")
    print(f"Accuracy      : {results['accuracy']:.2f}%")

    print("\nPer-class accuracy:")
    for i, name in enumerate(class_names):
        cnt = results["class_counts"][i]
        cor = results["class_correct"][i]
        acc_i = (100.0 * cor / cnt) if cnt > 0 else 0.0
        print(f"  [{i}] {name:<12} {cor}/{cnt}  ({acc_i:.2f}%)")

    print("\nConfusion matrix [true rows x pred cols]:")
    cm = results["confusion_matrix"]
    header = "pred-> " + " ".join([f"{i:>6}" for i in range(len(class_names))])
    print(header)
    for r in range(len(class_names)):
        row = " ".join([f"{cm[r, c]:>6d}" for c in range(len(class_names))])
        print(f"true {r:>2}: {row}")

    # Save misclassifications if requested
    if args.save_csv:
        maybe_save_misclassifications(args.save_csv, results["misclassified"], idx_to_class)


if __name__ == "__main__":
    main()