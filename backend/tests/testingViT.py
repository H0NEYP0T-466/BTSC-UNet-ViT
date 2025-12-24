#!/usr/bin/env python3
"""
Evaluate a trained ViT classifier on a separate 4-class folder dataset.

- Loads a ViT model checkpoint (e.g., vit_best.pth from your training).
- Walks a dataset laid out as ImageFolder:
    data_dir/
      notumor/
      glioma/
      meningioma/
      pituitary/
- Feeds images one-by-one (batch_size=1) through the model
- Computes overall loss and accuracy, plus per-class metrics and a confusion matrix
- Optionally writes misclassifications to CSV

Usage:
  python test_vit_on_folder.py \
    --data-dir /path/to/3k_dataset \
    --checkpoint /content/checkpoints/vit_best.pth \
    --model-name vit_base_patch16_224 \
    --image-size 224 \
    --device cuda \
    --save-csv misclassified.csv

Notes:
- Make sure the directory class names and ordering match what you trained on.
  If your training used ImageFolder with those 4 subfolders, this should line up.
- No data augmentation is applied during evaluation; only resize + normalization.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import timm
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ViT on an ImageFolder dataset (one-by-one).")
    p.add_argument("--data-dir", type=str, required=True, help="Path to ImageFolder dataset with 4 subfolders.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth).")
    p.add_argument("--model-name", type=str, default="vit_base_patch16_224", help="timm model name.")
    p.add_argument("--num-classes", type=int, default=4, help="Number of classes (default: 4).")
    p.add_argument("--image-size", type=int, default=224, help="Image size (short side) for ViT.")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size. Use 1 to process one-by-one as requested.")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"], help="Force device (default: auto).")
    p.add_argument("--save-csv", type=str, default=None, help="Optional: save misclassifications to CSV at this path.")
    p.add_argument("--class-map-json", type=str, default=None,
                   help="Optional: path to JSON mapping {class_name: index} used in training, to verify alignment.")
    p.add_argument("--limit", type=int, default=None, help="Optional: limit number of images for a quick check.")
    return p.parse_args()


class ImageFolderWithPaths(datasets.ImageFolder):
    """Same as ImageFolder but returns (img, label, path)."""
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.samples[index][0]
        return img, label, path


def build_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(model_name: str, num_classes: int, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    import timm
    import torch
    import torch.nn as nn

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Extract the actual state dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Normalize key prefixes: module./model./backbone. -> remove
    normalized = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "model.", "backbone."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        # Optional: common alternative head names -> map to timm's 'head'
        if nk.startswith("classifier."):
            nk = "head." + nk[len("classifier."):]
        if nk.startswith("fc."):
            nk = "head." + nk[len("fc."):]
        normalized[nk] = v

    missing, unexpected = model.load_state_dict(normalized, strict=False)
    if missing or unexpected:
        print("Warning when loading checkpoint after prefix normalization:")
        if missing:
            print(" - Missing keys:", missing)
        if unexpected:
            print(" - Unexpected keys:", unexpected)

    model.to(device)
    model.eval()
    return model

def evaluate(model: nn.Module,
             loader: torch.utils.data.DataLoader,
             device: torch.device,
             num_classes: int) -> Dict:
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    # Per-class stats
    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_counts = np.zeros(num_classes, dtype=np.int64)
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)  # [true, pred]

    misclassified = []  # (path, true_idx, pred_idx, prob_pred)

    with torch.no_grad():
        for i, (images, labels, paths) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)

            total_samples += images.size(0)
            total_correct += (preds == labels).sum().item()

            # Update per-class and confusion matrix
            for t, p, confv, path in zip(labels.cpu().numpy(), preds.cpu().numpy(), conf.cpu().numpy(), paths):
                class_counts[t] += 1
                class_correct[t] += int(t == p)
                conf_mat[t, p] += 1
                if t != p:
                    misclassified.append((path, int(t), int(p), float(confv)))

            # Simple progress print every 500 images
            if (i + 1) % 500 == 0:
                running_acc = 100.0 * total_correct / total_samples
                print(f"Processed {total_samples} images | Running Acc: {running_acc:.2f}%")

    avg_loss = total_loss / max(1, total_samples)
    acc = 100.0 * total_correct / max(1, total_samples)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "total_samples": total_samples,
        "total_correct": total_correct,
        "class_correct": class_correct,
        "class_counts": class_counts,
        "confusion_matrix": conf_mat,
        "misclassified": misclassified,
    }


def maybe_save_misclassifications(csv_path: str,
                                  miscls: List[Tuple[str, int, int, float]],
                                  idx_to_class: Dict[int, str]):
    import csv
    csv_dir = Path(csv_path).parent
    csv_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_idx", "true_label", "pred_idx", "pred_label", "pred_confidence"])
        for path, t, p, conf in miscls:
            writer.writerow([path, t, idx_to_class.get(t, str(t)), p, idx_to_class.get(p, str(p)), f"{conf:.6f}"])
    print(f"Saved misclassifications to: {csv_path} (count={len(miscls)})")


def main():
    args = parse_args()

    # Select device
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("Configuration")
    print(f"- Data dir     : {args.data_dir}")
    print(f"- Checkpoint   : {args.checkpoint}")
    print(f"- Model        : {args.model_name}")
    print(f"- Num classes  : {args.num_classes}")
    print(f"- Image size   : {args.image_size}")
    print(f"- Batch size   : {args.batch_size}")
    print(f"- Device       : {device.type}")
    print(f"- Save CSV     : {args.save_csv}")

    # Transforms and dataset
    transform = build_transforms(args.image_size)
    dataset = ImageFolderWithPaths(root=args.data_dir, transform=transform)

    # Optional: verify class index mapping alignment
    if args.class_map_json and os.path.isfile(args.class_map_json):
        with open(args.class_map_json, "r") as f:
            train_map = json.load(f)  # {class_name: index}
        # Show differences, if any
        eval_map = dataset.class_to_idx
        print("\nClass index alignment (training vs eval):")
        for cname in sorted(set(train_map.keys()).union(eval_map.keys())):
            print(f"  {cname:<15} train={train_map.get(cname, 'NA')}, eval={eval_map.get(cname, 'NA')}")
        # Warn if mismatch
        if any(train_map.get(k) != eval_map.get(k) for k in train_map.keys()):
            print("WARNING: Class-to-index mapping differs between training and evaluation. "
                  "This can invalidate results. Consider reindexing or matching folder names.")
    else:
        print("\nDetected classes (eval):", dataset.classes)

    if args.num_classes != len(dataset.classes):
        print(f"WARNING: --num-classes={args.num_classes} but dataset has {len(dataset.classes)} classes: {dataset.classes}")

    # Optional limit
    if args.limit is not None and args.limit > 0:
        # Create a subset by truncating samples list deterministically
        from torch.utils.data import Subset
        indices = list(range(min(args.limit, len(dataset))))
        dataset = Subset(dataset, indices)
        print(f"Limiting evaluation to first {len(indices)} images.")

    # DataLoader (batch_size=1 per requirement)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = load_model(args.model_name, args.num_classes, args.checkpoint, device)

    # Evaluate
    results = evaluate(model, loader, device, num_classes=args.num_classes)

    # Report
    idx_to_class = {v: k for k, v in (getattr(dataset, "dataset", dataset).class_to_idx).items()}

    print("\n=== Evaluation Summary ===")
    print(f"Total samples : {results['total_samples']}")
    print(f"Total correct : {results['total_correct']}")
    print(f"Loss          : {results['loss']:.6f}")
    print(f"Accuracy      : {results['accuracy']:.2f}%")

    print("\nPer-class accuracy:")
    for i in range(args.num_classes):
        cnt = results["class_counts"][i]
        cor = results["class_correct"][i]
        acc = (100.0 * cor / cnt) if cnt > 0 else 0.0
        print(f"  [{i}] {idx_to_class.get(i, str(i)):<12} {cor}/{cnt}  ({acc:.2f}%)")

    print("\nConfusion matrix [true rows x pred cols]:")
    # Pretty-print small confusion matrix
    cm = results["confusion_matrix"]
    header = "pred-> " + " ".join([f"{i:>6}" for i in range(args.num_classes)])
    print(header)
    for r in range(args.num_classes):
        row = " ".join([f"{cm[r, c]:>6d}" for c in range(args.num_classes)])
        print(f"true {r:>2}: {row}")

    # Save misclassifications if requested
    if args.save_csv:
        maybe_save_misclassifications(args.save_csv, results["misclassified"], idx_to_class)


if __name__ == "__main__":
    main()