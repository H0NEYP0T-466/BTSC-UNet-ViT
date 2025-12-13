import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

H5_PATH = "volume_194_slice_92.h5"  # <-- change to one real file

def analyze_h5(path):
    print("="*60)
    print(f"Analyzing file: {path}")
    print("="*60)

    with h5py.File(path, "r") as f:
        print("\n[1] ROOT KEYS")
        for k in f.keys():
            print(f"  - {k}")

        image = f["image"][:]
        mask  = f["mask"][:]

    print("\n[2] SHAPES")
    print("Image shape:", image.shape)
    print("Mask shape :", mask.shape)

    print("\n[3] DATA TYPES")
    print("Image dtype:", image.dtype)
    print("Mask dtype :", mask.dtype)

    print("\n[4] IMAGE STATISTICS (per channel)")
    if image.ndim == 3:
        for c in range(image.shape[-1]):
            ch = image[..., c]
            print(f" Channel {c}: min={ch.min():.3f}, max={ch.max():.3f}, mean={ch.mean():.3f}")
    else:
        print(" Unexpected image dimensions")

    print("\n[5] MASK UNIQUE VALUES")
    uniq = np.unique(mask)
    print(" Unique labels:", uniq)

    print("\n[6] MASK CLASS COUNTS")
    flat = mask.flatten()
    counts = Counter(flat)
    for k, v in counts.items():
        print(f" Label {k}: {v} pixels")

    print("\n[7] MASK TYPE INFERENCE")
    if set(uniq.tolist()) <= {0, 1}:
        print(" ✔ Binary segmentation mask")
    elif set(uniq.tolist()) <= {0, 1, 2, 3}:
        print(" ✔ Multi-class segmentation mask")
    else:
        print(" ⚠ Unknown / non-standard mask labels")

    print("\n[8] BRAIN vs TUMOR RATIO")
    tumor_ratio = np.mean(mask > 0)
    print(f" Tumor pixel ratio: {tumor_ratio*100:.2f}%")

    print("\n[9] VISUAL INSPECTION")
    plt.figure(figsize=(12, 4))

    for i in range(min(3, image.shape[-1])):
        plt.subplot(1, 4, i+1)
        plt.imshow(image[..., i], cmap="gray")
        plt.title(f"Image ch {i}")
        plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(mask, cmap="hot")
    plt.title("Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("\n[10] CONCLUSIONS")
    print("- This dataset is NOT original BraTS NIfTI")
    print("- It is preprocessed, sliced, and channel-packed")
    print("- Mask likely represents TUMOR vs NON-TUMOR")
    print("- Whole-brain predictions = label misinterpretation or loss misuse")
    print("="*60)
    mask_to_plot = np.max(mask, axis=-1)  # Combine channels if needed

    plt.figure(figsize=(6,6))
    plt.imshow(mask_to_plot, cmap='hot', interpolation='nearest')
    plt.title("Tumor mask (scaled for visibility)")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    analyze_h5(H5_PATH)
