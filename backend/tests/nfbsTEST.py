import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Correct NFBS Paths (Windows)
# ----------------------------
image_path = r"X:\file\FAST_API\BTSC-UNet-ViT\backend\dataset\NFBS_Dataset\A00028185\sub-A00028185_ses-NFB3_T1w.nii.gz"
mask_path  = r"X:\file\FAST_API\BTSC-UNet-ViT\backend\dataset\NFBS_Dataset\A00028185\sub-A00028185_ses-NFB3_T1w_brainmask.nii.gz"

# ----------------------------
# Load NIfTI files
# ----------------------------
image_nii = nib.load(image_path)
mask_nii  = nib.load(mask_path)

image = image_nii.get_fdata()
mask  = mask_nii.get_fdata()

print("Image shape:", image.shape)
print("Mask shape :", mask.shape)

# ----------------------------
# Select middle axial slice
# ----------------------------
slice_idx = image.shape[2] // 2

image_slice = image[:, :, slice_idx]
mask_slice  = mask[:, :, slice_idx]

# ----------------------------
# Plot side-by-side
# ----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_slice.T, cmap="gray", origin="lower")
plt.title("NFBS T1 MRI")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask_slice.T, cmap="gray", origin="lower")
plt.title("Ground Truth Brain Mask")
plt.axis("off")

plt.tight_layout()
plt.show()
