import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = r"X:\NFBS_Dataset\NFBS_Dataset"

subjects = sorted(os.listdir(DATASET_DIR))
subjects = [s for s in subjects if os.path.isdir(os.path.join(DATASET_DIR, s))]

print(f"Total subjects found: {len(subjects)}\n")

stats = []

for sid in subjects:
    subj_path = os.path.join(DATASET_DIR, sid)

    t1 = [f for f in os.listdir(subj_path) if f.endswith("_T1w.nii.gz")]
    mask = [f for f in os.listdir(subj_path) if f.endswith("_brainmask.nii.gz")]

    if len(t1) != 1 or len(mask) != 1:
        print(f"[WARNING] Missing files in {sid}")
        continue

    t1_img = nib.load(os.path.join(subj_path, t1[0]))
    mask_img = nib.load(os.path.join(subj_path, mask[0]))

    t1_data = t1_img.get_fdata()
    mask_data = mask_img.get_fdata()

    stats.append({
        "subject": sid,
        "shape": t1_data.shape,
        "voxel_size": t1_img.header.get_zooms(),
        "intensity_min": np.min(t1_data),
        "intensity_max": np.max(t1_data),
        "mask_unique": np.unique(mask_data)
    })

# Print summary
print("\n=== DATASET SUMMARY ===")
print(f"Subjects analyzed: {len(stats)}")
print(f"Image shape (example): {stats[0]['shape']}")
print(f"Voxel size (example): {stats[0]['voxel_size']}")
print(f"Mask values: {stats[0]['mask_unique']}")

# Visual check (middle slice)
example = stats[0]["subject"]
example_path = os.path.join(DATASET_DIR, example)

t1 = nib.load(os.path.join(example_path,
       [f for f in os.listdir(example_path) if f.endswith("_T1w.nii.gz")][0])).get_fdata()
mask = nib.load(os.path.join(example_path,
       [f for f in os.listdir(example_path) if f.endswith("_brainmask.nii.gz")][0])).get_fdata()

z = t1.shape[2] // 2

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(t1[:,:,z], cmap="gray")
plt.title("Raw T1")

plt.subplot(1,2,2)
plt.imshow(mask[:,:,z], cmap="gray")
plt.title("Brain Mask")
plt.show()
