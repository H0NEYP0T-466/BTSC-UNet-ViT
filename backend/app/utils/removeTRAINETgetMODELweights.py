import torch
import os

# 1. Define the path to your heavy model
# (Note the 'r' for raw string to fix backslashes)
checkpoint_path = r'X:\file\FAST_API\BTSC-UNet-ViT\backend\resources\checkpoints\vit\vit_best.pth'

# 2. Load the heavy 1GB checkpoint
print(f"Loading heavy model from: {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 3. Extract ONLY the weights
if 'model_state_dict' in checkpoint:
    weights = checkpoint['model_state_dict']
else:
    weights = checkpoint

# 4. Define where to save the new file (SAME FOLDER as the original)
# This gets the folder path: "X:\file\FAST_API\...\vit"
folder_path = os.path.dirname(checkpoint_path) 
# This joins the folder + new filename
save_path = os.path.join(folder_path, 'vit_base_production.pth')

# 5. Save it!
torch.save(weights, save_path)

print(f"✅ Success! Production model saved at:\n{save_path}")