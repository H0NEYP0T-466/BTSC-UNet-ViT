"""
Test UNet inference to ensure proper tumor visualization.
Tests that tiny tumors (0.17%) are properly visible in predictions.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.models.unet.model import get_unet_model
from app.models.unet.utils import visualize_tumor_mask


def test_inference(checkpoint_path: str, sample_h5: str, output_dir: str):
    """Test inference on a sample .h5 file."""
    print("="*80)
    print("UNET INFERENCE TEST")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sample data
    print(f"\n📁 Loading sample data: {sample_h5}")
    with h5py.File(sample_h5, 'r') as f:
        image = f['image'][...]
        mask_gt = f['mask'][...]
    
    print(f"   Image shape: {image.shape}")
    print(f"   Mask shape: {mask_gt.shape}")
    
    # Process image to (C, H, W)
    if len(image.shape) == 3 and image.shape[-1] == 4:
        image = np.transpose(image, (2, 0, 1))
    
    # Process ground truth mask
    mask_gt_binary = np.max(mask_gt, axis=-1)
    tumor_ratio_gt = (mask_gt_binary > 0).sum() / mask_gt_binary.size * 100
    print(f"   Ground truth tumor ratio: {tumor_ratio_gt:.3f}%")
    
    # Normalize image per channel
    image_normalized = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        channel = image[c]
        c_min, c_max = channel.min(), channel.max()
        if c_max > c_min:
            image_normalized[c] = (channel - c_min) / (c_max - c_min)
    
    # Create model
    print(f"\n🧠 Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    model = get_unet_model(in_channels=4, out_channels=1)
    model = model.to(device)
    model.eval()
    
    # Load checkpoint if available
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        print(f"   Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"   ✅ Checkpoint loaded")
    else:
        print(f"   ⚠️  Checkpoint not found, using untrained model")
    
    # Prepare input
    print(f"\n🔮 Running inference...")
    input_tensor = torch.from_numpy(image_normalized).unsqueeze(0).float().to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Get prediction
    prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]
    pred_binary = (prob_map > 0.5).astype(np.float32)
    
    tumor_ratio_pred = pred_binary.sum() / pred_binary.size * 100
    print(f"   Predicted tumor ratio: {tumor_ratio_pred:.3f}%")
    
    # Visualize results
    print(f"\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Ground Truth
    # Image (channel 0)
    axes[0, 0].imshow(image[0], cmap='gray')
    axes[0, 0].set_title('Input Image (Channel 0)')
    axes[0, 0].axis('off')
    
    # Ground truth mask (hot colormap for visibility)
    im1 = axes[0, 1].imshow(mask_gt_binary, cmap='hot', interpolation='nearest')
    axes[0, 1].set_title(f'Ground Truth Mask\n(Tumor: {tumor_ratio_gt:.3f}%)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # GT Overlay
    img_norm = (image[0] - image[0].min()) / (image[0].max() - image[0].min() + 1e-8)
    img_rgb = np.stack([img_norm] * 3, axis=-1)
    mask_colored = np.zeros_like(img_rgb)
    mask_colored[mask_gt_binary > 0] = [1, 0, 0]
    overlay_gt = np.clip(0.7 * img_rgb + 0.3 * mask_colored, 0, 1)
    axes[0, 2].imshow(overlay_gt)
    axes[0, 2].set_title('GT Overlay')
    axes[0, 2].axis('off')
    
    # Row 2: Predictions
    # Probability map
    im2 = axes[1, 0].imshow(prob_map, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    axes[1, 0].set_title('Probability Map')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # Binary prediction
    im3 = axes[1, 1].imshow(pred_binary, cmap='hot', interpolation='nearest')
    axes[1, 1].set_title(f'Binary Prediction\n(Tumor: {tumor_ratio_pred:.3f}%)')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    # Prediction overlay
    mask_colored_pred = np.zeros_like(img_rgb)
    mask_colored_pred[pred_binary > 0] = [0, 1, 0]  # Green for prediction
    overlay_pred = np.clip(0.7 * img_rgb + 0.3 * mask_colored_pred, 0, 1)
    axes[1, 2].imshow(overlay_pred)
    axes[1, 2].set_title('Prediction Overlay')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    viz_path = output_dir / 'inference_test_results.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Visualization saved: {viz_path}")
    plt.close()
    
    # Calculate Dice score
    intersection = (pred_binary * (mask_gt_binary > 0)).sum()
    dice = (2 * intersection + 1e-6) / (pred_binary.sum() + (mask_gt_binary > 0).sum() + 1e-6)
    
    print(f"\n📈 Metrics:")
    print(f"   Dice Score: {dice:.4f}")
    print(f"   Tumor pixels (GT): {(mask_gt_binary > 0).sum()}")
    print(f"   Tumor pixels (Pred): {pred_binary.sum()}")
    
    print("\n" + "="*80)
    print("✅ INFERENCE TEST COMPLETED")
    print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test UNet inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--sample", type=str, required=True,
                       help="Path to sample .h5 file")
    parser.add_argument("--output_dir", type=str, default="./inference_test_output",
                       help="Directory to save outputs")
    
    args = parser.parse_args()
    
    test_inference(args.checkpoint, args.sample, args.output_dir)


if __name__ == "__main__":
    main()
