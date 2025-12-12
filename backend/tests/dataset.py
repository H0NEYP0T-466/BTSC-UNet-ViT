"""
Deep inspection of H5 mask data.
"""
import h5py
import numpy as np
from pathlib import Path
import sys

def inspect_h5_files(dataset_path, num_samples=10):
    """Deeply inspect H5 file contents."""
    dataset_path = Path(dataset_path)
    h5_files = list(dataset_path.glob('*.h5'))[: num_samples]
    
    print(f"📂 Dataset: {dataset_path}")
    print(f"   Total files: {len(list(dataset_path.glob('*. h5')))}")
    print("\n" + "="*80)
    print("DETAILED INSPECTION")
    print("="*80)
    
    for i, h5_file in enumerate(h5_files, 1):
        print(f"\n[{i}] {h5_file.name}")
        print("-" * 80)
        
        try: 
            with h5py.File(h5_file, 'r') as f:
                print(f"Keys in file: {list(f.keys())}")
                
                for key in f.keys():
                    data = np.array(f[key])
                    print(f"\n  '{key}':")
                    print(f"    Shape: {data.shape}")
                    print(f"    Dtype: {data.dtype}")
                    print(f"    Min:  {data.min():.6f}")
                    print(f"    Max: {data.max():.6f}")
                    print(f"    Mean: {data.mean():.6f}")
                    print(f"    Unique values: {len(np.unique(data))}")
                    
                    # Show value distribution
                    unique_vals = np.unique(data)
                    if len(unique_vals) <= 10:
                        print(f"    Values: {unique_vals}")
                    else:
                        print(f"    Value range: [{unique_vals[0]}, .. ., {unique_vals[-1]}]")
                    
                    # Check if it's binary-ish
                    if 'mask' in key. lower() or 'label' in key.lower():
                        zeros = (data == 0).sum()
                        ones = (data == 1).sum()
                        total = data.size
                        
                        print(f"    Zeros: {zeros: ,} ({zeros/total*100:.2f}%)")
                        print(f"    Ones: {ones:,} ({ones/total*100:.2f}%)")
                        
                        if data.max() > 1:
                            mid = (data > 0) & (data < 255)
                            high = (data >= 255).sum() if data.max() >= 255 else (data > 200).sum()
                            print(f"    Mid values (1-254): {mid.sum():,}")
                            print(f"    High values (>=200): {high:,} ({high/total*100:.2f}%)")
        
        except Exception as e: 
            print(f"  ❌ Error:  {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    if len(sys. argv) > 1:
        path = sys.argv[1]
    else:
        path = r"X:\file\FAST_API\BTSC-UNet-ViT\backend\dataset\UNet_Dataset"
    
    inspect_h5_files(path, num_samples=5)