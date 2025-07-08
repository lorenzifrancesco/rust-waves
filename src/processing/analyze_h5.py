import h5py
import sys

def inspect_h5_file(filename):
    """Inspect the contents of an HDF5 file"""
    try:
        with h5py.File(filename, 'r') as f:
            print(f"=== Contents of {filename} ===")
            
            def print_structure(name, obj):
                print(f"/{name}: {type(obj).__name__}")
                if isinstance(obj, h5py.Dataset):
                    print(f"  Shape: {obj.shape}")
                    print(f"  Dtype: {obj.dtype}")
            
            f.visititems(print_structure)
            
            print("\n=== Root level keys ===")
            for key in f.keys():
                print(f"  {key}: {f[key].shape if hasattr(f[key], 'shape') else 'Group'}")
                
    except Exception as e:
        print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    # Check the file that caused the error
    filename = "results/pre-quench_3d.h5"  # Adjust path if needed
    inspect_h5_file(filename)