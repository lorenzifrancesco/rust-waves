import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_projections(name_list = ["psi_3d", "psi_3d_2"]):
  # Load the 3D array from the HDF5 file
  for name in name_list:
    file_name = "results/"+name+".h5"
    field_key = "field"
    l_x_key = "l_x"
    l_y_key = "l_y"
    l_z_key = "l_z"

    with h5py.File(file_name, "r") as file:
        assert l_x_key in file, f"Key 'l_x' is missing, this does not seem to be a 3D dataset."
        field = file[field_key][:]
        l_x = file[l_x_key][()]
        l_y = file[l_y_key][()]
        l_z = file[l_z_key][()]
        print(f"Loaded dataset with shape: {field.shape}")

    # check the normalization
    dx = l_x[1]-l_x[0] 
    dy = l_y[1]-l_y[0] 
    dz = l_z[1]-l_z[0] 
    print(f"Normalization: {np.sum(field)/(dx*dy*dz)}")
    
    # Calculate projections
    projection_xy = np.sum(field, axis=2)
    projection_xz = np.sum(field, axis=1)
    projection_yz = np.sum(field, axis=0)

    # Create a figure with three vertically stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(3, 9))  # Adjust size for better aspect ratio

    # Plot the XY projection
    axes[0].imshow(projection_xy, cmap="viridis", aspect="auto")
    axes[0].set_title("XY Projection (Summed along Z)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    # Plot the XZ projection
    axes[1].imshow(projection_xz, cmap="viridis", aspect="auto")
    axes[1].set_title("XZ Projection (Summed along Y)")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")

    # Plot the YZ projection
    axes[2].imshow(projection_yz, cmap="viridis", aspect="auto")
    axes[2].set_title("YZ Projection (Summed along X)")
    axes[2].set_xlabel("Y")
    axes[2].set_ylabel("Z")

    # Adjust layout for readability
    plt.tight_layout()

    # Save the plot as a PNG file
    output_file = "media/"+name+".png"
    plt.savefig(output_file, dpi=900)
    print(f"Saved 3D projections as '{output_file}'.")