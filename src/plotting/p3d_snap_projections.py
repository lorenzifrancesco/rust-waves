import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio
import toml 
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
    
def movie(filename):
    t, frames = load_hdf5_data(filename)
    create_gif(t, frames, "media/3d_movie.gif")
    
def load_hdf5_data(filename):
  """Load HDF5 file and extract projections."""
  with h5py.File(filename, "r") as f:
      # Read time values
      t = np.array(f["t"])

      # Load projection frames
      frames = []
      for i in range(len(t)):  # Assuming `t` defines the number of frames
          frame_group = f[f"movie/frame_{i}"]
          xz = np.array(frame_group["xz"])
          yz = np.array(frame_group["yz"])
          frames.append((xz, yz))
      # print(">>>>", len(frames))
      # print(">>", len(frames[-1]))
      # print("->", frames[-1][0])
  return t, frames

def create_gif(t, frames, output_filename="movie.gif"):
  """Generate and save a GIF from the projection heatmaps."""
  images = []
  
  fig, axes = plt.subplots(1, 2, figsize=(6, 2.3), width_ratios=[3, 1])
  par = toml.load("input/params.toml")
  params = par["numerics"]
  all_data = np.array([np.concatenate((xz.flatten(), yz.flatten())) for xz, yz in frames[:95]])
  vmin, vmax = np.nanmin(all_data), np.nanmax(all_data)
  vmax = min(2.0, abs(vmax))
  for i, (xz, yz) in enumerate(frames):
      print(f"plotting frame {i:>10d}")
      axes[0].clear()
      axes[1].clear()
      
      im1 = axes[0].imshow(xz.T,
                           aspect="auto", 
                           origin="lower", 
                           cmap="gist_ncar", 
                           interpolation="bicubic",
                           extent=[-params["l"]/2, params["l"]/2, -params["l_z"]/2, params["l_z"]/2],
                           vmin=vmin, vmax=vmax)
      axes[0].set_aspect(2.2)
      im2 = axes[1].imshow(yz.T,
                           aspect="auto", 
                           origin="lower", 
                           cmap="gist_ncar", 
                           interpolation="bicubic",
                           extent=[-params["l_y"]/2, params["l_y"]/2, -params["l_z"]/2, params["l_z"]/2], 
                           vmin=vmin, vmax=vmax)
      
      axes[0].set_title(f"XZ (t = {t[i]:.2f})")
      axes[1].set_title(f"YZ (t = {t[i]:.2f})")
      axes[0].set_xlabel(r"$x$")
      axes[0].set_ylabel(r"$z$")
      axes[1].set_xlabel(r"$y$")
      axes[1].set_ylabel(r"$z$")
      
      # plt.colorbar(im1, ax=axes[0])
      # plt.colorbar(im2, ax=axes[1])

      plt.tight_layout()
      
      # Save current figure as an image in memory
      fig.canvas.draw()
      image = np.array(fig.canvas.renderer.buffer_rgba())
      images.append(image)

  plt.close(fig)  # Close figure to free memory
  
  # Save images as GIF
  imageio.mimsave(output_filename, images, fps=10, dpi=(1200, 900))
  print(f"Saved GIF: {output_filename}")
  
if __name__ == "__main__":
  movie("results/base_3d.h5")