import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio
import toml
 
def plot_projections(name_list = ["psi_3d", "psi_3d_2"], i = -1):
  # Load the 3D array from the HDF5 file
  interpolation = "none"
  for name in name_list:
    file_name = "results/"+name+".h5"
    field_key = "psi_squared"
    l_x_key = "l_x"
    l_y_key = "l_y"
    l_z_key = "l_z"

    par = toml.load("input/params.toml")
    params = par["numerics"]
    with h5py.File(file_name, "r") as file:
        assert l_x_key in file, f"Key 'l_x' is missing, this does not seem to be a 3D dataset."
        field = file[field_key][:]
        l_x =   file[l_x_key][()]
        l_y =   file[l_y_key][()]
        l_z =   file[l_z_key][()]
        print(f"Loaded dataset with shape: {field.shape}")

    # check the normalization
    dx = l_x[1]-l_x[0] 
    dy = l_y[1]-l_y[0] 
    dz = l_z[1]-l_z[0] 
    print(f"Normalization: {np.sum(field)*(dx*dy*dz)}")
    
    # Calculate projections
    xy = np.sum(field, axis=2)
    xz = np.sum(field, axis=1)
    yz = np.sum(field, axis=0)

    all_data = np.array([np.concatenate((xz.flatten(), yz.flatten()))])
    vmin, vmax = np.nanmin(all_data), np.nanmax(all_data)
    vmax = min(2.0, abs(vmax))
    # Create a figure with three vertically stacked subplots

    
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.3), width_ratios=[3, 1])
    im1 = axes[0].imshow(xz.T,
                          aspect="auto", 
                          origin="lower", 
                          cmap="nipy_spectral", 
                          interpolation=interpolation,
                          extent=[-params["l"]/2, params["l"]/2, -params["l_z"]/2, params["l_z"]/2],
                          vmin=vmin, vmax=vmax)
    axes[0].set_aspect(0.5)
    im2 = axes[1].imshow(yz.T,
                          aspect="auto", 
                          origin="lower", 
                          cmap="nipy_spectral", 
                          interpolation=interpolation,
                          extent=[-params["l_y"]/2, params["l_y"]/2, -params["l_z"]/2, params["l_z"]/2], 
                          vmin=vmin, vmax=vmax)
    
    axes[0].set_title(f"XZ")
    axes[1].set_title(f"YZ")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$z$")
    axes[1].set_xlabel(r"$y$")
    axes[1].set_ylabel(r"$z$")
 
    # plt.colorbar(im1, ax=axes[0])
    # plt.colorbar(im2, ax=axes[1])
    plt.tight_layout()

    # Save the plot as a PNG file
    output_file = f"media/idx-{i}_heatmap_3d.png"
    plt.savefig(output_file, dpi=900)
    print(f"Saved 3D projections as '{output_file}'.")



def movie(name, i = -1):
    path = "results/"+name+".h5"
    t, frames = load_hdf5_data(path)
    output_file = f"media/idx-{i}_heatmap_3d.gif"
    create_gif(t, frames, output_file)

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
  
  # two figures
  # fig, axes = plt.subplots(1, 2, figsize=(6, 2.2), width_ratios=[4, 1.5], dpi=600)
  # single figure
  fig, ax = plt.subplots(figsize=(6, 2.2), dpi=600)
  axes = [ax]
  par = toml.load("input/params.toml")
  params = par["numerics"]
  all_data = np.array([np.concatenate((xz.flatten(), yz.flatten())) for xz, yz in frames[:95]])
  vmin, vmax = np.nanmin(all_data), np.nanmax(all_data)
  vmax = min(100.0, abs(vmax))
  cmap = "nipy_spectral"
  interpolation = "bicubic"
  
  print()
  for i, (xz, yz) in enumerate(frames):
      print(f"\rplotting frame {i:>10d}", end="")
      axes[0].clear()
      # axes[1].clear()
      im1 = axes[0].imshow(xz.T,
                           aspect="auto", 
                           origin="lower", 
                           cmap=cmap, 
                           interpolation=interpolation,
                           extent=[-params["l"]/2, params["l"]/2, -params["l_z"]/2, params["l_z"]/2],
                           vmin=vmin, vmax=vmax)
      axes[0].set_aspect(0.5)
      # im2 = axes[1].imshow(yz.T,
      #                      aspect="equal", 
      #                      origin="lower", 
      #                      cmap=cmap, 
      #                      interpolation=interpolation,
      #                      extent=[-params["l_y"]/2, params["l_y"]/2, -params["l_z"]/2, params["l_z"]/2], 
      #                      vmin=vmin, vmax=vmax)
      
      axes[0].set_title(rf"$xz \; (t = {t[i]:.1f} t_\perp)$", fontsize=8)
      # axes[1].set_title(rf"$yz \; (t = {t[i]:.1f} t_\perp)$", fontsize=8)
      axes[0].set_xlabel(r"$x$")
      axes[0].set_ylabel(r"$z$")
      # axes[1].set_xlabel(r"$y$")
      # axes[1].set_ylabel(r"$z$")
      
      if i==0:
        plt.colorbar(im1, ax=axes[0])
      #   plt.colorbar(im1, ax=axes[1])

      plt.tight_layout()
      
      # Save current figure as an image in memory
      fig.canvas.draw()
      image = np.array(fig.canvas.renderer.buffer_rgba())
      images.append(image)

  plt.close(fig)  # Close figure to free memory
  
  # Save images as GIF
  print(". Adding more last frames...")
  images.extend([images[-1]] * 20)
  imageio.mimsave(output_filename, images, fps=10, loop=0)
  print(f"Saved GIF: {output_filename}")
  
if __name__ == "__main__":
  movie("dyn_linear_3d", -1)