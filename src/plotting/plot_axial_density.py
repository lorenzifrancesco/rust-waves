import numpy as np
import h5py
import matplotlib.pyplot as plt
import toml
from plot_widths import width_from_wavefunction

def init_plotting():
  fig, ax = plt.subplots(1, 1, figsize=(4, 2.2), dpi=600)
  return fig, ax

def plot_1d_axial_density(fig, ax, name_list = ["psi_3d", "psi_3d_2"]):
  # Load the 3D array from the HDF5 file
  interpolation = "none"
  for name in name_list:
    file_name = "results/"+name+".h5"
    field_key = "psi_squared"
    l_x_key = "l"

    par = toml.load("input/params.toml")
    params = par["numerics"]
    with h5py.File(file_name, "r") as file:
        assert l_x_key in file, f"Key 'l_x' is missing, this does not seem to be a 3D dataset."
        field = file[field_key][:]
        l_x =   file[l_x_key][()]
        print(f"Loaded dataset with shape: {field.shape}")

    ax.plot(l_x, field, lw=0.5, linestyle="--")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$|f|^2$")
    plt.tight_layout()

    # Save the plot as a PNG file
    output_file = f"media/axial_density.png"
    plt.savefig(output_file, dpi=900)
    print(f"Saved 3D projections as '{output_file}'.")
    return fig


def plot_3d_axial_density(fig, ax, name_list = ["psi_1d"]):
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

    # Calculate projections
    x = np.sum(field, axis=(1, 2)) * dy * dz

    vmin, vmax = np.nanmin(x), np.nanmax(x)
    vmax = min(2.0, abs(vmax))

    ax.plot(l_x, x, lw=0.5, linestyle="-")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$|f|^2$")
    plt.tight_layout()

    # Save the plot as a PNG file
    output_file = f"media/axial_density.png"
    plt.savefig(output_file, dpi=900)
    print(f"Saved 3D projections as '{output_file}'.")
    return fig, ax
  
if __name__ == "__main__":
  
  # fig, ax = init_plotting()
  # num = 9
  # plot_1d_axial_density(fig, ax, name_list=[f"width-check_1d"])
  # plot_3d_axial_density(fig, ax, name_list=[f"width-check_3d"])
  # plt.grid(True, which='major', linestyle='-', color='black', alpha=0.4)
  # plt.grid(True, which='minor', linestyle=':', color='gray', alpha=0.3)
  # plt.xlim([-3, 3])
  # plt.minorticks_on()  
  # plt.savefig("media/axial_density.png", dpi=900)
  # width_from_wavefunction(f"width-check", dimensions=1)
  # width_from_wavefunction(f"width-check", dimensions=3)
  # exit()
  
  fig, ax = init_plotting()
  num = 8
  plot_1d_axial_density(fig, ax, name_list=[f"idx-{num}_1d"])
  plot_3d_axial_density(fig, ax, name_list=[f"idx-{num}_3d"])
  plt.grid(True, which='major', linestyle='-', color='black', lw=0.4, alpha=0.4)
  plt.grid(True, which='minor', linestyle=':', color='gray' , lw=0.4, alpha=0.3)
  plt.xlim([-5, 5])
  plt.minorticks_on()  
  plt.savefig("media/axial_density.png", dpi=900)
  print(width_from_wavefunction(f"idx-{num}", dimensions=1))
  print(width_from_wavefunction(f"idx-{num}", dimensions=3))
  
  exit()
  # SPR consistency check
  cnt = 0
  fig, ax = init_plotting()
  for name in ["check_1d", "check-np_1d"]:
    plot_1d_axial_density(fig, ax, name_list=[name])
    
  for name in ["check_3d"]:
    plot_3d_axial_density(fig, ax, name_list=[name])
    
  plt.savefig("media/axial_density.png", dpi=900)
