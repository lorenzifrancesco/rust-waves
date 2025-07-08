import numpy as np
import h5py
import matplotlib.pyplot as plt
import toml
from plot_widths import width_from_wavefunction
from scipy.interpolate import interp1d
from projections_volumetric import load_hdf5_data
import pandas as pd

def init_plotting():
  fig, ax = plt.subplots(1, 1, figsize=(3.9, 2.2), dpi=600)
  return fig, ax


def plot_1d_axial_density(fig, ax, name_list = ["psi_1d", "psi_1d_2"], color="blue"):
  # Load the 3D array from the HDF5 file
  interpolation = "none"
  for name in name_list:
    file_name = "results/"+name+".h5"
    field_key = "psi_squared"
    l_x_key = "l"
    
    par = toml.load("input/_params.toml")    
    dl =par["physics"]["dl"]

    with h5py.File(file_name, "r") as file:
        assert l_x_key in file, f"Key 'l_x' is missing, this does not seem to be a 3D dataset."
        field = file[field_key][:]
        l_x =   file[l_x_key][()]
        print(f"Loaded dataset with shape: {field.shape}")

    ax.plot(l_x/dl, field, lw=1, linestyle="--", color=color)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$|f|^2$")
    plt.tight_layout()

    # Save the plot as a PNG file
    # output_file = f"media/axial_density.png"
    # plt.savefig(output_file, dpi=900)
    # print(f"Saved 1D as '{output_file}'.")
    return fig, ax


def plot_3d_axial_density(fig, ax, name_list = ["psi_1d"], color="blue", ls="-"):
  # Load the 3D array from the HDF5 file
  for name in name_list:
    file_name = "results/"+name+".h5"
    field_key = "psi_squared"
    l_x_key = "l_x"
    l_y_key = "l_y"
    l_z_key = "l_z"

    par = toml.load("input/_params.toml")    
    dl =par["physics"]["dl"]
    
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
    print(np.abs(np.sum(x) * dx - 1.0) < 1e-4)
    vmin, vmax = np.nanmin(x), np.nanmax(x)
    vmax = min(2.0, abs(vmax))

    ax.plot(l_x/dl, x, lw=1, linestyle=ls, color=color)
    ax.set_xlabel(r"$x \ [d_L]$")
    ax.set_ylabel(r"$|f|^2$")
    plt.tight_layout()

    # Save the plot as a PNG file
    # output_file = f"media/axial_density.png"
    # plt.savefig(output_file, dpi=900)
    # print(f"Saved 3D projections as '{output_file}'.")
    return fig, ax
  
 
def plot_3d_radial_density(fig, ax, name_list = ["psi_1d"], color="blue", ls="-"):
    # Load the 3D array from the HDF5 file
  for name in name_list:
    file_name = "results/"+name+".h5"
    field_key = "psi_squared"
    l_x_key = "l_x"
    l_y_key = "l_y"
    l_z_key = "l_z"

    par = toml.load("input/_params.toml")    
    params = par["numerics"]
    dl =par["physics"]["dl"]
    
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
    y = np.sum(field, axis=(0, 2)) * dx * dz
    # assert(np.abs(np.sum(y) * dy - 1.0) < 1e-9)

    vmin, vmax = np.nanmin(y), np.nanmax(y)
    vmax = min(2.0, abs(vmax))

    ax.plot(l_y, y, 
            lw=0.5, 
            linestyle=ls, 
            color="k", )
    l_y_resampled = np.linspace(l_y[0], l_y[-1], 1000)
    y_resampled = interp1d(l_y, y, "cubic")(l_y_resampled)
    ax.plot(l_y_resampled, y_resampled,
            lw=0.5, 
            linestyle=ls, 
            color=color, )
    ax.set_xlabel(r"$y \ [l_\perp]$")
    ax.set_ylabel(r"n(y)")
    plt.tight_layout()

    # Save the plot as a PNG file
    output_file = f"media/axial_density.png"
    plt.savefig(output_file, dpi=900)
    print(f"Saved 3D projections as '{output_file}'.")
    return fig, ax
  

def plot_3d_radial_density_dyn(fig, 
                               ax, 
                               name_list = ["psi_1d"], color="blue", 
                               ls="-",
                               time=5.0,
                               upsampling=True):
    # Load the 3D array from the HDF5 file
  ex = toml.load("input/experiment_pre_quench.toml")
  # scales
  hbar = 1.0545e-34
  l_perp = np.sqrt(hbar/(ex["omega_perp"]*ex["m"]))
  e_perp = hbar * ex["omega_perp"]
  t_perp = ex["omega_perp"]**(-1)
  print("TPERP_>>>>", t_perp)
  print(f"Plotting at t={time*t_perp*1e3} ms")
  for name in name_list:
    t, frames = load_hdf5_data(name)
    idx = np.searchsorted(t, time, side='left')
    xy = frames[idx][0]
    print("shape of the xy: ", np.shape(xy))
    par = toml.load("input/_params.toml")
    params = par["numerics"]
    dx = params["l"]   /(params["n_l"] - 1)
    dy  = params["l_y"]/(params["n_l_y"] - 1)
    l_y = np.linspace(-params["l_y"]/2, params["l_y"]/2, params["n_l_y"])
    # Calculate projections
    y = np.sum(xy, axis=(0)) * dx
    # assert(np.abs(np.sum(y) * dy - 1.0) < 1e-9)

    vmin, vmax = np.nanmin(y), np.nanmax(y)
    vmax = min(2.0, abs(vmax))

    # ax.plot(l_y, y, 
    #         lw=0.5, 
    #         linestyle=ls, 
    #         color="k", )
    if upsampling:
      nn = 1000
    else:
      nn = len(l_y)
    
    x_data = l_y
    y_data = y
    print("y: ", len(y))
    print("l_y: ", len(l_y))
    fraction = np.sum(y_data) * (x_data[1]-x_data[0])
    l_y_resampled = np.linspace(l_y[0], l_y[-1], nn)
    y_resampled = interp1d(l_y, y, "cubic")(l_y_resampled)
    print("normalization:", np.sum(y_data) / fraction * dy)

    ax.plot(l_y_resampled * l_perp, y_resampled/fraction/l_perp,
            lw=0.5,
            linestyle=ls,
            color=color,)
    ax.set_xlabel(r"$y $")
    ax.set_ylabel(r"$n(y, t)/N(t)$")
    plt.tight_layout()
    
    x_resampled = np.linspace(x_data[0], x_data[-1], 1000)

    print(f" Fracscion {fraction}")
    gaussian = (1/(np.sqrt(np.pi)) * np.exp(-x_resampled**2))
    # * fraction
    ax.plot(x_resampled * l_perp, gaussian/ l_perp, lw=0.5, ls="--")
    # Save the plot as a PNG file
    output_file = f"media/axial_density.png"
    plt.savefig(output_file, dpi=900)
    print(f"Saved 3D projections as '{output_file}'.")
    return fig, ax


def paper_plot():
  fig, ax = init_plotting()
  num = 9
  color = "blue"
  # plot_1d_axial_density(fig, ax, name_list=["pre-quench_1d"], color=  color)
  plot_3d_axial_density(fig, ax, name_list=["pre-quench_3d"], color=color, ls="--")
  color = "red"
  # plot_1d_axial_density(fig, ax, name_list=[name+"_1d"], color=color)
  plot_3d_axial_density(fig, ax, name_list=["idx-0_3d"], color="red", ls="-.")
  plot_3d_axial_density(fig, ax, name_list=["idx-13_3d"], color="green", ls="-")
  # plt.grid(True, which='major', linestyle='-', color='black', alpha=0.4)
  # plt.grid(True, which='minor', linestyle=':', color='gray', alpha=0.3)
  plt.xlim([-3, 3])
  plt.minorticks_on()
  plt.savefig("media/axial_density.pdf", dpi=900)
  width_from_wavefunction(f"width-check", dimensions=1)
  width_from_wavefunction(f"width-check", dimensions=3)
  print("Saved media/axial_density.pdf")
  
  # fig, ax = init_plotting()
  # num = 8
  # plot_1d_axial_density(fig, ax, name_list=[f"idx-{num}_1d"])
  # plot_3d_axial_density(fig, ax, name_list=[f"idx-{num}_3d"])
  # plt.grid(True, which='major', linestyle='-', color='black', lw=0.4, alpha=0.4)
  # plt.grid(True, which='minor', linestyle=':', color='gray' , lw=0.4, alpha=0.3)
  # plt.xlim([-5, 5])
  # plt.minorticks_on()  
  # plt.savefig("media/axial_density.png", dpi=900)
  # print(width_from_wavefunction(f"idx-{num}", dimensions=1))
  # print(width_from_wavefunction(f"idx-{num}", dimensions=3))
  
  # exit() 

  
def spr_consistency_check():
  # SPR consistency check
  cnt = 0
  fig, ax = init_plotting()
  for name in ["check_1d", "check-np_1d"]:
    plot_1d_axial_density(fig, ax, name_list=[name])
    
  for name in ["check_3d"]:
    plot_3d_axial_density(fig, ax, name_list=[name])
    
  plt.savefig("media/Fig_S5.pdf", dpi=900)
  
def ns():
  pass
  

def linear_consistency_check():
  a = 1/2
  a = np.sqrt(np.sqrt(1/2))
    
  cnt = 0
  fig, ax = init_plotting()
  for name in ["linear_3d"]:
    plot_3d_radial_density(fig, ax, name_list=[name])
  
  # linear wavefunction as per SPR transverse ansatz
  # ax.plot()
  x_data = ax.get_lines()[0].get_xdata().copy()
  dy = x_data[1]-x_data[0]
  # print("y: ", x_data)

  # gaussian  = (1/(np.sqrt(np.pi)) * np.exp(-x_data**2/2))**2
  gaussian  = 1/(np.sqrt(np.pi)) * np.exp(-x_data**2)
  gaussian2  = 1/(np.pi**(1/2)) * np.sqrt(a) * np.exp(-a * x_data**2)
 
  # gaussian_2  = (1/(np.sqrt(np.pi) * 2)  * np.exp(-x_data**2/4))**2
  # gaussian_1_2  = (1/(np.sqrt(np.pi) * 1/2) * np.exp(-x_data**2))**2
  # print(2 * np.pi * np.sum( 
  #   gaussian[int(np.round(len(gaussian)/2)):] * x_data[int(np.round(len(gaussian)/2)):]
  #   )  * dy)
  # print(2 * np.pi * np.sum( 
  #   gaussian_1_2[int(np.round(len(gaussian)/2)):] * x_data[int(np.round(len(gaussian)/2)):]
  #   )  * dy)
  print(np.sum(gaussian) * dy)
  print(np.sum(gaussian2) * dy)

  ax.plot(x_data, gaussian, ls=":", color="red")
  ax.plot(x_data, gaussian2, ls="--", color="green")
  # ax.plot(x_data, gaussian_2, ls=":", color="orange")
  # ax.plot(x_data, gaussian_1_2, ls=":", color="green")
  plt.savefig("media/linear_radial.pdf", dpi=900)
  print("Saved media/linear_radial.pdf")
  
  plt.clf()
  fig, ax = init_plotting()
  for name in ["linear_3d"]:
    plot_3d_axial_density(fig, ax, name_list=[name])
  
  # linear wavefunction as per SPR transverse ansatz
  # ax.plot()
  par = toml.load("input/_params.toml")    
  dl =par["physics"]["dl"]
  x_data = ax.get_lines()[0].get_xdata()
  dy = (x_data[1]-x_data[0])* dl
  # print("y: ", x_data)
  # gaussian  = (1/(np.sqrt(np.pi)) * np.exp(-x_data**2/2))**2
  actual_x_axis = x_data * dl
  gaussian  = 1/(np.sqrt(np.pi)) * np.exp(-actual_x_axis**2)
  gaussian2  = 1/(np.pi**(1/2)) * np.sqrt(a) * np.exp(-a * actual_x_axis**2)
 
  # gaussian_2  = (1/(np.sqrt(np.pi) * 2)  * np.exp(-x_data**2/4))**2
  # gaussian_1_2  = (1/(np.sqrt(np.pi) * 1/2) * np.exp(-x_data**2))**2
  # print(2 * np.pi * np.sum( 
  #   gaussian[int(np.round(len(gaussian)/2)):] * x_data[int(np.round(len(gaussian)/2)):]
  #   )  * dy)
  # print(2 * np.pi * np.sum( 
  #   gaussian_1_2[int(np.round(len(gaussian)/2)):] * x_data[int(np.round(len(gaussian)/2)):]
  #   )  * dy)
  print(np.sum(gaussian) * dy)
  print(np.sum(gaussian2) * dy)

  ax.plot(x_data, gaussian, ls="--", color="red")
  ax.plot(x_data, gaussian2, ls="--", color="green")

  # ax.plot(x_data, gaussian_2, ls=":", color="orange")
  # ax.plot(x_data, gaussian_1_2, ls=":", color="green")
  plt.savefig("media/linear_axial.pdf", dpi=900)
  print("Saved media/linear_axial.pdf")


if __name__ == "__main__":
  plt.rcParams.update({
      "text.usetex": True,
      "font.family": "serif",  # or 'sans-serif', or any LaTeX-supported font
      "text.latex.preamble": r"\usepackage{amsmath}",  # or any other packages
  })

  # linear_consistency_check()
  fig, ax = init_plotting()
  # plot_3d_axial_density(fig, ax, name_list=["idx-0_3d"], color="red", ls="-")
  # plot_3d_radial_density(fig, ax, name_list=["idx-1_1700_3d"], color="red", ls="-")
  t_range = np.linspace(0.8, 1.9, 2)
  for time in t_range:
    plot_3d_radial_density_dyn(fig, ax, 
                               name_list=["results/dyn_idx-3_1700_3d.h5"], 
                               color=None, 
                               ls="-",
                               time=time,
                               upsampling=True)
  ex = toml.load("input/experiment_pre_quench.toml")
  # scales
  hbar = 1.0545e-34
  l_perp = np.sqrt(hbar/(ex["omega_perp"]*ex["m"]))
  e_perp = hbar * ex["omega_perp"]
  t_perp = ex["omega_perp"]**(-1)
  plt.xlim([-3*l_perp, 3*l_perp])
  
  x_data = ax.get_lines()[0].get_xdata()
  print("number of lines: ", len(ax.get_lines()))
  try:
    df = pd.DataFrame({
          "y":        np.compress(np.abs(x_data) < 0.5e-5, x_data),
          "t=1":      np.compress(np.abs(x_data) < 0.5e-5, ax.get_lines()[0].get_ydata()),
          "gaussian": np.compress(np.abs(x_data) < 0.5e-5, ax.get_lines()[1].get_ydata()),
          "t=2":      np.compress(np.abs(x_data) < 0.5e-5, ax.get_lines()[2].get_ydata()),
      })
    df.to_csv("results/transverse.csv", index=False)
  except:
    pass
  print("Saved results/transverse.csv")
  plt.savefig("media/supplemental/axial_density.pdf", dpi=900)
  print("Saved media/supplemental/axial_density.pdf")