import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import genlaguerre
import seaborn as sns
import matplotlib.animation as animation
import toml
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import Colorbar
import re
import h5py
from launch.rw import Params

def width_from_wavefunction(title, dimensions=1):
  filename = "".join(["results/", title, "_", str(dimensions), "d.h5"])
  print("Computing wavefunction for ", filename)
  params = Params.read("input/params.toml")
  # we sum only on [-4, +4] lattice sites.
  min_idx = params.dl * -4
  max_idx = -min_idx
  if dimensions == 1:
    with h5py.File(filename, "r") as f:
      l = np.array(f["l"])
      final_psi2 = np.array(f["psi_squared"])
    mask  = (l >= min_idx) & (l <= max_idx)
    # print(mask)
    dz = l[1] - l[0]
    particle_fraction = np.sum(final_psi2[mask]) * dz
    center = np.sum(l[mask] * final_psi2[mask]) / particle_fraction
    std = np.sqrt(dz * np.sum(l[mask]**2 * final_psi2[mask]) - np.sum(l[mask] * final_psi2[mask])**2 / particle_fraction)
    print(f"\n center = {center:3.2e}, std = {std:3.2e} l_perp\n")
  else:
    with h5py.File(filename, "r") as f:
        l_x = np.array(f["l_x"])
        l_y = np.array(f["l_y"])
        l_z = np.array(f["l_z"])
        psi_squared = np.array(f["psi_squared"])
    mask  = (l_x >= min_idx) & (l_x <= max_idx)
    # print(mask)
    dx = l_x[1] - l_x[0]
    dy = l_y[1] - l_y[0]
    dz = l_z[1] - l_z[0]
    dV = dx * dy * dz
    particle_fraction = np.sum(psi_squared[mask, :, :]) * dV
    x_mean = np.sum(l_x[mask, None, None] * psi_squared[mask, :, :]) * dV / particle_fraction
    std = np.sqrt(np.sum(l_x[mask, None, None]**2 * psi_squared[mask, :, :]) * dV / particle_fraction - x_mean**2)

  if np.isnan(particle_fraction):
    particle_fraction = 0
  return particle_fraction, std

def apply_noise_to_widths(w, l, noise_atoms, n_atoms):
  return (w*n_atoms+1/12*l**2*noise_atoms)/(n_atoms+noise_atoms)

def plot_widths(noise=0.0):
  """
  Confrontation with the experimental data
  """
  data_1 = pd.read_csv("results/widths_final_1d.csv", header=0, names=["a_s", "width", "width_sim", "width_rough", "particle_fraction"])
  data_npse = pd.read_csv("results/widths_final_npse.csv", header=0, names=["a_s", "width", "width_sim", "width_rough", "particle_fraction"])
  data_3 = pd.read_csv("results/widths_final_3d.csv", header=0, names=["a_s", "width", "width_sim", "width_rough", "particle_fraction"])
  
  # Extract columns
  a_s = data_1["a_s"]  # First column as x-axis
  width = data_1["width"]  # Second column as y-axis
  # Create the plot
  plt.figure(figsize=(3.6, 3))
  plt.plot(a_s, width, marker='o', linestyle='-', color='b', label='Width vs a_s')
  cf = toml.load("input/experiment.toml")
  n_atoms = cf["n_atoms"]
  print(f"applying the noise of ", noise)
  noise_atoms = n_atoms * noise
  l = 8 # lattice sites
  for data in [data_1, data_npse, data_3]:
    width = data["width_sim"]
    print("before: \n ", width)
    width = apply_noise_to_widths(width, l, noise_atoms, n_atoms)
    print("after: \n ", width)
    plt.plot(a_s, width, marker='.', linestyle='-.', label='Width vs a_s (sim)')
  plt.xlabel(r"$a_s/a_0$")
  plt.ylabel(r"$w_z$ [sites] ")
  plt.tight_layout()
  plt.savefig("media/widths.pdf", dpi=300)

  for data in [data_1, data_npse, data_3]:
    fraction = data["particle_fraction"]  # Second column as y-axis
    plt.clf()
    plt.figure(figsize=(3.6, 3))
    plt.plot(a_s, fraction, marker='o', linestyle='-.', color='r', label='Width vs a_s (sim)')
  plt.xlabel(r"$a_s/a_0$")
  plt.ylabel(r"$N_{\mathrm{tot}}/N_0$")
  plt.tight_layout()
  plt.savefig("media/fraction.pdf", dpi=300)
    
if __name__ == "__main__":
  plot_widths(noise=0.0)