import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import h5py
from matplotlib import cm
import toml
from matplotlib.colorbar import Colorbar
import re
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for text rendering
    "font.family": "serif",  # Use a serif font
    "font.serif": ["Computer Modern"],  # Default LaTeX font
    "font.size": 12,  # Set the base font size
})

hbar = 1.0545718e-34
m = 2.21e-25

def plot_heatmap_h5(filename="results/1d.h5", i=-1, experiment_file = "input/experiment.toml"):
  with h5py.File(filename, "r") as f:
      l = np.array(f["l"])  # Load l (spatial coordinate)
      t = np.array(f["t"])  # Load t (time coordinate)
      psi_squared = np.array(f["psi_squared"]).reshape(len(t), len(l))  # Load psi_squared dataset
  
  par = toml.load("input/_params.toml")
  plt.figure(figsize=(3, 3))
  dl = par["physics"]["dl"]
  # extent = [t.min(), t.max(), l.min(), l.max()]
  extent = [t.min(), t.max(), l.min(), l.max()]
  exp_par = toml.load(experiment_file)
  
  l_perp = np.sqrt(hbar / (m * exp_par["omega_perp"]))
  print("l_perp ", l_perp)
  x_max = l.max() * l_perp * 1e6
  x_min = -x_max
  time_points = len(t)
  
  d = dl * l_perp
  
  # print(f"Saved 3D heatmap as 'media/idx-{i}_heatmap.png'.")
  fig = plt.figure(figsize=(4, 3.5))
  gs = fig.add_gridspec(2, 2, width_ratios=[40, 1], height_ratios=[4, 1], wspace=0.15, hspace=0.2)

  # Heatmap plot
  ax_heatmap = fig.add_subplot(gs[0, 0])

  ax_heatmap.imshow(
      psi_squared.T,
      cmap="nipy_spectral",
      # cbar=False,  # Disable the default colorbar
      # ax=ax_heatmap, 
      interpolation="none",
      extent=extent,
  )
  x_zoom = 10
  assert(x_zoom < x_max)
  if x_zoom is not None:
    lim_bottom = -x_zoom 
    lim_top    =  x_zoom   
  else: 
    lim_bottom = x_min
    lim_top =    x_max

  # ax_heatmap.set_yticks([0, lim_bottom, int(round(space_points/2)), lim_top, space_points - 1])  # Positions: start and end of space
  ax_heatmap.set_ylim(bottom=lim_bottom, top=lim_top)
  ax_heatmap.set_xlim([0.0, t.max()])
  # if x_zoom is not None:
  #   ax_heatmap.set_yticklabels([f"{x_min:.1f}", f"{-x_zoom:.1f}", f"{0.0:.1f}", f"{x_zoom:.1f}", f"{x_max:.1f}"])  # Labels: min and max space
  ax_heatmap.set_ylabel(r'$z \quad [\mu m]$')
  ax_heatmap.axhline(+d/2*1e6, color='w', linestyle='-', lw=0.3)
  ax_heatmap.axhline(-d/2*1e6, color='w', linestyle='-', lw=0.3)
  ax_heatmap.axvline(t.max() /(2 * np.pi))
  ax_heatmap.axvline(t.max()+1.5)

  ax_heatmap.set_aspect(t.max() / (lim_top-lim_bottom))
  # ax_heatmap.set_aspect(aspect)
  # Add colorbar to the right of the entire plot
  cbar_ax = fig.add_subplot(gs[:, 1])  # Colorbar spans both rows
  norm = plt.Normalize(vmin=np.min(psi_squared), vmax=np.max(psi_squared))
  sm = plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm)
  cbar = Colorbar(cbar_ax, sm, orientation='vertical')
  cbar.set_label(r'$n(z)$', rotation=90)

  atom_number = np.sum(psi_squared, axis=1) * (l[1] - l[0])
  # Line plot for atom number
  ax_lineplot = fig.add_subplot(gs[1, 0], sharex=ax_heatmap)
  # ax2 = fig.add_axes(ax_heatmap.get_position(), sharex=ax_hea)
  ax_lineplot.plot(atom_number, color='blue')
  # ax_lineplot.set_xlabel(r'$t \quad [\mathrm{ms}]$')
  ax_lineplot.set_xlabel(r'$t \quad [\omega_\perp^{-1}]$')
  ax_lineplot.set_ylabel(r'$N(t)/N_0$')
  ax_lineplot.set_xticks([0, t.max()])
  ax_lineplot.set_xticklabels([f"{0.0:.1f}", fr"{{{t.max():3.2f}}}"])  # Labels: min and max time
  ax_lineplot.axhline(2/3, color='b', linestyle='-.', lw=0.3)
  
  ax_heatmap.get_xaxis().set_visible(False)
  # ax_lineplot.text(f'{atom_number[:-1]}')
  ax_lineplot.text(
    0.95, 0.05,  # Position of text (relative to axes, [x, y] from bottom-left corner)
    f'$N(t_f)/N_0 = {atom_number[-1]:.2f}$',  # Format the final value
    transform=ax_lineplot.transAxes,  # Use axes coordinates
    color='black', fontsize=8, ha='right', va='bottom'
  )
  ax_lineplot.set_ylim((0.33, 1.1))
  # ax_lineplot.legend(loc="upper right")
  fig.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=0.15)
  
  plt.tight_layout()

  heatmap_filename = "media/idx-"+str(i)+".pdf"
  plt.savefig(heatmap_filename, dpi=300, pad_inches=0.1)
  plt.close()
  print(f"Heatmap saved as {heatmap_filename}")
  
  # # print("WARN, omega_perp depends!")
  # # ex = toml.load("input/experiment.toml")
  # # omega_perp = params["experiment"]["omega_perp"]
  # plt.figure(figsize=(3, 3))
  # extent = [t.min(), t.max(), l.min()/dl, l.max()/dl]
  # aspect = (t.max() - t.min()) / (l.max()/dl - l.min()/dl)

  # plt.imshow(psi_squared.T, extent=extent, origin="lower", aspect=aspect, cmap="gist_ncar")
  # # plt.colorbar(label=r"$|\psi|^2$")
  # plt.axhline(+5, color="w", lw=0.4)
  # plt.axhline(-5, color="w", lw=0.4)
  # plt.xlabel(r"$t \; [\omega_\perp]$")
  # plt.ylabel(r"$x \; [d_L]$")
  # plt.axhline(+1, color="w", lw=0.3)
  # plt.axhline(-1, color="w", lw=0.3)
  # plt.axvline(t.max()/(2 * np.pi), color="w", lw=0.3)


def load_hdf5_data_single_axis(filename):
  """Load HDF5 file and extract projections."""
  with h5py.File(filename, "r") as f:
      # Read time values
      t   = np.array(f["t"])
      l_x = np.array(f["l_x"])
      l_z = np.array(f["l_z"])
      
      # Load projection frames
      frames = []
      numbers = []
      dx = l_x[1] - l_x[0]
      dz = l_z[1] - l_z[0]
      
      for i in range(len(t)):
          frame_group = f[f"movie/frame_{i}"]
          # print(frame_group)
          x = dz * np.sum(np.array(frame_group["xz"]), axis=1)
          frames.append(x)
          numbers.append(np.sum(x) * dx)
      # print(">>>>", len(frames))
      # print(">>", len(frames[-1]))
      # print("->", frames[-1][0])
  return t, l_x, frames, np.array(numbers)


def plot_heatmap_h5_3d(filename="3d", i=-1, experiment_file = "input/experiment.toml"):
  par = toml.load("input/_params.toml")
  params = par["numerics"]
  t, l, frames, atom_number = load_hdf5_data_single_axis(filename) 
  plt.figure(figsize=(3, 3))
  
  extent = [t.min(), t.max(), l.min(), l.max()]
  # print(t)
  # print(len(t))
  psi2_values = np.array([f for f in frames]).reshape(len(t), len(l)).T
  exp_par = toml.load(experiment_file)
  l_perp = np.sqrt(hbar / (m * exp_par["omega_perp"]))
  t_perp = 1 / exp_par["omega_perp"]
  x_min = l.min() * l_perp * 1e6
  x_max = l.max() * l_perp * 1e6
  x_min = -x_max
  space_points = len(l)
  time_points = len(t)
  
  d = par["physics"]["dl"] * l_perp

  # fig, axes = plt.subplots(1, 1, figsize=(3, 2.2), dpi=600)
  # # aspect = (t.max() - t.min()) / (l.max() - l.min())

  # im2 = plt.imshow(psi_squared,
  #                     aspect="equal", 
  #                     origin="lower", 
  #                     cmap="nipy_spectral", 
  #                     interpolation="bicubic",
  #                     extent=extent, 
  #                     # vmin=vmin, vmax=vmax)
  # )
  # plt.colorbar(label=r"$|f|^2$")
  # plt.xlabel(r"$t$")
  # plt.ylabel(r"$x$")
  # plt.tight_layout()
  # plt.savefig(f"media/idx-{i}_heatmap.png", dpi=600)
  # print(f"Saved 3D heatmap as 'media/idx-{i}_heatmap.png'.")
  fig = plt.figure(figsize=(4, 3.5))
  gs = fig.add_gridspec(2, 2, width_ratios=[40, 1], height_ratios=[4, 1], wspace=0.15, hspace=0.2)

  ax_heatmap = fig.add_subplot(gs[0, 0])  
  
  # saving for export
  # FIXME 
  print("WARN: check the normalization used")
  psi2_values /= l_perp
  psi2_values *= exp_par["n_atoms"]
  # print(f"l_perp = {l_perp}, dl = {(l[1]-l[0])*l_perp}")
  df = pd.DataFrame(psi2_values)
  df.to_csv("results/widths/"+str(i)+".csv", index=False, header=False)
  
  psi2_values *= l_perp
  ax_heatmap.imshow(
      psi2_values,
      cmap="nipy_spectral",
      # cbar=False,  # Disable the default colorbar
      # ax=ax_heatmap, 
      interpolation="none",
  )
  x_zoom = 10
  lim_bottom = int(round((x_max-x_zoom)/(2*x_max) * space_points))
  lim_top = int(round((x_max+x_zoom)/(2*x_max)* space_points))
  ax_heatmap.set_yticks([0, lim_bottom, int(round(space_points/2)), lim_top, space_points - 1])  # Positions: start and end of space
  ax_heatmap.set_ylim(bottom=lim_bottom, top=lim_top)
  ax_heatmap.set_yticklabels([f"{x_min:.1f}", f"{-x_zoom:.1f}", f"{0.0:.1f}", f"{x_zoom:.1f}", f"{x_max:.1f}"])  # Labels: min and max space
  ax_heatmap.set_ylabel(r'$x \quad [\mu m]$')
  ax_heatmap.axhline((x_max+d/2*1e6)/(2*x_max)*space_points, color='w', linestyle='--', lw=1)
  ax_heatmap.axhline((x_max-d/2*1e6)/(2*x_max)*space_points, color='w', linestyle='--', lw=1)
  # ax_heatmap.set_aspect(t_max / (2*x_max))
  ax_heatmap.set_aspect(len(t) / (lim_top-lim_bottom))
  # Add colorbar to the right of the entire plot
  cbar_ax = fig.add_subplot(gs[:, 1])  # Colorbar spans both rows
  norm = plt.Normalize(vmin=np.min(psi2_values), vmax=np.max(psi2_values))
  sm = plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm)
  cbar = Colorbar(cbar_ax, sm, orientation='vertical')
  cbar.set_label(r'$n(x) \; [\mathrm{atom fraction} /\mathrm{m}^3]$', rotation=90)

  # Line plot for atom number
  dz = (l[1]-l[0])
  atom_number = np.sum(psi2_values/l_perp, axis=0) * dz * l_perp
  ax_lineplot = fig.add_subplot(gs[1, 0], sharex=ax_heatmap)
  ax_lineplot.plot(atom_number, color='blue', lw=0.2)
  ax_lineplot.set_xlabel(r'$t \quad [\mathrm{ms}]$')
  ax_lineplot.set_ylabel(r'$N(t)/N_0$')
  ax_lineplot.set_xticks([0, time_points - 1])  # Positions: start and end of time
  ax_lineplot.set_xticklabels([f"{0.0:.1f}", f"{t_perp*t[-1]:.1f}"])  # Labels: min and max time
  ax_heatmap.get_xaxis().set_visible(False)
  # ax_lineplot.text(f'{atom_number[:-1]}')
  ax_lineplot.text(
    0.95, 0.05,  # Position of text (relative to axes, [x, y] from bottom-left corner)
    f'$N(t_f)/N_0 = {atom_number[-1]:.2f}$',  # Format the final value
    transform=ax_lineplot.transAxes,  # Use axes coordinates
    color='black', fontsize=8, ha='right', va='bottom'
  )
  # ax_lineplot.set_ylim((0.0, 1.1))
  # ax_lineplot.legend(loc="upper right")
  fig.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=0.15)

  heatmap_filename = "media/3d_heatmap_idx-"+str(i)+".png"
  plt.savefig(heatmap_filename, dpi=300, pad_inches=0.1)
  plt.close()
  print(f"Heatmap saved as {heatmap_filename}")


def plot_heatmap(filename):
    # Initialize lists to store the data
    x = []
    matrix = []

    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip header row
        # Initialize the matrix with empty lists for each column in psi
        num_columns = len(header) - 1

        # Read the rest of the rows
        for row in csvreader:
            x.append(float(row[0]))  # The first column is l_range
            row_data = []
            for col in range(1, num_columns + 1):
                try:
                    complex_num = complex(row[col].replace('i', 'j'))
                    row_data.append(abs(complex_num)**2)
                except ValueError:
                    if row[col].strip().lower() == 'nan':
                        row_data.append(np.nan)
                    elif row[col].strip().lower() == 'inf':
                        row_data.append(np.nan)
                    else:
                        print(f"Unknown value encountered: {row[col]}")
                        row_data.append(np.nan)
            matrix.append(row_data)

    # Convert matrix to numpy array for easier plotting
    matrix = np.array(matrix).T  # Transpose to get the correct orientation

    # Create a heatmap
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, aspect='auto', cmap='nipy_spectral', origin='lower', zorder=2)
    # sns.heatmap(matrix, annot=False, cmap='nipy_spectral', xticklabels=x, zorder=3)

    # Add titles and labels
    plt.xlabel('Space')
    plt.ylabel('n_saves')

    # Display the heatmap
    plt.tight_layout()
    plt.grid(False, zorder=0)
    plt.savefig("media/heatmap.png", dpi=600)
    
    
def plot_final(filename, ax, ix):
    x = []
    y = []
    nipy_spectral = cm.get_cmap('nipy_spectral')
    with h5py.File(filename, 'r') as hdf:
        # Assuming the datasets are named 'l' and 'field'
        x = np.array(hdf['l'])
        y_data = np.array(hdf['psi_squared'])
        if len(x) == 0:
          x = np.linspace(0, len(y_data), len(y_data))
        # Process `field` values to handle potential complex and special cases
        for value in y_data:
            try:
                # Convert to complex if necessary
                complex_num = complex(str(value).replace('i', 'j'))
                y.append(complex_num)
            except ValueError:
                if str(value).strip().lower() == 'nan':
                    y.append(np.nan)
                elif str(value).strip().lower() == 'inf':
                    y.append(np.nan)
                else:
                    print(f"Unknown value encountered: {value}")
                    y.append(np.nan)

    ax.plot(x, y, linestyle='-', color=nipy_spectral(
      ix/1.5), lw=0.5)
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'$|\psi|^2$')
    return ax


def plot_final_3d(filename, ax, ix):
    x = []
    y = []
    nipy_spectral = cm.get_cmap('nipy_spectral')
    with h5py.File(filename, 'r') as hdf:
        # Assuming the datasets are named 'l' and 'field'
        x = np.array(hdf['l'])
        y_data = np.array(hdf['psi_squared'])
        if len(x) == 0:
          x = np.linspace(0, len(y_data), len(y_data))
        # Process `field` values to handle potential complex and special cases
        for value in y_data:
            try:
                # Convert to complex if necessary
                complex_num = complex(str(value).replace('i', 'j'))
                y.append(complex_num)
            except ValueError:
                if str(value).strip().lower() == 'nan':
                    y.append(np.nan)
                elif str(value).strip().lower() == 'inf':
                    y.append(np.nan)
                else:
                    print(f"Unknown value encountered: {value}")
                    y.append(np.nan)

    ax.plot(x, y, linestyle='-', color=nipy_spectral(
      ix/1.5), lw=0.5)
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'$|\psi|^2$')
    return ax


def plot_first_last():
  plt.figure(figsize=(3, 2))
  ax = plt.gca()
  for ix, name in enumerate(["1d_psi_0.h5", "1d_psi_end.h5"]):
    plot_final('results/'+name, ax, ix)
  plt.tight_layout()
  plt.savefig("media/1d_first_last.png", dpi=600)
  print("Saved 1D first-last plot as 'media/1d_first_last.png'.")

 
def plot_snap(filename, i=-1):
  plt.figure(figsize=(3, 1.5))
  ax = plt.gca()
  plot_final(filename, ax, 0)
  plt.tight_layout()
  plt.savefig(f"media/idx-{i}_snap.png", dpi=600)

    
if __name__ == "__main__":
  print("____ Plotting _____")
  # plot_heatmap_h5()
  # plot_first_last()
  # plot_heatmap_h5('results/dyn_check-np_1d.h5')
  # fig = plt.figure(figsize=(3, 2))
  # ax = plt.gca()
  # filename = 'results/idx-0_1d.h5'
  # ax = plot_final(filename, ax, 0)
  # plt.tight_layout()
  # plt.show()
  # plot_heatmap_h5_3d('dyn_idx-26_3d')
  # plot_heatmap_h5_3d('dyn_idx-12_3d')
  losses = 5e-39
  particles = 1200
  for case in [x for x in range(0, 26)][::2]:
    plot_heatmap_h5_3d('results-'+str(losses)+'/dyn_idx-'+str(case)+'_'+str(particles)+'_3d.h5', case)