import pandas as pd
import numpy as np
from launch.rust_launcher import Simulation
from launch.rw import Params, write_from_experiment
from scipy.constants import physical_constants
from plot_widths import width_from_wavefunction, apply_noise_to_widths, plot_widths
import os
import pandas
from p1d_dyn_heatmap import *
from p3d_snap_projections import *

data_widths = pd.read_csv("input/widths.csv", header=None, names=["a_s", "width", "number"])

recompute          = False
recompute          = True
#
plotting_evolution = False
plotting_evolution = True
#
harmonium          = False
harmonium          = True

fig3 = False

default = Params.read("input/default.toml")
d = default.dimension
params = data_widths["a_s"].to_numpy()
n = len(params)
interleaved_points_n = 5
x_new = np.linspace(0, n - 1, interleaved_points_n * n - 1)
params = np.interp(x_new, np.arange(n), params)
# params = [params[38]]
# params = [2.0]
if fig3:
  params = [-5.6]
  exp_data = "fig3"
else:
  exp_data = "experiment"
  
cases = ["", "_low", "_high"]
cases = [""]
print("_____ computing the widths ______")
for case in cases:
  # Initialize result arrays
  result_widths = np.zeros(len(params))
  result_widths_rough = np.zeros(len(params))
  remaining_particle_fraction = np.zeros(len(params))

  print("a_s list: ", params)
  print("_____ computing the GS ______")
  write_from_experiment("input/"+exp_data+"_pre_quench"+case+".toml",
                      "input/params.toml",
                      "pre-quench"+case,
                      a_s = 20.0,
                      load_gs = False)
  l = Simulation(input_params="input/params.toml",
                output_file="results/",
                rust="./target/release/rust_waves")
  l.compile("release")
  # exit()
  if not os.path.exists(f"results/pre-quench"+case+f"_{d}d.h5") or recompute:
    l.run()
  if plotting_evolution:
    if d == 1:
      # plot_heatmap_h5(f"results/dyn_pre-quench"+case+f"_{d}d.h5")
      plot_snap(f"results/pre-quench_{d}d"+case+".h5")
    elif d == 3:
      # plot_projections([f"pre-quench"+case+f"_{d}d"+case])
      # plot_heatmap_h5_3d(f"dyn_pre-quench"+case+f"_{d}d"+case, -1)
      movie(f"dyn_pre-quench_{d}d"+case)
      # plot_snap(f"results/pre-quench_{d}d.h5")
  pf0, w0 = width_from_wavefunction(f"pre-quench",
          dimensions=d, 
          harmonium=harmonium)
  print(f"pre-quench: fraction = {pf0:3.2f}, width = {w0:3.2f}")
  
  # exit()
  ## Iterate over the scattering lengths
  start_from = 0
  for i, a_s in enumerate(params):
    # a_s = a_s/2
    i += start_from
    write_from_experiment("input/"+exp_data+case+".toml", 
                          "input/params.toml", 
                          f"idx-{i}"+case, 
                          a_s=a_s,
                          load_gs=True)
    l = Simulation(input_params="input/params.toml",
                output_file="results/",
                rust="./target/release/rust_waves",)
    # exit() # save the zero simulation
    if not os.path.exists(f"results/idx-{i}"+case+f"_{d}d.h5") or recompute:
      print("Computing wavefunction for ", f"results/idx-{i}"+case+f"_{d}d.h5")
      l.run()
    if plotting_evolution:
      if d == 1: 
        plot_heatmap_h5(f"results/dyn_idx-{i}"+case+f"_{d}d.h5", i)
        plot_snap(f"results/idx-{i}_{d}d"+case+".h5", i)
      elif d == 3:
        # plot_projections([f"idx-{i}"+case+f"_{d}d"], i)
        plot_heatmap_h5_3d(f"dyn_idx-{i}"+case+f"_{d}d"+case, i)
        movie(f"dyn_idx-{i}"+case+f"_{d}d"+case, i)
        # movie(f"dyn_idx-{i}_{d}d"+case, i)
        # plot_snap(f"results/idx-{i}_{d}d.h5", i)
      
    if start_from == 0:
      remaining_particle_fraction[i], result_widths[i] = width_from_wavefunction(f"idx-{i}",
          dimensions=d,
          harmonium=harmonium)
      print("Width: ", result_widths[i])

  if start_from == 0 and not fig3:
    print("Saving the width csv file...")
    df = pd.DataFrame({
        "a_s": params,
        "width": np.zeros_like(len(params)),  # Use .to_numpy() if it's a Pandas series
        "width_sim": result_widths,
        "width_rough": result_widths_rough,
        "particle_fraction": remaining_particle_fraction
    })
    if default.npse == True and default.dimension == 1:
      df.to_csv(f"results/widths/widths_final_npse"+case+".csv", index=False)
    else:
      df.to_csv(f"results/widths/widths_final_{d}d"+case+".csv", index=False)
  if not fig3:
    print("Plotting...")
    plot_widths(noise=0.0, 
                plot=True,
                initial_number=3000,
                case = case)
  print("Done!")