import pandas as pd
import numpy as np
from launch.rust_launcher import Simulation
from launch.rw import Params, write_from_experiment
from scipy.constants import physical_constants
from plot_widths import width_from_wavefunction, apply_noise_to_widths, plot_widths, plot_widths_cumulative
import os
from p1d_dyn_heatmap import *
from p3d_snap_projections import *
import time

data_widths = pd.read_csv("input/widths.csv", header=None, names=["a_s", "width", "number"])

recompute          = True
plotting_evolution = True
harmonium          = True

idx = np.array([3])

fig3 = False

default = Params.read("input/default.toml")
d = default.dimension
params = data_widths["a_s"].to_numpy()

n = len(params)
interleaved_points_n = 2
x_new = np.linspace(0, n - 1, interleaved_points_n * n - 1)
params = np.interp(x_new, np.arange(n), params)
params_original = params.copy()
indexes = np.array(range(len(params)))
print(params_original)
# exit()
# make a subselection
params =  params[idx]
indexes = indexes[idx]
print("Selecting the case of params: ", params)
assert(len(params) == len(indexes))
# params = [params[38]]
# params = [2.0]
if fig3:
  params = [-5.6]
  exp_data = "fig3"
else:
  exp_data = "experiment"

cases = np.linspace(1200, 2200, 3, dtype=int)
cases = [cases[1]]
# params = [-9.474, -4.281]
 
print("_____ computing the widths ______")
for cs in cases:
  # time.sleep(2)
  if cs == None:
    case = ""
  else: 
    case = "_"+str(cs)
  # Initialize result arrays
  result_widths = np.zeros(len(params_original))
  result_widths_rough = np.zeros(len(params_original))
  remaining_particle_fraction = np.zeros(len(params_original))

  # print("a_s list: ", params)
  print("_____ computing the GS ______")
  write_from_experiment("input/"+exp_data+"_pre_quench.toml",
                      "input/params.toml",
                      "pre-quench",
                      a_s = 20.0,
                      load_gs = False,
                      t_imaginary = 20.0,
                      n_atoms = 1700) # TODO notice this!
  l = Simulation(input_params="input/params.toml",
                output_file="results/",
                rust="./target/release/rust_waves")
  l.compile("release")
  # exit()

  if not os.path.exists(f"results/pre-quench"+case+f"_{d}d.h5") or recompute:
    l.run()
  if plotting_evolution:
    if d == 1:
      plot_heatmap_h5(f"results/dyn_pre-quench"+case+f"_{d}d.h5")
      plot_snap(f"results/pre-quench_{d}d"+case+".h5")
    elif d == 3:
      # plot_projections([f"pre-quench"+case+f"_{d}d"+case])
      plot_heatmap_h5_3d(f"results/dyn_pre-quench"+case+f"_{d}d.h5", -1)
      # movie(f"dyn_pre-quench"+case+f"_{d}d")

  pf0, w0 = width_from_wavefunction(f"pre-quench",
          dimensions=d, 
          harmonium=harmonium)
  print(f"pre-quench: fraction = {pf0:3.2f}, width = {w0:3.2f}")
  
  # exit()
  ## Iterate over the scattering lengths
  for i, a_s in zip(indexes, params):
    # a_s = a_s/2
    write_from_experiment("input/"+exp_data+".toml", 
                          "input/params.toml", 
                          f"idx-{i}"+case, 
                          a_s=a_s,
                          load_gs=True,
                          n_atoms=cs)
    l = Simulation(input_params="input/params.toml",
                output_file="results/",
                rust="./target/release/rust_waves",)
    # exit() # save the zero simulation
    name = f"results/idx-{i}"+case+f"_{d}d.h5"
    print("Searching for ", name)
    if not os.path.exists(name) or recompute:
      # print("Computing wavefunction for ", f"results/idx-{i}"+case+f"_{d}d.h5")
      l.run()
      # pass
    else:
      print("  Found!")
    if plotting_evolution:
      if d == 1:
        plot_heatmap_h5(f"results/dyn_idx-{i}"+case+f"_{d}d.h5", i)
        plot_snap(f"results/idx-{i}_{d}d"+case+".h5", i)
      elif d == 3:
        # plot_projections([f"idx-{i}"+case+f"_{d}d"], i)
        plot_heatmap_h5_3d(f"results/dyn_idx-{i}"+case+f"_{d}d.h5", i)
        # movie(f"dyn_idx-{i}"+case+f"_{d}d"+case, i)
        # movie(f"dyn_idx-{i}"+case+f"_{d}d", i)
    try:
      remaining_particle_fraction[i], result_widths[i] = width_from_wavefunction(f"idx-{i}"+case,
          dimensions=d,
          harmonium=harmonium,
          particle_threshold=0.05)
    except:
      remaining_particle_fraction[i], result_widths[i] = 0.0, 0.0
    # print("Width: ", result_widths[i])  
  
  if len(params)==len(params_original) and not fig3:
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
    print("Plotting...")
    plot_widths(noise=0.0,
                  plot=True,
                  initial_number=3000,
                  case = case)

  # cases = np.linspace(1200, 2200, 5, dtype=int)
  # # cases= cases[:-2]
  # plot_widths_cumulative(cases = cases,
  #                        a_s_limit = -30)