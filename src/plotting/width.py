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

recompute = True
plotting_evolution = True
# dimension
default = Params.read("input/default.toml")
d = default.dimension
params = data_widths["a_s"].to_numpy()
# params = [params[0]]

# Initialize result arrays
result_widths = np.zeros(len(params))
result_widths_rough = np.zeros(len(params))
remaining_particle_fraction = np.zeros(len(params))

print("a_s list: ", params)
print("_____ computing the GS ______")
write_from_experiment("input/experiment_pre_quench.toml",
                     "input/params.toml",
                     "pre-quench",
                     a_s = 20.0,
                     load_gs = False)
l = Simulation(input_params="input/params.toml",
               output_file="results/",
               rust="./target/release/rust_waves",
               dimension=d)
l.compile("release")
if not os.path.exists(f"results/pre-quench_{d}d.h5") or recompute:
  l.run()
if plotting_evolution:
  if d == 1:
    plot_heatmap_h5(f"results/dyn_pre-quench_{d}d.h5")
    plot_snap(f"results/pre-quench_{d}d.h5")
  elif d == 3:
    plot_projections([f"pre-quench_{d}d"])
    movie(f"dyn_pre-quench_{d}d")
    # plot_snap(f"results/pre-quench_{d}d.h5")

# exit()
print("_____ computing the widths ______")
for i, a_s in enumerate(params):
  write_from_experiment("input/experiment.toml", 
                        "input/params.toml", 
                        f"idx-{i}", 
                        a_s=a_s, 
                        load_gs=True)
  l = Simulation(input_params="input/params.toml",
               output_file="results/",
               rust="./target/release/rust_waves",
               dimension=d)
  # exit() # save the zero simulation
  if not os.path.exists(f"results/idx-{i}_{d}d.h5") or recompute:
    print("Computing wavefunction for ", f"results/idx-{i}_{d}d.h5")
    l.run()
  if plotting_evolution:
    if d == 1: 
      plot_heatmap_h5(f"results/dyn_idx-{i}_{d}d.h5", i)
      plot_snap(f"results/idx-{i}_{d}d.h5", i)
    elif d == 3:
      plot_projections([f"idx-{i}_{d}d"], i)
      movie(f"dyn_idx-{i}_{d}d", i)
      # plot_snap(f"results/idx-{i}_{d}d.h5", i)
    
  remaining_particle_fraction[i], result_widths[i] = width_from_wavefunction(f"idx-{i}", dimensions=d)
  print("Width: ", result_widths[i])

print("Saving the csv file...")
df = pd.DataFrame({
    "a_s": params,
    "width": data_widths["width"],  # Use .to_numpy() if it's a Pandas series
    "width_sim": result_widths,
    "width_rough": result_widths_rough,
    "particle_fraction": remaining_particle_fraction
})
if default.npse == True and default.dimension == 1:
  df.to_csv(f"results/widths_final_npse.csv", index=False)
else:
  df.to_csv(f"results/widths_final_{d}d.csv", index=False)

print("Plotting...")
plot_widths(noise=0.0, 
            plot=True,
            initial_number=3000)
print("Done!")
