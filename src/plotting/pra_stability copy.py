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
from scipy.optimize import bisect
import time

data_widths = pd.read_csv("input/widths.csv", header=None, names=["a_s", "width", "number"])

recompute          = True
plotting_evolution = False
# dimension
default = Params.read("input/default.toml")
d = default.dimension
params = data_widths["a_s"].to_numpy()
# params = [params[0]]


kl = np.linspace(0, 4, 10)
v0s = [0.1, 0.5, 1, 5]
v0s = [5.0]
"""
A function to be used with the bisection method: return -1 or +1 if 
the collapse is happening or not
"""
def collapse_or_not(g, v0, kl):
  
  write_from_experiment("input/experiment_pre_quench.toml",
                      "input/params.toml",
                      "pre-quench",
                      g = g,
                      load_gs = False, 
                      v_0 = v0, 
                      free_x=True, 
                      t_imaginary = 100.0)
  l = Simulation(input_params="input/params.toml",
                output_file="results/",
                rust="./target/release/rust_waves")
  l.compile("release")
  flag = 1
  try:
    l.run()
  except:
    print("Collapse happened")
    flag = -1
  print(f"Trying g = {g:>10.3f}, v0 = {v0:>10.3f} => flag = {flag}")
  # time.sleep(2)
  return flag

# exit()
# cases = ["", "_low", "_high"]
gcs = np.zeros(len(v0s))
print("_____ computing the widths ______")
for iv, v0 in enumerate(v0s):
  gcs[iv] = bisect(collapse_or_not, 0.0, -1.5, 
                   args=(v0, kl),
                  maxiter=100, 
                  xtol=1e-3)

print("Saving the csv file...")
df = pd.DataFrame({
    "v0": v0s,
    "g_c": gcs,  # Use .to_numpy() if it's a Pandas series
})


if default.npse == True and default.dimension == 1:
  df.to_csv(f"results/ol_stability_npse.csv", index=False)
else:
  df.to_csv(f"results/ol_stability_{d}d.csv", index=False)

print("Plotting...")

plt.figure(figsize=(4, 3))
plt.plot(v0s, gcs)
plt.xlabel(r"$V_0$")
plt.ylabel(r"$g_c$")
plt.axhline(-1.333, color="red", linestyle="--")
plt.axhline(-1.333/2, color="gray", linestyle=":")
plt.grid()
plt.tight_layout()
plt.savefig(f"media/ol_stability.png")

print("Saved plot in media/ol_stability.png")
print("Done!")