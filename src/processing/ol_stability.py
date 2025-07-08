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
from scipy.interpolate import interp1d

"""
A function to be used with the bisection method: return -1 or +1 if 
the collapse is happening or not
"""
def collapse_or_not(g, v0):
  
  write_from_experiment("input/experiment_pre_quench.toml",
                      "input/_params.toml",
                      "pre-quench",
                      g = g,
                      load_gs = False, 
                      v_0 = v0, 
                      free_x=True, 
                      t_imaginary = 8.0)
  l = Simulation(input_params="input/_params.toml",
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

data_widths = pd.read_csv("input/widths.csv", header=None, names=["a_s", "width", "number"])
ex = toml.load("input/experiment_pre_quench.toml")
# scales
a0 = physical_constants["Bohr radius"][0]
l_perp = np.sqrt(hbar/(ex["omega_perp"]*ex["m"]))
e_perp = hbar * ex["omega_perp"]
t_perp = ex["omega_perp"]**(-1) 
e_recoil = (np.pi * hbar / ex["d"])**2 / (2 * ex["m"])
a0_l_perp = a0 / l_perp 

recompute          = True
plotting_evolution = False
# dimension
default = Params.read("input/_default.toml")
d = default.dimension
params = data_widths["a_s"].to_numpy()
# params = [params[0]]

physical_v0_max = 3 # Er
v0_max = physical_v0_max * e_recoil / e_perp
print(f"Maximum v0 = {v0_max} e_perp")
print(f"e_recoil / e_perp = {e_recoil/e_perp}")
# exit()
v0s = [5.0]
v0s = np.linspace(0.0, 3, 5)
v0s = np.linspace(0.0, v0_max, 8)

if default.npse == True and default.dimension == 1:
  name = f"results/ol_stability_npse.csv"
else:
  name = f"results/ol_stability_{d}d.csv"
 
if ~os.path.exists(name) or recompute:
  # exit()
  # cases = ["", "_low", "_high"]
  gcs = np.zeros(len(v0s))
  print("_____ searching for collapse points ______")
  for iv, v0 in enumerate(v0s):
    try:
      gcs[iv] = bisect(collapse_or_not, 0.0, -1.5, 
                    args=(v0),
                    maxiter=100, 
                    xtol=1e-2)
    except ValueError as e:
      print(f">>> Error: {e}")
      gcs[iv] = np.nan

  print("Saving the csv file...")
  df = pd.DataFrame({
      "v0": v0s,
      "g_c": gcs,  # Use .to_numpy() if it's a Pandas series
  })

  df.to_csv(name, index=False)

print("Plotting...")

x_var = np.array([-1.5359746434231374, -1.481122543380501, -1.3928290442211617, -1.3279694627012153, 
     -1.2817170392656887, -1.2641033891896671, -1.246465605315871, -1.2388192137208685])
y_var = np.array([0.0, 0.11167512690355386, 0.3248730964467006, 0.7512690355329954, 
     1.4365482233502542, 2.0507614213197973, 2.68020304568528, 3.0050761421319807]) * e_recoil / e_perp

f_var = interp1d(x_var, y_var, kind='cubic')
x_new_var = np.linspace(x_var[0], x_var[-1], 100)
y_new_var = f_var(x_new_var)

df_loaded = pd.read_csv(name)
df_loaded = pd.read_csv(f"results/ol_stability_npse.csv")
v0s = df_loaded["v0"].to_numpy()
gcs = df_loaded["g_c"].to_numpy()
f = interp1d(v0s, gcs, kind='cubic')
x_new = np.linspace(v0s[0], v0s[-1], 100)
y_new = f(x_new)

plt.figure(figsize=(4, 3))

plt.plot(y_new, x_new  * e_perp / e_recoil, label="NPSE")
plt.scatter(gcs, v0s * e_perp / e_recoil , marker="+")

plt.plot(x_new_var, y_new_var  * e_perp / e_recoil, ls = "--", label="GVA")
plt.scatter(x_var, y_var * e_perp / e_recoil, marker="o")
plt.legend()
plt.ylabel(r"$V_0 \ [E_r]$")
plt.xlabel(r"$g_c$")
plt.axvline(-1.333, color="gray", linestyle="--"  , lw=0.5)
plt.axvline(-1.333/2, color="gray", linestyle=":", lw=0.5)
plt.axhline(1.4, ls="-.", color="red")
plt.axhline(1.4/2, ls="--", lw=0.8, color="red")
plt.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.7)  # Thin major grid
plt.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
plt.tight_layout()
plt.savefig(f"media/ol_stability.png", dpi=400)
print("Saved plot in media/ol_stability.png")


plt.figure(figsize=(6.3, 3.3))
colors = ["blue", "green", "orange", "purple"]
for ix, N in enumerate([1000, 1400, 1800, 2200]):
  plt.plot(1/(2 * N * a0_l_perp) * y_new, x_new  * e_perp / e_recoil , label=rf"$N_0 =$ {N}", color=colors[ix])
  plt.scatter(1/(2 * N * a0_l_perp) * gcs, v0s * e_perp / e_recoil , marker="+"             , color=colors[ix])
  plt.plot(1/(2 * N * a0_l_perp) * x_new_var, y_new_var  * e_perp / e_recoil, ls="--"       , color=colors[ix], lw = 0.8)
  plt.scatter(1/(2 * N * a0_l_perp) * x_var, y_var * e_perp / e_recoil , marker="o"         , color=colors[ix], lw = 0.8, 
              s=11)

plt.legend()
plt.ylabel(r"$V_0 \ [E_r]$")
plt.xlabel(r"$a_{s, c}$")
plt.xlim([-20, 0])
plt.title("Static collapse (NPSE)")
plt.axhline(1.4, ls="-.", color="red")
plt.axhline(1.4/2, ls="--", lw=0.8, color="red")
# plt.axhline(-1.333, color="red", linestyle="--")
# plt.axhline(-1.333/2, color="gray", linestyle=":")
plt.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.7)  # Thin major grid
plt.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
plt.tight_layout()
plt.savefig(f"media/ol_stability_as.png", dpi=400)
print("Saved plot in media/ol_stability_as.png")