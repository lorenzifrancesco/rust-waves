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
from scipy.optimize import minimize
from scipy.interpolate import interp1d

"""
Data processing from final snapshots to get the width and particle fraction  
"""
def width_from_wavefunction(title, 
                            dimensions=1, 
                            harmonium=False, 
                            case="", 
                            particle_threshold=0.3):
    filename = "".join(["results/", title,"_", str(dimensions), "d.h5"])
    # print("Computing wavefunction for ", filename)
    params = Params.read("input/params.toml")
    # we sum only on [-4, +4] lattice sites.
    min_idx = params.dl * -4
    max_idx = -min_idx
    if dimensions == 1:
        with h5py.File(filename, "r") as f:
            l = np.array(f["l"])
            final_psi2 = np.array(f["psi_squared"])
        mask = (l >= min_idx) & (l <= max_idx)
        # print(mask)
        dz = l[1] - l[0]
        particle_fraction = np.sum(final_psi2[mask]) * dz
        if particle_fraction < particle_threshold:
          particle_fraction = np.nan
        # print(f"particle fraction = {particle_fraction}")
        center = dz * np.sum(l[mask] * final_psi2[mask]) / particle_fraction
        
        if harmonium:
          min_idx = params.dl * -4
          max_idx = -min_idx
          std = 0.0
          for site in range(-4, 5):
            lower_end = (site - 0.5) * params.dl
            upper_end = (site + 0.5) * params.dl
            mask = (l >= lower_end) & (l <= upper_end)
            n_i = np.sum(dz * final_psi2[mask]) / particle_fraction
            std += site**2 * n_i
            # print(f"Site = {site:3d}, [{lower_end:3.2f}, {upper_end:3.2f} ] n_site ={n_i:3.2f} ")
          std = np.sqrt(std - center**2)
        else:
          std = np.sqrt(dz * 
                      np.sum(
                        l[mask]**2 * final_psi2[mask]
                        )/particle_fraction - center**2)
        # now we overwrite with the total particle fraction
        particle_fraction = np.sum(final_psi2) * dz
        # print(f"\n center = {center:3.2e}, std = {std:3.2e} l_perp\n")
    else:
        with h5py.File(filename, "r") as f:
            l_x = np.array(f["l_x"])
            l_y = np.array(f["l_y"])
            l_z = np.array(f["l_z"])
            psi_squared = np.array(f["psi_squared"])
        mask = (l_x >= min_idx) & (l_x <= max_idx)
        # print(mask)
        dx = l_x[1] - l_x[0]
        dy = l_y[1] - l_y[0]
        dz = l_z[1] - l_z[0]
        dV = dx * dy * dz
        particle_fraction = np.sum(psi_squared[mask, :, :]) * dV
        if particle_fraction < particle_threshold:
          particle_fraction = np.nan
        # print(f"particle fraction = {particle_fraction}")
        center = np.sum(l_x[mask, None, None] * psi_squared[mask, :, :]) * dV / particle_fraction
        
        if harmonium:
          min_idx = params.dl * -4
          max_idx = -min_idx
          std = 0.0
          for site in range(-4, 5):
            lower_end = (site - 0.5) * params.dl
            upper_end = (site + 0.5) * params.dl
            mask = (l_x >= lower_end) & (l_x <= upper_end)
            n_i = np.sum(dV * psi_squared[mask, :, :]) / particle_fraction
            std += site**2 * n_i
            # print(f"Site = {site:3d}, [{lower_end:3.2f}, {upper_end:3.2f} ] n_site ={n_i:3.2f} ")
          std = np.sqrt(std - center**2)
        else:
          std = np.sqrt(dV *
                  np.sum(
                    l_x[mask, None, None]**2 * psi_squared[mask, :, :]
                    ) / particle_fraction - center**2)
        # now we overwrite with the total particle fraction
        particle_fraction = np.sum(psi_squared)* dV
        # print(f"\n center = {center:3.2e}, std = {std:3.2e} l_perp\n")

    if np.isnan(particle_fraction):
        particle_fraction = 0
    return particle_fraction, std/params.dl


def apply_noise_to_widths(w, l, noise_atoms, n_atoms):
    return (w*n_atoms+1/12*l**2*noise_atoms)/(n_atoms+noise_atoms)


def optimize_widths(noise, file):
    data = pd.read_csv("input/widths.csv", names=["a_s", "width", "number"])
    try:
        data_1 = pd.read_csv(file, header=0, names=[
                             "a_s", "width", "width_sim", "width_rough", "particle_fraction"])
    except:
        print("No data!")
    cf = toml.load("input/experiment.toml")
    noise_atoms = cf["n_atoms"] * noise
    n_atoms = cf["n_atoms"]
    widths_noise = apply_noise_to_widths(
        data_1["width_sim"], 8, noise_atoms, n_atoms)
    mse = np.sum((data["width"]-widths_noise)**2)/len(data_1)
    return mse


def plot_widths_cumulative(noise=0.0,
                           cases=[None],
                           a_s_limit=-40.0):
    data = pd.read_csv("input/widths.csv", names=["a_s", "width", "number"])
    data_list = []
    labels = []
    plt.figure(figsize=(3.6, 3))
    # Extract columns
    a_s = data["a_s"]  # First column as x-axis
    width = data["width"]  # Second column as y-axis
    # number = data["number"]
    # Create the plot
    plt.plot(a_s, width, marker='+', linestyle='-', lw=0.5,
            color='b', label='experiment')
    # cm = cm.viridis
    # colors = 
    for idx, cs in enumerate(cases):
      case = "_"+str(cs)
      # try:
      #     data_1 = pd.read_csv("results/widths/widths_final_1d"+case+".csv", header=0, names=[
      #                         "a_s", "width", "width_sim", "width_rough", "particle_fraction"])
      #     data_list.append(data_1)
      #     labels.append("1D"+case)
      # except:
      #     print("No 1D data")
      # try:
      #     data_npse = pd.read_csv("results/widths/widths_final_npse"+case+".csv", header=0, names=[
      #                             "a_s", "width", "width_sim", "width_rough", "particle_fraction"])
      #     data_list.append(data_npse)
      #     labels.append("NPSE"+case)
      # except:
      #     print("No NPSE data")
      try:
          data_3 = pd.read_csv("results/widths/widths_final_3d"+case+".csv", header=0, names=[
                              "a_s", "width", "width_sim", "width_rough", "particle_fraction"])
          data_list.append(data_3)
          labels.append("3D"+case)
      except:
          print("No 3D data")

  
    linestyle = ['-.', ':', '-', ':'] * len(cases)
    linestyle = ['--', '--', '--', '--'] * len(cases)

    cf = toml.load("input/experiment.toml")
    n_atoms = cf["n_atoms"]
    # if noises is not None:
        # noise = 0.0
    # print(f"applying the noise of ", noise)
    noise_atoms = n_atoms * noise
    l = 8  # lattice sites
    # print(data_list)
    # FIXME, I should use the right number of a_s
    all_widths = np.zeros((len(data_list), len(a_s)-1))
    all_fractions = np.zeros((len(data_list), len(a_s)-1))
    print(all_widths.shape)
    for i, data in enumerate(data_list):
        print(f"Case: {i}")
        width = np.array(data["width_sim"])
        a_s = data["a_s"]
        fraction = data["particle_fraction"]
        # if noises is not None:
        #     noise_atoms = n_atoms * noises[i]
        print(f"num {i}, width last {width[-1]}")
        width = apply_noise_to_widths(width, l, noise_atoms, n_atoms)
        # print("before: \n ", width)
        # print("after: \n ", width)
        sketchy = 1
        if i == 2:
          sketchy = 1
        print("len", len(width), " pigeon ", len(all_widths[0, :]))
        all_widths[i, :] = width
        all_fractions[i, :] = fraction
        plt.scatter(a_s*sketchy, width, s=2, marker='x')
        # possible = ~np.isnan(width)
        possible = ~np.isnan(width) & (a_s >= a_s_limit)
        a_s_filtered = a_s[possible]
        print("a_s_filtered", a_s_filtered)
        a_s_resampled = np.linspace(a_s_filtered.min(), a_s_filtered.max(), 100)
        width = interp1d(a_s[possible], width[possible], kind='cubic')(a_s_resampled)
        plt.plot(a_s_resampled*sketchy, width,
                  label=labels[i],
                  linestyle=linestyle[i],
                  lw=0.7)
    
    
    min_width = np.min(all_widths, axis=0)
    max_width = np.max(all_widths, axis=0)
    min_fraction = np.min(all_fractions, axis=0)
    max_fraction = np.max(all_fractions, axis=0)
    
    avg_width = np.mean(all_widths, axis=0)
    a_s_resampled = np.linspace(a_s_filtered.min(), a_s_filtered.max(), 50)
    
    min_width = interp1d(a_s, min_width, kind='quadratic')(a_s_resampled)
    plt.plot(a_s_resampled, min_width)
    
    max_width = interp1d(a_s, max_width, kind='quadratic')(a_s_resampled)
    plt.plot(a_s_resampled, max_width)
    
    avg_width = interp1d(a_s, avg_width, kind='quadratic')(a_s_resampled)
    plt.plot(a_s_resampled, avg_width, ls="-.")
    
    min_fraction = interp1d(a_s, min_fraction, kind='quadratic')(a_s_resampled)
    max_fraction = interp1d(a_s, max_fraction, kind='quadratic')(a_s_resampled)
    plt.plot(a_s_resampled, min_fraction, ls=":", color="red")
    plt.plot(a_s_resampled, max_fraction, ls=":", color="red")
    # plt.ylim([0, 1])
    df = pd.DataFrame({
        "a_s": a_s_resampled,
        "max_width": max_width,
        "min_width": min_width,
        "max_fraction": max_fraction,
        "min_fraction": min_fraction
    })
    df.to_csv(f"results/widths/cumulative.csv", index=False)
    print("Saved CSV to results/widths/cumulative.csv")
    plt.xlabel(r"$a_s/a_0$")
    plt.ylabel(r"$w_z$ [sites] ")
    plt.xlim([-21, 1.0])
    plt.tight_layout()
    # plt.legend(fontsize=8, labelspacing=0.2)
    plt.savefig("media//widths/widths_cumulative.pdf", dpi=300)
    print("Saved media//widths/widths_cumulative.pdf")
    
    
    # ## Plotting the particle fraction
    # plt.clf()
    # plt.figure(figsize=(3.6, 3))
    # data = pd.read_csv("input/widths.csv", names=["a_s", "width", "number"])
    # a_s = data["a_s"]  # First column as x-axis
    # width = data["width"]  # Second column as y-axis
    # # plt.plot(a_s, number/initial_number, 
    # #          marker='+', 
    # #          linestyle='-', 
    # #          lw = 0.5,
    # #          color='b', 
    # #          label='experiment')
    # for i, data in enumerate(data_list):
    #     fraction = data["particle_fraction"]  # Second column as y-axis
    #     a_s = data["a_s"] 
    #     sketchy = 1
    #     if i == 2:
    #       sketchy = 1
          
    #     plt.scatter(a_s*sketchy, fraction, s=2, marker='x')
    #     possible = ~np.isclose(fraction, 0.0)
    #     a_s_filtered = a_s[possible]
    #     a_s_resampled = np.linspace(a_s_filtered.min(), a_s_filtered.max(), 100)
    #     fraction = interp1d(a_s[possible], fraction[possible], kind='cubic')(a_s_resampled)
    #     plt.plot(a_s_resampled*sketchy, fraction,
    #               label=labels[i],
    #               linestyle=linestyle[i], 
    #               lw=0.7)
    #     # plt.scatter(a_s*sketchy, fraction, s=2, marker='x')
    #     # plt.plot(a_s * sketchy, fraction,
    #     #          linestyle=linestyle[i],
    #     #          label=labels[i], 
    #     #          lw=0.7)
    # plt.xlabel(r"$a_s/a_0$")
    # plt.ylabel(r"$N_{\mathrm{tot}}/N_0$")
    # plt.xlim([-21, 1.0])
    # plt.tight_layout()
    # plt.legend(fontsize=8, labelspacing=0.2)
    # plt.savefig("media/fraction/fraction"+case+".pdf", dpi=300)
    # print("Saved media/fraction/fraction"+case+".pdf")

"""
Plotting loading the CSV files with the widths and the particle fraction
"""
def plot_widths(noise=0.0,
                plot=False,
                initial_number=2000,
                eqs=["1d", "npse", "3d"],
                noises=None, 
                case=""):
    """
    Confrontation with the experimental data
    """
    data = pd.read_csv("input/widths.csv", names=["a_s", "width", "number"])
    data_list = []
    labels = []
    try:
        data_1 = pd.read_csv("results/widths/widths_final_1d"+case+".csv", header=0, names=[
                             "a_s", "width", "width_sim", "width_rough", "particle_fraction"])
        if "1d" in eqs:
            data_list.append(data_1)
        labels.append("1D")
    except:
        print("No 1D data")
    try:
        data_npse = pd.read_csv("results/widths/widths_final_npse"+case+".csv", header=0, names=[
                                "a_s", "width", "width_sim", "width_rough", "particle_fraction"])
        if "npse" in eqs:
            data_list.append(data_npse)
        labels.append("NPSE")
    except:
        print("No NPSE data")
    try:
        data_3 = pd.read_csv("results/widths/widths_final_3d"+case+".csv", header=0, names=[
                             "a_s", "width", "width_sim", "width_rough", "particle_fraction"])
        if "3d" in eqs:
            data_list.append(data_3)
        labels.append("3D")
    except:
        print("No 3D data")
    
    linestyle = ['-.', ':', '-', ':']
    # Extract columns
    a_s = data["a_s"]  # First column as x-axis
    width = data["width"]  # Second column as y-axis
    number = data["number"]
    # Create the plot
    plt.figure(figsize=(3.6, 3))
    plt.plot(a_s, width, marker='+', linestyle='-', lw=0.5,
             color='b', label='experiment')
    cf = toml.load("input/experiment.toml")
    n_atoms = cf["n_atoms"]
    if noises is not None:
        noise = 0.0
    # print(f"applying the noise of ", noise)
    noise_atoms = n_atoms * noise
    l = 8  # lattice sites
    for i, data in enumerate(data_list):
        width = data["width_sim"]
        a_s = data["a_s"]
        if noises is not None:
            noise_atoms = n_atoms * noises[i]

        width = apply_noise_to_widths(width, l, noise_atoms, n_atoms)
        # print("before: \n ", width)
        # print("after: \n ", width)
        sketchy = 1
        if i == 2:
          sketchy = 1
        if plot:
            plt.scatter(a_s*sketchy, width, s=2, marker='x')
            possible = ~np.isnan(width)
            a_s_filtered = a_s[possible]
            print("a_s_filtered = ", a_s_filtered)
            a_s_resampled = np.linspace(a_s_filtered.min(), a_s_filtered.max(), 100)
            width = interp1d(a_s[possible], width[possible], 
                             kind='linear', 
                             fill_value='extrapolate')(a_s_resampled)
            plt.plot(a_s_resampled*sketchy, width,
                     label=labels[i],
                     linestyle=linestyle[i], 
                     lw=0.7)
        else:
            mse = np.mean((width - data["width"])**2)
            return mse
    plt.xlabel(r"$a_s/a_0$")
    plt.ylabel(r"$w_z$ [sites] ")
    plt.xlim([-21, 1.0])
    plt.tight_layout()
    plt.legend(fontsize=8, labelspacing=0.2)
    if noises is not None:
        plt.savefig("media/widths/widths_optim"+case+".pdf", dpi=300)
        print("Saved media/widths/widths_optim"+case+".pdf")
    else:
        plt.savefig("media//widths/widths"+case+".pdf", dpi=300)
        print("Saved media//widths/widths"+case+".pdf")
        
    ## Plotting the particle fraction
    plt.clf()
    plt.figure(figsize=(3.6, 3))
    data = pd.read_csv("input/widths.csv", names=["a_s", "width", "number"])
    a_s = data["a_s"]  # First column as x-axis
    width = data["width"]  # Second column as y-axis
    # plt.plot(a_s, number/initial_number, 
    #          marker='+', 
    #          linestyle='-', 
    #          lw = 0.5,
    #          color='b', 
    #          label='experiment')
    for i, data in enumerate(data_list):
        fraction = data["particle_fraction"]  # Second column as y-axis
        a_s = data["a_s"] 
        sketchy = 1
        if i == 2:
          sketchy = 1
          
        plt.scatter(a_s*sketchy, fraction, s=2, marker='x')
        possible = ~np.isclose(fraction, 0.0)
        a_s_filtered = a_s[possible]
        a_s_resampled = np.linspace(a_s_filtered.min(), a_s_filtered.max(), 100)
        fraction = interp1d(a_s[possible], fraction[possible], kind='cubic')(a_s_resampled)
        plt.plot(a_s_resampled*sketchy, fraction,
                  label=labels[i],
                  linestyle=linestyle[i], 
                  lw=0.7)
        # plt.scatter(a_s*sketchy, fraction, s=2, marker='x')
        # plt.plot(a_s * sketchy, fraction,
        #          linestyle=linestyle[i],
        #          label=labels[i], 
        #          lw=0.7)
    plt.xlabel(r"$a_s/a_0$")
    plt.ylabel(r"$N_{\mathrm{tot}}/N_0$")
    plt.xlim([-21, 1.0])
    plt.tight_layout()
    plt.legend(fontsize=8, labelspacing=0.2)
    plt.savefig("media/fraction/fraction"+case+".pdf", dpi=300)
    print("Saved media/fraction/fraction"+case+".pdf")


if __name__ == "__main__":
    # print(width_from_wavefunction("idx-9", dimensions=1))
    # print(width_from_wavefunction("idx-9", dimensions=3))
    
    cases = np.linspace(1200, 2200, 10, dtype=int)
    # cases = cases[
    cases = [1700]
    plot_widths_cumulative(cases = cases)
    
    exit()
    plot_widths(0.0, 
                plot=True, 
                initial_number=3000)
    file_list = ["results/widths/widths_final_1d.csv",
                 "results/widths/widths_final_npse.csv",
                 "results/widths/widths_final_3d.csv"]
    noises = []
    for f in file_list:
        print("evaluating -> ", f)
        def foo(x): return optimize_widths(x, f)
        res = minimize(foo,
                       0.35,
                       method='nelder-mead',
                       options={'xatol': 1e-8, 'disp': True})
        print("Opt. Noise: ", res.x)
        print("MSE       : ", res.fun)
        noises.append(res.x)
        data_1 = pd.read_csv(f, header=0, names=[
            "a_s", "width", "width_sim", "width_rough", "particle_fraction"])
        cf = toml.load("input/experiment.toml")
        noise_atoms = cf["n_atoms"] * res.x
        ww = apply_noise_to_widths(
            data_1["width_sim"], 8, noise_atoms, cf["n_atoms"])
        print(f"Values    : ", ww)
    plot_widths(0.0,
                plot=True,
                initial_number = 1700,
                noises=noises)