import launch.rust_launcher as rust_launcher
import os
import time
import projections_evolution
import projections_volumetric
import plot_axial_density
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import numpy as np
from launch.rw import write_from_experiment
import re
from plot_axial_density import get_available_filename
import pandas as pd

EXPERIMENT_FILE = "input/experiment_fig3c.toml"

"""
x in lattice sites,
y in arb units
"""
def get_axial_density_csv(filename):
    df = pd.read_csv(filename)
    df = df.sort_values(by='x')
    x = df.iloc[:, 0]
    y_values = df.iloc[:, 1:]
    window_size = 5
    y_smoothed = y_values.rolling(window=window_size, center=True).mean()
    return np.array(x), np.array(y_smoothed).ravel()


def after_run(l,
              filename_list, cf):
    name_list = [re.search(r"dyn_(.+?)\.h5", ff).group(1) for ff in filename_list]
    fig, ax = plot_axial_density.init_plotting()
    if l.dimension == 1:
        plot_axial_density.plot_1d_axial_density(
            fig, ax, name_list=name_list)
    else:
        fig, ax = plot_axial_density.plot_3d_axial_density(
            fig, ax, name_list=name_list,)

    line = ax.get_lines()[0]
    y = line.get_ydata()
    y_max = max(y)
    finals = ["3c-multisoliton", "3c-disperse"]
    colors = ["blue", "red"]
    for idx, final in enumerate(finals):
        x, y = get_axial_density_csv("input/"+final+".csv")
        valid = ~np.isnan(y)
        x = x[valid]
        y = y[valid]
        if idx == 0:
            peaks, _ = find_peaks(y)
            peak_index = peaks[np.argmax(y[peaks])]
            peak_x = x[peak_index]
            y_floor = (y[0]+y[-1])/2
            peak_y = y[peak_index]
            level = y_max
            print(level)
        x_shift = -peak_x
        x = x + x_shift
        y = (y-y_floor) / (peak_y-y_floor) * level
        ax.plot(x, y,
                label="3c-multisoliton", ls=":", lw=0.9, color=colors[idx])
        df = pd.DataFrame({
        'z [dl]':  x,
        'n [AU]' : y
        })
        df.to_csv("results/export/"+final+".csv", index=False)
    plt.xlim([-6, 6])
    plot_name = "media/axial-density.pdf"
    pn = get_available_filename(plot_name)
    plt.savefig(pn, dpi=900)
    print("Saved "+pn)

    for filename in filename_list:
        if l.dimension == 1:
            projections_evolution.plot_heatmap_h5(
                filename=filename, experiment_file=EXPERIMENT_FILE)
        else:
            projections_evolution.plot_heatmap_h5_3d(
                filename=filename, experiment_file=EXPERIMENT_FILE)


def continuously_update_screen():
    try:
        last_mtime = 0
        while True:
            current_mtime = os.path.getmtime(EXPERIMENT_FILE)
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                os.system('clear')
                write_from_experiment(
                    input_filename=EXPERIMENT_FILE,
                    output_filename="input/_params.toml",
                    title="fig3c-blue",
                    load_gs=True,
                    t_imaginary=None,
                    free_x=True,
                    a_s=None,
                    g=None,
                    v_0=None
                )
                l = rust_launcher.Simulation(
                    input_params="input/_params.toml",
                    output_file="results/",
                    rust="./target/release/rust_waves")
                filename = "results/dyn_" + \
                    l.cf["title"]+"_"+str(int(l.dimension))+"d.h5"
                # filename = "results/dyn_fig3c_1d.h5"
                # print("Dimension: ", l.dimension)
                l.compile("release")

                # l.run()

                filename_red =  "results/dyn_fig3c-red_"+str(int(l.dimension))+"d.h5"
                filename_blue = "results/dyn_fig3c-blue_"+str(int(l.dimension))+"d.h5"
                # assert(filename in [filename_red, filename_blue])
                # after_run(l, filename_red, l.cf)
                after_run(l, [filename_blue, filename_red], l.cf)
                # after_run(l, [filename], l.cf)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopped by user")

if __name__ == "__main__":
    continuously_update_screen()