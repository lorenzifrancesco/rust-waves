import launch.rust_launcher as rust_launcher
import os
import time
import p1d_dyn_heatmap
import p3d_snap_projections
import plot_axial_density
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import numpy as np
from launch.rw import write_from_experiment
import re

"""
x in lattice sites,
y in arb units
"""
def get_axial_density_csv(filename):
    df = pd.read_csv(filename)
    df = df.sort_values(by='x')
    x = df.iloc[:, 0]
    y_values = df.iloc[:, 1:]
    window_size = 1
    y_smoothed = y_values.rolling(window=window_size, center=True).mean()
    return np.array(x), np.array(y_smoothed).ravel()


def after_run(l,
              filename, cf):
    name = re.search(r"dyn_(.+?)\.h5", filename).group(1)
    print(">> after_run: ", name)
    if l.dimension == 1:
        # print(">> plotting heatmap")
        # p1d_dyn_heatmap.plot_heatmap_h5(
        #     filename=filename)
        print(">> plotting axial density")
        fig, ax = plot_axial_density.init_plotting()
        plot_axial_density.plot_1d_axial_density(
            fig, ax, name_list=[name],)

        line = ax.get_lines()[0]
        y = line.get_ydata()
        y_max = max(y)
        x, y = get_axial_density_csv("input/3c-multisoliton.csv")
        peaks, _ = find_peaks(y)
        peak_index = peaks[np.argmax(y[peaks])]
        peak_x = x[peak_index]
        y_floor = (y[0]+y[-1])/2
        peak_y = y[peak_index] - y_floor
        level = y_max
        yfix = level
        x_shift = -peak_x
        x = x + x_shift
        y = (y-y_floor) / peak_y * level
        ax.plot(x * cf["physics"]["dl"], y,
                label="3c-multisoliton", color="red", ls="-")
        plt.xlim([-6, 6])
        filename = "input/3c-multisoliton.csv"
        plt.savefig("media/axial-test.pdf", dpi=900)
        print("Saved media/axial-test.pdf")
    else:
        p1d_dyn_heatmap.plot_heatmap_h5_3d(
            filename=filename)
        # p3d_snap_projections.movie(name="dyn_test_3d")
        # fig, ax = plot_axial_density.init_plotting()
        # plot_axial_density.plot_3d_axial_density(fig, ax, name_list=["test_3d"], color="blue", ls="-")
        # plt.savefig("media/test.pdf", dpi=900)


def continuously_update_screen():
    try:
        last_mtime = 0
        while True:
            current_mtime = os.path.getmtime('input/experiment_fig3c.toml')
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                os.system('clear')
                write_from_experiment(
                    input_filename="input/experiment_fig3c.toml",
                    output_filename="input/_params.toml",
                    title="fig3c",
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
                filename = "results/dyn_fig3c_1d.h5"
                # print("Dimension: ", l.dimension)
                assert (l.dimension == 1)
                l.compile("release")

                l.run()

                after_run(l, filename, l.cf)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopped by user")

if __name__ == "__main__":
    continuously_update_screen()