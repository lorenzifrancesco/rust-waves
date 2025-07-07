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
    if l.dimension == 1:
        # print(">> plotting heatmap")
        # p1d_dyn_heatmap.plot_heatmap_h5(
        #     filename=filename)
        print(">> plotting axial density")
        fig, ax = plot_axial_density.init_plotting()
        plot_axial_density.plot_1d_axial_density(
            fig, ax, name_list=["3c_1d"],)

        x, y = get_axial_density_csv("input/3c-multisoliton.csv")
        peaks, _ = find_peaks(y)
        peak_index = peaks[np.argmax(y[peaks])]
        peak_x = x[peak_index]
        peak_y = y[peak_index]
        level = 0.35
        yfix = level
        x_shift = -peak_x
        x = x + x_shift
        y = y / peak_y * level
        y_floor = (y[0]+y[-1])/2 - 0.01
        y = y-y_floor
        ax.plot(x * cf["physics"]["dl"], y,
                label="3c-multisoliton", color="red", ls="-")
        plt.xlim([-6, 6])
        filename = "input/3c-multisoliton.csv"
        plt.savefig("media/pre-axial.pdf", dpi=900)
        print("Saved media/pre-axial.pdf")
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
            current_mtime = os.path.getmtime('input/params.toml')
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                os.system('clear')

                l = rust_launcher.Simulation(
                    input_params="input/params.toml",
                    output_file="results/",
                    rust="./target/release/rust_waves")
                filename = "results/dyn_" + \
                    l.cf["title"]+"_"+str(int(l.dimension))+"d.h5"
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
