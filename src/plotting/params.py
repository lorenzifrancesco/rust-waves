import launch.rust_launcher as rust_launcher
import os
import time
import p1d_dyn_heatmap
import p3d_snap_projections
import plot_axial_density
import matplotlib.pyplot as plt

def after_run(l):
  if l.dimension == 1:
    print(">> plotting heatmap")
    p1d_dyn_heatmap.plot_heatmap_h5(
        filename="results/dyn_pre-quench_1d.h5")
    fig, ax = plot_axial_density.init_plotting()
    plot_axial_density.plot_1d_axial_density(
        fig, ax, name_list=["pre-quench_1d"],)
    plt.xlim([-3, 3])
    plt.savefig("media/pre-axial.pdf", dpi=900)
    print("Saved media/pre-axial.pdf")
  else:
    p1d_dyn_heatmap.plot_heatmap_h5_3d(
        name="dyn_pre-quench_3d")
    # p3d_snap_projections.movie(name="dyn_test_3d")
    # fig, ax = plot_axial_density.init_plotting()
    # plot_axial_density.plot_3d_axial_density(fig, ax, name_list=["test_3d"], color="blue", ls="-")
    # plt.savefig("media/test.pdf", dpi=900)
  print("Done.")
            
      
                  
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
                print("Dimension: ", l.dimension)
                
                l.compile("release")
                
                # l.run()
                
                after_run(l)
                
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopped by user")


if __name__ == "__main__":
    continuously_update_screen()
