import rust_launcher
import os
import time
import p1d_dyn_heatmap
def continuously_update_screen():
    try:
        last_mtime = 0
        while True:
            current_mtime = os.path.getmtime('input/params.toml')
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                os.system('clear')
                l = rust_launcher.Simulation(input_params="input/params.toml", 
                                   output_file="results/",
                                   rust="./target/debug/rust_waves")
                l.compile("debug")
                l.run()
                print(">> plotting first-last")
                p1d_dyn_heatmap.plot_first_last()
                print(">> plotting heatmap")
                p1d_dyn_heatmap.plot_heatmap_h5()
                print("Done.")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopped by user")


if __name__ == "__main__":
    continuously_update_screen()
