from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import p1d_dyn_heatmap
import p3d_snap_projections
import launch.rust_launcher as rust_launcher


def exec(sim):
    try:
        # p3d_snap_projections.plot_projections()
        # p1d_dyn_heatmap.plot_heatmap()
        sim.compile("debug")
        sim.run(realtime=True)
        p1d_dyn_heatmap.plot_first_last()
    except Exception as e:
        print(e)


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_modified(self, event):
        if event.is_directory:
            return

        print("="*50)
        print(f"File {event.src_path} has been modified.")
        self.callback(event.src_path)


if __name__ == "__main__":
    sim = rust_launcher.Simulation(input_params="input/_params.toml",
                                   output_file="results/",
                                   rust="./target/debug/rust_waves")
    exec(sim)
    event_handler = FileChangeHandler(exec)
    observer = Observer()
    observer.schedule(event_handler, path=sim.input_params, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()