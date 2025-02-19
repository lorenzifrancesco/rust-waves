from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import p1d_dyn_heatmap
import p3d_snap_projections

def exec(path):
  try:
    p3d_snap_projections.plot_projections()
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
    exec("")
    event_handler = FileChangeHandler(exec)
    observer = Observer()
    observer.schedule(event_handler, path='/home/lorenzi/bench/rust-waves/results', recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:   
        observer.stop()

    observer.join()