import numpy as np
from dataclasses import dataclass
import toml
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
import csv
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, ListedColormap
from scipy.constants import lambda2nu, Boltzmann, c, h
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colorbar import Colorbar

class Simulation:
  input_params: str
  output_file: str
  rust: str
  rust_mode: str 
  dimension: int

  def __init__(self, input_params, output_file, rust, rust_mode="release", dimension=3):
    self.input_params = input_params
    self.output_file = output_file
    self.rust = rust
    self.rust_mode = rust_mode
    self.dimension = dimension
 
  def compile(self, rust_mode):
    if rust_mode == "debug":
      rust_compile = 'cargo build'
    else:
      rust_compile = 'cargo build --'+rust_mode
    print("Compiling...")
    result = subprocess.run(rust_compile, shell=True,
                            capture_output=True, text=True)
    if result.returncode != 0:
      print("!! problems in the compilation")
    print("Done.")

  def run(self):
    print("Running...")
    result = subprocess.run(
            [self.rust], text=True, stdout=subprocess.PIPE, capture_output=False)
    print("Done.")
    return
  
if __name__ == "__main__":
  l = Simulation(input_params="input/params.toml",
                 output_file="results/",
                 rust="./target/debug/rust_waves")
  l.compile("debug")
  l.run()
  print("Done.")