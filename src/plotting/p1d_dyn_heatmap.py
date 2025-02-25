import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import h5py
from matplotlib import cm

def plot_heatmap_h5(filename="results/1d.h5", i=-1):
  with h5py.File(filename, "r") as f:
      l = np.array(f["l"])  # Load l (spatial coordinate)
      t = np.array(f["t"])  # Load t (time coordinate)
      psi_squared = np.array(f["psi_squared"]).reshape(len(t), len(l))  # Load psi_squared dataset

  plt.figure(figsize=(3, 3))
  extent = [t.min(), t.max(), l.min(), l.max()]
  aspect = (t.max() - t.min()) / (l.max() - l.min())

  plt.imshow(psi_squared.T, extent=extent, origin="lower", aspect=aspect, cmap="gist_ncar")
  plt.colorbar(label=r"$|\psi|^2$")
  plt.xlabel(r"$t$")
  plt.ylabel(r"$x$")
  plt.tight_layout()
  plt.savefig(f"media/idx-{i}_heatmap.png", dpi=600)

def plot_heatmap(filename):
    # Initialize lists to store the data
    x = []
    matrix = []

    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip header row
        # Initialize the matrix with empty lists for each column in psi
        num_columns = len(header) - 1

        # Read the rest of the rows
        for row in csvreader:
            x.append(float(row[0]))  # The first column is l_range
            row_data = []
            for col in range(1, num_columns + 1):
                try:
                    complex_num = complex(row[col].replace('i', 'j'))
                    row_data.append(abs(complex_num)**2)
                except ValueError:
                    if row[col].strip().lower() == 'nan':
                        row_data.append(np.nan)
                    elif row[col].strip().lower() == 'inf':
                        row_data.append(np.nan)
                    else:
                        print(f"Unknown value encountered: {row[col]}")
                        row_data.append(np.nan)
            matrix.append(row_data)

    # Convert matrix to numpy array for easier plotting
    matrix = np.array(matrix).T  # Transpose to get the correct orientation

    # Create a heatmap
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, aspect='auto', cmap='viridis', origin='lower', zorder=2)
    # sns.heatmap(matrix, annot=False, cmap='viridis', xticklabels=x, zorder=3)

    # Add titles and labels
    plt.xlabel('Space')
    plt.ylabel('n_saves')

    # Display the heatmap
    plt.tight_layout()
    plt.grid(False, zorder=0)
    plt.savefig("media/heatmap.png", dpi=600)
    
def plot_final(filename, ax, ix):
    x = []
    y = []
    viridis = cm.get_cmap('viridis')
    with h5py.File(filename, 'r') as hdf:
        # Assuming the datasets are named 'l' and 'field'
        x = np.array(hdf['l'])
        y_data = np.array(hdf['psi_squared'])
        if len(x) == 0:
          x = np.linspace(0, len(y_data), len(y_data))
        # Process `field` values to handle potential complex and special cases
        for value in y_data:
            try:
                # Convert to complex if necessary
                complex_num = complex(str(value).replace('i', 'j'))
                y.append(complex_num)
            except ValueError:
                if str(value).strip().lower() == 'nan':
                    y.append(np.nan)
                elif str(value).strip().lower() == 'inf':
                    y.append(np.nan)
                else:
                    print(f"Unknown value encountered: {value}")
                    y.append(np.nan)

    ax.plot(x, y, linestyle='-', color=viridis(
      ix/1.5), lw=0.5)
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'$|\psi|^2$')
    return ax

def plot_first_last():
  plt.figure(figsize=(3, 2))
  ax = plt.gca()
  for ix, name in enumerate(["1d_psi_0.h5", "1d_psi_end.h5"]):
    plot_final('results/'+name, ax, ix)
  plt.tight_layout()
  plt.savefig("media/1d_first_last.png", dpi=600)
 
def plot_snap(filename, i=-1):
  plt.figure(figsize=(3, 1.5))
  ax = plt.gca()
  plot_final(filename, ax, 0)
  plt.tight_layout()
  plt.savefig(f"media/idx-{i}_snap.png", dpi=600)
   
    
if __name__ == "__main__":
  print("____ Plotting _____")
  # plot_heatmap_h5()
  plot_first_last()
  # plot_heatmap('results/1d_psi.h5')
