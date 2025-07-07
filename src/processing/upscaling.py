import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

input_file =  "results/widths/0.csv"
output_file = "results/widths/colormap_1_v2.csv"

l_perp=1.591101125e-6 

matrix = np.loadtxt(input_file, delimiter=",")
scaled_matrix = zoom(matrix, zoom=(2, 6), order=2)
pd.DataFrame(scaled_matrix).to_csv(output_file, index=False, header=False)

dz = 2.4909606655503284e-07

plt.imshow(scaled_matrix)
plt.show()

plt.plot(range(len(scaled_matrix[0, :])), dz * np.sum(scaled_matrix, axis=0))
plt.show()