import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
filename = "results/transverse.csv"  # Change this to your actual file path
df = pd.read_csv(filename)

# Extract x values (first column) and y values (other columns)
x = df.iloc[:, 0]
y_values = df.iloc[:, 1:]

# Plot all y columns
plt.figure(figsize=(8, 6))
for column in y_values:
    plt.plot(x, y_values[column], label=column)

# Labeling
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Plot from CSV Data")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()