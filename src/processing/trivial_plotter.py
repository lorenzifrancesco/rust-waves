import pandas as pd
import matplotlib.pyplot as plt

filename = "input/3c-multisoliton.csv"  # Change this to your actual file path
df = pd.read_csv(filename)

x = df.iloc[:, 0]
y_values = df.iloc[:, 1:]

# Apply rolling mean smoothing to each column in y_values
window_size = 3  # You can change this value (odd numbers like 5, 7, 9 work well)
y_smoothed = y_values.rolling(window=window_size, center=True).mean()

plt.figure(figsize=(8, 6))
for column in y_smoothed:
    plt.plot(x, y_smoothed[column], label=column)

plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Smoothed Plot from CSV Data")
plt.legend()
plt.grid(True)

plt.show()
