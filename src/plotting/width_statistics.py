import pandas as pd
import matplotlib.pyplot as plt

width_files = []
atom_numbers = [1200, 1700, 2200]
l3s = [5e-38, 1e-38, 5e-39]
for i in [1200, 1700, 2200]:
    for j in l3s:
        width_files.append(f"results-{j}/widths/widths_final_3d_{i}.csv")

base = "input/widths.csv"  # Change this to your actual file path
df = pd.read_csv(base)
print(df)
# Extract x values (first column) and y values (other columns)
x = df.iloc[:, 0]
print(x)
y_values = df.iloc[:, 1]
print(y_values)

# Plot all y columns
plt.figure(figsize=(4, 3))
plt.plot(x, y_values, label=base.split("/")[-1])
for ww in width_files:
    try:
        df = pd.read_csv(ww)
        # Extract x values (first column) and y values (other columns)
        x = df["a_s"]
        y_values = df["width_sim"]
        plt.plot(x, y_values, label=ww.split("/")[-1])
    except:
        print(f"File {ww} not found or empty.")

plt.xlabel("a_s [a_0]")
plt.ylabel("width [sites]")
# plt.legend()
plt.grid(True)
plt.tight_layout()

name = "media/combinazzio.pdf"
plt.savefig(name)
print("Saved figure as media/combinazzio.pdf")