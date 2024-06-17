import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv


def plot_csv(filename):
    # Initialize lists to store the data
    x = []
    matrix = []

    # Read the CSV file
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
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap='viridis', xticklabels=x)

    # Add titles and labels
    plt.title('Heatmap from CSV Matrix')
    plt.xlabel('Space')
    plt.ylabel('n_saves')

    # Display the heatmap
    plt.tight_layout()
    plt.show()
    plt.savefig("media/heatmap.pdf")


# Example usage:
plot_csv('results/output.csv')
