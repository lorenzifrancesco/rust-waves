import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv

def plot_heatmap(filename):
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
    sns.heatmap(matrix, annot=False, cmap='viridis', xticklabels=x, zorder=3)

    # Add titles and labels
    plt.title('Heatmap from CSV Matrix')
    plt.xlabel('Space')
    plt.ylabel('n_saves')

    # Display the heatmap
    plt.tight_layout()
    plt.grid(False, zorder=0)
    plt.savefig("media/heatmap.png")
    
def plot_final(filename):
    x = []
    y = []

    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header row if there is one
        # for row in csvreader:
        #     x.append(float(row[0]))
        #     y.append(float(row[1]))
        for row in csvreader:
            x.append(float(row[0]))
            try:
                # Replace 'i' with 'j' for complex parsing
                complex_num = complex(row[1].replace('i', 'j'))
                y.append(complex_num)
            except ValueError:
                if row[1].strip().lower() == 'NaN':
                    y.append(np.nan)
                elif row[1].strip().lower() == 'inf':
                    y.append(np.nan)
                else:
                    print(f"Unknown value encountered: {row[1]}")
                    y.append(np.nan)
                
    plt.figure(figsize=(8, 6))
    plt.plot(x, np.abs(y)**2, marker='+',
             linestyle='-', color='b', label='Data')
    plt.title('Plot from CSV File')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("media/final.pdf")


# Example usage:
print("____ Plotting _____")
plot_heatmap('results/output.csv')


# with open('matrix.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     matrix = np.loadtxt('matrix.csv', delimiter=',')

#     # Create a heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(matrix, annot=True, cmap='viridis')

#     # Display the heatmap
#     plt.show()
