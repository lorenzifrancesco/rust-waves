import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv


def plot_csv(filename):
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
    plt.show()
    plt.savefig("media/initial.pdf")


# Example usage:
plot_csv('results/output.csv')


# with open('matrix.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     matrix = np.loadtxt('matrix.csv', delimiter=',')

#     # Create a heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(matrix, annot=True, cmap='viridis')

#     # Display the heatmap
#     plt.show()
