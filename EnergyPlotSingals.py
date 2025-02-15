import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import glob


def calculate_energy(df):
    touching_rows = df[df['isTouching_SMAC'] == 1]
    # Energy is the sum of squared values
    energy = np.sum(touching_rows['Fz'] ** 2)
    x = df['posx'].iloc[0]
    y = df['posy'].iloc[0]
    return x, y, energy


# Get list of all CSV files
csv_files = glob.glob('Dataset/20250205_082609_HIST_006_CPXE_*.csv')

# Calculate energy for each file
energies = []
for file in csv_files:
    df = pd.read_csv(file)
    x, y, energy = calculate_energy(df)

    if 20 < x < 110 and 20 < y < 140:
        energies.append((x, y, energy))

# Convert to numpy array
energies = np.array(energies)
print(energies.shape)

# Method 1: Multiply by -1 (inverts the peaks)
energies[:, 2] = -energies[:, 2]

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
trisurf = ax.plot_trisurf(energies[:, 0], energies[:, 1], energies[:, 2],
                          cmap='viridis',
                          edgecolor='none',  # Removes grid lines for a smoother surface look
                          alpha=0.8)

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Inverted Energy')
fig.colorbar(trisurf, label='Inverted Energy')
plt.title('Signal Energy Distribution (Inverted)')
plt.show()
