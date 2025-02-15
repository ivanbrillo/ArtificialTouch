import matplotlib.pyplot as plt
import pandas as pd
import random
import glob

# Number of random curves you want to plot
N = 50  # You can adjust this number

# Get list of all CSV files
csv_files = glob.glob('Dataset/20250205_082609_HIST_006_CPXE_*.csv')

# Randomly select N files
selected_files = random.sample(csv_files, N)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each selected file
for file in selected_files:
    df = pd.read_csv(file)

    mask = df['isTouching_SMAC'] == 1
    df.loc[mask, 'CPXEts'] = df.loc[mask, 'CPXEts'] - df.loc[mask, 'CPXEts'].min()
    touching_rows = df[mask]

    touching_rows.plot(x='CPXEts', y='Fz',
                      color='gray',        # Set color to gray
                      alpha=1,           # Add some transparency
                      linewidth=0.4,       # Very thin lines
                      ax=ax,
                      legend=False)        # Don't show legend for each line

plt.xlabel('CPXEts')
plt.ylabel('Fz (Force)')
plt.title(f'{N} Random Force Curves')
plt.show()