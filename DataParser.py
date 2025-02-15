import matplotlib.pyplot as plt
import pandas as pd

# Display all columns side by side
pd.set_option('expand_frame_repr', False)

df = pd.read_csv('Dataset/20250205_082609_HIST_006_CPXE_1.csv')
print(df.head())

# Create figure and primary axis (for Fz)
fig, ax1 = plt.subplots()

# Plot 'Fz' values with smaller markers
df.plot(x='CPXEts', y='Fz', color='blue', label='Not Touching', ax=ax1)
touching_rows = df[df['isTouching_SMAC'] == 1]
arrived_rows = df[df['isArrived_Festo'] == 1]

arrived_rows.plot(x='CPXEts', y='Fz', color='green', label='Arrived', ax=ax1)
touching_rows.plot(x='CPXEts', y='Fz', color='red', label='Touching', ax=ax1)

# Create a secondary y-axis for 'Position'
ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis

df.plot(x='CPXEts', y='posz2', color='purple', linestyle='dashed', label='Position', ax=ax2)

# Labels and legends
ax1.set_xlabel('CPXEts')
ax1.set_ylabel('Fz (Force)', color='blue')
ax2.set_ylabel('Position', color='purple')

# Adjust legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Fz and Position over Time')
plt.show()
