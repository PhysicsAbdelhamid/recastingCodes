import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Create a common background and data points
x = np.linspace(0, 10, 20)  # Keep the number of points lower
y_background = np.exp(-((x - 5)**2) / 4) * 0.1  # Reducing the amplitude of the background

# Extend x and y_background to start and end at zero
x_extended = np.concatenate(([0], x, [10]))
y_background_extended = np.concatenate(([0], y_background, [0]))

# Generate data points close to the background by adding small random noise
y_data = 1.5*np.exp(-((x - 5.5)**2) / 5) * 0.1

# Extend y_data to start and end at zero
y_data_extended = np.concatenate(([0], y_data, [0]))

# Smooth the data to avoid outliers
y_data_smooth = gaussian_filter1d(y_data_extended, sigma=1)

# Clip the data to stay within a reasonable range
y_data_smooth = np.clip(y_data_smooth, 0, 0.2)

# Ensure signals start and end at zero
y_signal1 = np.concatenate(([0], 0.5*np.exp(-((x - 6)**2) / 1.5) * 0.15, [0]))
y_signal2 = np.concatenate(([0], 0.6*np.exp(-((x - 4)**2) / 1.4) * 0.15, [0]))

# Define the signal region
signal_region_start = 4
signal_region_end = 6.3  # Adjusted signal region end

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# First graph with original signal position
axes[0].fill_between(x_extended, y_background_extended, y_background_extended + y_signal1, color='red', alpha=0.3, label='Signal')
axes[0].fill_between(x_extended, 0, y_background_extended, color='purple', alpha=0.5, label='Background')
axes[0].fill_between(x_extended, 0, y_data_smooth, color='blue', alpha=0.2, label='Data Distribution')  # Data distribution
axes[0].scatter(x_extended, y_data_smooth, color='black', label='Data', marker='x')  # Crosses for data points
axes[0].text((signal_region_start + signal_region_end) / 2, 0.18, 'Signal Region', horizontalalignment='center', verticalalignment='center', fontsize=12, color='grey')  # Annotation for signal region
axes[0].axvline(signal_region_start, color='black', linestyle='--')  # Vertical line at signal region start
axes[0].axvline(signal_region_end, color='black', linestyle='--')    # Vertical line at signal region end
axes[0].set_xlim(0, 10)
axes[0].set_ylim(0, 0.2)
axes[0].legend(loc='upper left')  # Add legend explicitly

# Second graph with new signal position
axes[1].fill_between(x_extended, y_background_extended, y_background_extended + y_signal2, color='green', alpha=0.3, label='Signal')
axes[1].fill_between(x_extended, 0, y_background_extended, color='purple', alpha=0.5, label='Background')
axes[1].fill_between(x_extended, 0, y_data_smooth, color='blue', alpha=0.2, label='Data Distribution')  # Data distribution
axes[1].scatter(x_extended, y_data_smooth, color='black', label='Data', marker='x')  # Crosses for data points
axes[1].text((signal_region_start + signal_region_end) / 2, 0.18, 'Signal Region', horizontalalignment='center', verticalalignment='center', fontsize=12, color='grey')  # Annotation for signal region
axes[1].axvline(signal_region_start, color='black', linestyle='--')  # Vertical line at signal region start
axes[1].axvline(signal_region_end, color='black', linestyle='--')    # Vertical line at signal region end
axes[1].set_xlim(0, 10)
axes[1].set_ylim(0, 0.2)
axes[1].legend(loc='upper left')  # Add legend explicitly

# Remove x and y labels and ticks
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.tight_layout()
plt.savefig("Test_reint.png")
plt.show()


