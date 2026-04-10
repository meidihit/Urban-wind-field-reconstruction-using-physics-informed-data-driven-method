import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import seaborn as sns
import os

# Your root directory (containing CSV folder and grid_250k.csv)
# ========== 1) Basic parameter settings ==========  
root_folder = "your path/Urban_flow"  # Root directory
save_folder = 'images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

test_direction = '281.25'  # Test wind direction (pure numbers, e.g., '0', '90', '180')
test_heights = [10, 30, 50, 70, 90, 120]  # Test heights (m)
n_s_per_plane = 2 # Number of sensors per height level 2, 5, 10 (std), 20
sensor_nums = [n_s_per_plane] * len(test_heights)  # Total sensors
r = 10  # Number of POD modes 2, 5, 10 (std), 20
##============= Select a number of wind directions (for POD, it must be greater than the number of modes) ===========##
n_wind_dirc = 48 ## 16, 32, 48

# ========== 2) Read grid file ==========  
grid_file = os.path.join(root_folder, 'grid_250k.csv')
grid_raw = pd.read_csv(grid_file, header=None, skiprows=2, dtype=np.float64).values
grid_data = grid_raw[0:, :]
num_pts = grid_data.shape[0]
# Assume grid file format is consistent for all directions; x/y columns are fixed as 2nd and 3rd columns (index starting from 0)
x_vec = grid_data[:, 0].astype(float)  # Example: column 0 is x coordinate
y_vec = grid_data[:, 2].astype(float)  # Example: column 2 is y coordinate
n = int(np.sqrt(len(x_vec)))
if n * n != len(x_vec):
    raise ValueError(f'Grid file points {len(x_vec)} cannot form a square matrix')

x = x_vec.reshape(n, n)
y = y_vec.reshape(n, n)

# ========== 3) Collect test file paths ==========  
test_dir = os.path.join(root_folder, test_direction)
if not os.path.exists(test_dir):
    raise FileNotFoundError(f'Test direction folder does not exist: {test_dir}')

test_fnames = []
for h in test_heights:
    pattern = f'{h}m.csv'  # File name format: Number_Heightm.csv
    file_path = os.path.join(test_dir, pattern)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Test file does not exist: {file_path}')
    test_fnames.append(file_path)

# ========== 4) Build training dataset ==========  
# Traverse root directory to collect all wind direction folders named with pure numbers
training_dirs = []

for dir_name in os.listdir(root_folder): ## Traverse folders for each wind direction in the database
    dir_path = os.path.join(root_folder, dir_name) #
    ##if dir_name == test_direction: ## Since it's database analysis, do not exclude test direction
    ##    continue 
    if os.path.isdir(dir_path):
        training_dirs.append(dir_path)
        
all_u_combined = []
count_read = 0
effective_wind_directions = []
for train_dir in training_dirs:

    wind_angle = os.path.basename(train_dir).rsplit('.')[0]  # Split once from the right

    print("Reading data %d of %d" %(count_read, len(training_dirs)))
    direction = os.path.basename(train_dir)  # Wind direction name (pure number string)
    direction_files = []
    print("Reading case %s" %train_dir)
    # Check if this wind direction contains data for all test heights
    for h in test_heights:
        pattern = f'{h}m.csv'
        file_path = os.path.join(train_dir, pattern)
        if os.path.exists(file_path):
            direction_files.append(file_path)
        else:
            warnings.warn(f'Wind direction {direction} is missing data file for height {h}m')
            
            break
    
    if len(direction_files) != len(test_heights):
        continue  # Skip incomplete wind direction data
    
    # Read and merge data for all heights of this wind direction
    direction_data = []

    for file_path in direction_files:

        try:
            data = pd.read_csv(file_path, header=None, skiprows=1, dtype=np.float64).values
            if data.shape[0] != n * n:
                warnings.warn(f'File {file_path} line count mismatch, skipping')
                continue
            # Assume data format is [u, v, w], calculate wind speed magnitude
            # Calculate wind speed magnitude
            if data.shape[1] >= 4:
                mag = np.sqrt(np.sum(data[:, 1:4]**2, axis=1))  # Wind speed magnitude
            else:
                mag = data.flatten()   
            direction_data.append(mag)
        except Exception as e:
            warnings.warn(f'Error reading file {file_path}: {str(e)}')
            continue

    count_read = count_read + 1 
    if direction_data:
        effective_wind_directions.append(float(wind_angle))
        all_u_combined.append(np.hstack(direction_data))  # Merge height dimensions
        if len(direction_data) < len(test_heights):
            print("Direction float(wind_angle) %f is problematic" %float(wind_angle))
##===============###=========###=
## Obtain wind directions sorted from smallest to largest
effective_wind_directions = np.array(effective_wind_directions)

# Get index list of elements sorted from smallest to largest
idx_list_wind_direction = np.argsort(effective_wind_directions)

# Rearrange array using index list
effective_wind_directions = effective_wind_directions[ idx_list_wind_direction]

if not all_u_combined:
    raise ValueError('No valid training data found')

# Convert to final data structure: (spatial points × number of heights) × training snapshots
all_u_combined = np.vstack(all_u_combined).T  # Shape: (n²×5, training snapshots)
n_train = all_u_combined.shape[1]
print(f'Number of training wind directions: {len(training_dirs)}')
print(f'Data dimension: {all_u_combined.shape}')
all_u_combined = all_u_combined[:, idx_list_wind_direction] ## All wind directions, shape N_point X N_direction

# Calculate correlation matrix (N_direction x N_direction)
corr_matrix = np.corrcoef(all_u_combined, rowvar=False)

flag_out_of_memory = False
if flag_out_of_memory:
    # Downsample data before calculation
    sample_size = min(1000, all_u_combined.shape[0])
    indices = np.random.choice(all_u_combined.shape[0], sample_size, replace=False)
    sampled_data = all_u_combined[indices, :]
    corr_matrix = np.corrcoef(sampled_data, rowvar=False)

# Get number of wind directions actually used
n_directions = corr_matrix.shape[0]

# ========== 6) Visualize correlation matrix ==========
plt.figure(figsize=(12, 10))

# Create heatmap
ax = sns.heatmap(
    corr_matrix,
    cmap='coolwarm',  # Blue-white-red color scheme
    vmin=-1, vmax=1,   # Correlation coefficient range
    center=0,         # Center value at 0
    square=True,      # Maintain square cells
    linewidths=.5,
    annot=False,       # Do not display values (too many points will be crowded)
    cbar_kws={"shrink": .8}
)

# Replace xticklabels/yticklabels of the heatmap (originally indices, not values)
tick_labels = [f"{ang}°" for ang in effective_wind_directions]
'''
## Show all labels
ax.set_xticks(np.arange(n_directions)+0.5)
ax.set_yticks(np.arange(n_directions)+0.5)
ax.set_xticklabels(tick_labels, fontsize=8, rotation=90)
ax.set_yticklabels(tick_labels, fontsize=8,)
'''

# Define tick interval step (e.g., show one label every 4 directions)
step = 4

# Generate tick positions and labels after interval
xticks = np.arange(0, n_directions, step) + 0.5
yticks = np.arange(0, n_directions, step) + 0.5
xtick_labels = tick_labels[::step]  # Take one label every step
ytick_labels = tick_labels[::step]

# Set tick positions and labels
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels(xtick_labels, fontsize=15, rotation=90)
ax.set_yticklabels(ytick_labels, fontsize=15)

# Set title and labels
plt.title(f'Wind Direction Correlation Matrix ({n_directions} directions)', fontsize=25)
plt.xlabel('Wind Direction Index', fontsize=25)
plt.ylabel('Wind Direction Index', fontsize=25)

# Add colorbar description
cbar = ax.collections[0].colorbar
cbar.set_label('Pearson Correlation Coefficient', fontsize=25)
# **New: Adjust colorbar tick font size**
cbar.ax.tick_params(axis='y', labelsize=22)  # Adjust tick font to 22pt (modify as needed)

# Optimize layout
plt.tight_layout()

# Save image
output_path = os.path.join(save_folder, 'wind_correlation_matrix.png')
os.makedirs(save_folder, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Correlation matrix saved to: {output_path}")

# Optional: show image
# plt.show()
plt.close()

# ========== 7) Optional: Print partial correlation coefficients ==========
print("\nSample correlation values:")
print(f"Diagonal (self-correlation): {corr_matrix[0, 0]:.4f}")
print(f"First vs last direction: {corr_matrix[0, -1]:.4f}")
print(f"Average absolute correlation: {np.mean(np.abs(corr_matrix)):.4f}")