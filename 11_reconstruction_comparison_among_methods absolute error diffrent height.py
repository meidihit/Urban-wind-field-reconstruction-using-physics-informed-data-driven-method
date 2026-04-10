import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.linalg as la
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')
import re

np.random.seed(42)  # Set global seed

# ========== 1) User modification section ==========
##/data/meidi_data/Urban_flow/
# Your root directory (containing CSV folder and grid_250k.csv)
# ========== 1) Basic parameter settings ==========  
root_folder = "your path/Urban_flow/"  # Root directory
result_repository = 'images/'
save_folder = 'images_paper_test_direction_281.25/'


test_direction = '281.25'  # Test wind direction (pure numbers, e.g., '0', '90', '180')

test_heights = [10, 30, 50, 70, 90, 120]  # Test heights (m)
visualization_height = [10, 30, 50, 70, 90, 120]

n_s_per_plane = 16 # Number of sensors per height level 16, 2, 5, 10 (std), 20
sensor_nums = [n_s_per_plane] * len(test_heights)  # Total sensors
r = 10  # Number of POD modes 10, latent 128

latent_dim = r
##============= Select a number of wind directions (for POD, it must be greater than the number of modes) ===========##
n_wind_dirc = 32 ## 8, 16, 32, 64, baseline 32

case_Number = 1 # 1, 2, 3, 4
###==================== Load saved reconstructed wind fields =======================###
##===POD======###
if case_Number == 1:
    u_rec_combined = np.load(result_repository + "U_rec_n_s_per_plane_%d_POD_latent_%d_test_direction_281.25.npy" %(n_s_per_plane, r) )
    err_pct_combined = np.load(result_repository + "U_error_n_s_per_plane_%d_POD_latent_%d_test_direction_281.25.npy" %(n_s_per_plane, r))
    
##========POD-NN ========###
if case_Number == 2:
    u_rec_combined = np.load(result_repository + "U_rec_n_s_per_plane_%d_POD_NN_latent_%d_test_direction_281.25.npy" %(n_s_per_plane, r) )
    err_pct_combined = np.load(result_repository + "U_error_n_s_per_plane_%d_POD_NN_latent_%d_test_direction_281.25.npy" %(n_s_per_plane, r))

##========AE ========###
if case_Number == 3:
    u_rec_combined = np.load(result_repository + "U_rec_n_s_per_plane_%d_AE_latent_%d_test_direction_281.25.npy" %(n_s_per_plane, r) )
    err_pct_combined = np.load(result_repository + "U_error_n_s_per_plane_%d_AE_latent_%d_test_direction_281.25.npy" %(n_s_per_plane, r))
##========AE-NN ========###
if case_Number == 4:
    u_rec_combined = np.load(result_repository + "U_rec_n_s_per_plane_%d_AE_NN_latent_%d_test_direction_281.25.npy" %(n_s_per_plane, r) )
    err_pct_combined = np.load(result_repository + "U_error_n_s_per_plane_%d_AE_NN_latent_%d_test_direction_281.25.npy" %(n_s_per_plane, r))

err_pct_combined = err_pct_combined / 100.0 * 10 ## Restore absolute error
# ========== 2) Read grid file ==========  
grid_file = os.path.join(root_folder, 'grid_250k.csv')
grid_raw = pd.read_csv(grid_file, header=None, skiprows=2, dtype=np.float64).values
grid_data = grid_raw[0:, :]
num_pts = grid_data.shape[0]
# Assume grid file format is consistent for all directions; x/y columns are fixed as 2nd and 3rd columns (index starting from 0)

x_vec = grid_data[:, 2].astype(float)  # Example: column 2 is x coordinate
y_vec = grid_data[:, 3].astype(float)  # Example: column 3 is y coordinate

# Counter-clockwise 90-degree rotation transform
rotated_x = -y_vec  # New X = -Original Y
rotated_y =  x_vec  # New Y = Original X

##x_vec = rotated_x
##y_vec = rotated_y 

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
    if dir_name == test_direction: ## Exclude test wind direction
        continue
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
all_u_combined = np.vstack(all_u_combined).T  # Shape: (n²×number of heights, training snapshots)
n_train = all_u_combined.shape[1]
print(f'Number of training wind directions: {len(training_dirs)}')
print(f'Data dimension: {all_u_combined.shape}')
all_u_combined = all_u_combined[:, idx_list_wind_direction] ## All wind directions

## Select several wind directions
## Note: Since the starting angle is 3.75 and the final angle is 360, which are close, we set endpoint=False for uniform angle spacing.
float_points = np.linspace(0, all_u_combined.shape[1]-1, n_wind_dirc, endpoint=False) ## Uniform spacing, wind directions already sorted from smallest to largest

# Round to nearest integer and convert to integer list
selected_cols = np.round(float_points).astype(int)

all_u_combined = all_u_combined[:, selected_cols]


# Read all height data for the test wind direction and combine
test_snapshot_combined = None
test_data_list = []

for iz, h_m in enumerate(test_heights):
    test_file = os.path.join(root_folder, test_fnames[iz])
    test_data = pd.read_csv(test_file, header=None, skiprows=1, dtype=np.float64).values
    
    if test_data.shape[0] != num_pts:
        raise ValueError(f'Test file {test_fnames[iz]} line count mismatch with grid point count {num_pts}')
    
    # Calculate original wind speed magnitude
    if test_data.shape[1] >= 4:
        original_u = np.sqrt(np.sum(test_data[:, 1:4]**2, axis=1))
    else:
        original_u = test_data.flatten()
    
    test_data_list.append(original_u)
    
    if test_snapshot_combined is None:
        test_snapshot_combined = original_u
    else:
        test_snapshot_combined = np.concatenate((test_snapshot_combined, original_u))

# Select sensor positions across the entire multi-height space
total_sensors = sum(sensor_nums)
combined_num_pts = num_pts * len(test_heights)

# Randomly select sensor positions (across the entire combined space)
idx_sens_combined = np.random.choice(combined_num_pts, size=total_sensors, replace=False)

# Global font enhancement settings (Key modification 1)
plt.rcParams.update({
    ##'font.family': 'sans-serif',
    'font.family': 'Times New Roman',
    
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 25,
    'figure.titlesize': 20
})

# Create figure
fig = plt.figure(figsize=(4 * len(visualization_height), 10))
# ========== 7) Separate results for each height and visualize ==========

count_iz = 0
for iz, h_m in enumerate(test_heights):
    # Extract original data, reconstructed data, and error for this height
    start_idx = iz * num_pts
    end_idx = (iz + 1) * num_pts
    
    original_u = test_data_list[iz]
    u_rec = u_rec_combined[start_idx:end_idx]
    err_pct = err_pct_combined[start_idx:end_idx]
    
    # Extract sensor positions for this height
    height_sensor_indices = []
    for sens_idx in idx_sens_combined:
        if start_idx <= sens_idx < end_idx:
            height_sensor_indices.append(sens_idx - start_idx)
    
    # If there are not enough sensors for this height, supplement with random selection
    num_sens = sensor_nums[iz]
    if len(height_sensor_indices) < num_sens:
        additional_needed = num_sens - len(height_sensor_indices)
        available_indices = list(set(range(num_pts)) - set(height_sensor_indices))
        additional_indices = np.random.choice(available_indices, size=additional_needed, replace=False)
        height_sensor_indices.extend(additional_indices)
    elif len(height_sensor_indices) > num_sens:
        height_sensor_indices = height_sensor_indices[:num_sens]
    
    height_sensor_indices = np.array(height_sensor_indices)
    
    # Reshape results
    orig_2d = original_u.reshape(n, n)
    reco_2d = u_rec.reshape(n, n)
    err_2d = err_pct.reshape(n, n)
    
    if h_m in visualization_height:
        # Plot original data
        ax1 = plt.subplot(3, len(visualization_height), 0 * 3 + count_iz + 1)
        im1 = ax1.pcolormesh(x, y, orig_2d, shading='auto', cmap='rainbow')
        ax1.set_aspect('equal')
        ax1.set_title(f'Original @ {h_m}m')
        plt.colorbar(im1, ax=ax1, shrink=0.85, aspect=15)
        im1.set_clim(0, 15)
        
        # Plot reconstructed data
        ax2 = plt.subplot(3, len(visualization_height), 1 * 3 + count_iz + 1)
        im2 = ax2.pcolormesh(x, y, reco_2d, shading='auto', cmap='rainbow')
        ax2.set_aspect('equal')
        ax2.set_title(f'Reconstructed @ {h_m}m')
        plt.colorbar(im2, ax=ax2, shrink=0.85, aspect=15)
        im2.set_clim(0, 15)
        
        # Plot error
        ax3 = plt.subplot(3, len(visualization_height), 2 * 3 + count_iz + 1)
        im3 = ax3.pcolormesh(x, y, err_2d, shading='auto', cmap='rainbow')
        ax3.set_aspect('equal')
        ax3.set_title('Error (m/s)')
        plt.colorbar(im3, ax=ax3, shrink=0.85, aspect=15)
        im3.set_clim(0, 5)
    
        # Mark buildings (wind speed = 0) with white squares
        bld_mask = (orig_2d == 0)
        for ax in [ax1, ax2, ax3]:

            ax.scatter(x[bld_mask], y[bld_mask], s=1, marker='s', 
                    facecolors='white', edgecolors='none', alpha=0.5)  # [6,8](@ref)
            
        # Mark sensor locations with red dots
        x_sens = x_vec[height_sensor_indices]
        y_sens = y_vec[height_sensor_indices]
        for ax in [ax1, ax2, ax3]:
            ax.scatter(x_sens, y_sens, s=5, marker='o', 
                        facecolors='r', edgecolors='r')
        count_iz = count_iz + 1

##plt.suptitle(f'Multi-Height Gappy POD for {test_direction} (Combined POD)')
plt.tight_layout()


# Save image
out_fig = os.path.join(save_folder, 'U_error_distrinbion_case_%d_test_wind_direction_%.2f.png' %(case_Number, float(test_direction)))
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print(f'Image saved: {out_fig}')
mean_err = np.mean(err_pct_combined)
print(f'Mean Relative Error: {mean_err:.2f}%')
plt.show()

##=================== Plot error distribution plane by plane =====================##

# Global font enhancement settings (Key modification 1)
plt.rcParams.update({
    ##'font.family': 'sans-serif',
    'font.family': 'Times New Roman',
    
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 25,
    'figure.titlesize': 20
})


# Pre-calculate statistics for all heights
all_heights = []
mean_errors = []
std_errors = []
median_errors = []
q95_errors = []

for iz, h_m in enumerate(test_heights):
    start_idx = iz * num_pts
    end_idx = (iz + 1) * num_pts
    err_pct_plane = err_pct_combined[start_idx:end_idx]
    
    # Calculate statistics
    mean_err = np.mean(err_pct_plane)
    std_err = np.std(err_pct_plane)
    median_err = np.median(err_pct_plane)
    q95_err = np.percentile(err_pct_plane, 95)
    
    all_heights.append(h_m)
    mean_errors.append(mean_err)
    std_errors.append(std_err)
    median_errors.append(median_err)
    q95_errors.append(q95_err)

## Create plotting area (Key modification)
fig, ax = plt.subplots(figsize=(7, 6))

# Sort data by height (ensure continuous curves)
sorted_indices = np.argsort(all_heights)
sorted_heights = np.array(all_heights)[sorted_indices]
sorted_mean = np.array(mean_errors)[sorted_indices]
sorted_std = np.array(std_errors)[sorted_indices]

# Plot mean curve and standard deviation area (Key modification)
ax.plot(sorted_heights, sorted_mean, 
        color="#b4211f", linewidth=2, label='Mean Error')
ax.fill_between(sorted_heights, 
               sorted_mean - sorted_std,
               sorted_mean + sorted_std,
               color='#1f77b4', alpha=0.2, label='Std Dev')

# Add auxiliary statistical lines (median and 95th percentile)
ax.plot(sorted_heights, np.array(median_errors)[sorted_indices], 
        color="#67bd6b", linestyle='--', linewidth=1.5, label='Median Error')
ax.plot(sorted_heights, np.array(q95_errors)[sorted_indices], 
        color="#564b8c", linestyle='-.', linewidth=1.5, label='95th Percentile')

# Axis optimization (Key modification)
ax.set_xlabel('Height (m)', fontweight='semibold', fontsize=22)
ax.set_ylabel('Absolute Error (m/s)', fontweight='semibold', fontsize=22)
##ax.set_title('Error Statistics vs. Height', fontsize=20, pad=20)
ax.tick_params(axis='both', which='major', labelsize=18, labelcolor='black', width=1.5)

# Set grid and legend
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', 
         fontsize=20, frameon=True, edgecolor='#444444')

# Enhance visualization effect
ax.set_xticks(sorted_heights)
ax.set_xticklabels([f'{h:.1f}' for h in sorted_heights], rotation=45)
##ax.set_yticks(np.arange(0, 101, 10))
ax.set_ylim(0, 3.5)

# Save optimized figure
out_fig_errors = os.path.join(save_folder, 'U_error_plot_case_%d_test_wind_direction_%.2f.png' %(case_Number, float(test_direction)) )
plt.savefig(out_fig_errors, dpi=300, bbox_inches='tight',  transparent=False)

print(f'Optimized error distribution plot saved: {out_fig_errors}')