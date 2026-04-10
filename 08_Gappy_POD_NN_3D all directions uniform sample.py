import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import griddata
from scipy.linalg import svd, pinv
from matplotlib.colors import Normalize
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
import seaborn as sns
from scipy import stats

np.random.seed(42)  # Set global seed
# ===== Path Settings =====
root_folder = "your path/Urban_flow"  # Root directory
save_folder = "images/"
log_folder = 'log'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

use_pytorch = True  # Use PyTorch model; if False, use scikit-learn model
num_epochs = 5000   # Number of training epochs
n_s_per_plane = 20 # Number of sensors per height level 2, 5, 10 (std), 20
learning_rate = 5e-5 ## Learning rate should not be too high
r = 10  # Number of POD modes 2, 5, 10 (std), 20, 25
##============= Select a number of wind directions (for POD, it must be greater than the number of modes) ===========##
n_wind_dirc = 32 ##8,  16, 32,  64
    
##test_direction = '11.25'  # Test wind direction (pure numbers, e.g., '0', '90.0', '180.0', '270.0') 281.25
test_direction = '281.25'
test_heights = [10, 30, 50, 70, 90, 120]  # Test heights (m)
sensor_nums = [n_s_per_plane] * len(test_heights)  # Total number of sensors

# ========== 2) Read grid file ==========  
grid_file = os.path.join(root_folder, 'grid_250k.csv')
grid_raw = pd.read_csv(grid_file, header=None, skiprows=2, dtype=np.float64).values
grid_data = grid_raw[0:, :]
num_pts = grid_data.shape[0]
# Assume grid file format is consistent for all directions; x/y columns are fixed as 2nd and 3rd columns (index starting from 0)
x_vec = grid_data[:, 2].astype(float)  # Example: column 2 is x coordinate
y_vec = grid_data[:, 3].astype(float)  # Example: column 3 is y coordinate
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

    wind_angle = os.path.basename(train_dir)

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

##===============###=========###=
## Obtain wind directions sorted from smallest to largest
effective_wind_directions = np.array(effective_wind_directions)

# Get index list of elements sorted from smallest to largest
idx_list_wind_direction = np.argsort(effective_wind_directions)

# Rearrange array using index list
effective_wind_directions = effective_wind_directions[ idx_list_wind_direction]

if not all_u_combined:
    raise ValueError('No valid training data found')

# Convert to final data structure: (spatial points × number of heights) × number of training snapshots
all_u_combined = np.vstack(all_u_combined).T  # Shape: (n²×number of heights, number of training snapshots)
n_train = all_u_combined.shape[1]
print(f'Number of training wind directions: {len(training_dirs)}')
print(f'Data dimension: {all_u_combined.shape}')
all_u_combined = all_u_combined[:, idx_list_wind_direction]


## Select several wind directions
## Note: Since the starting angle is 3.75 and the final angle is 360, which are close, we set endpoint=False for uniform angle spacing.
float_points = np.linspace(0, all_u_combined.shape[1]-1, n_wind_dirc, endpoint=False) ## Uniform spacing, wind directions already sorted from smallest to largest

# Round to nearest integer and convert to integer list
selected_cols = np.round(float_points).astype(int)

all_u_combined = all_u_combined[:, selected_cols]
    
# ===== Multi-height Joint POD Decomposition =====
print('Multi-height POD processing')
U_combined, s_combined, VT_combined = svd(all_u_combined, full_matrices=False)
energy_ratio = np.cumsum(s_combined**2) / np.sum(s_combined**2)
r_domi = np.argmax(energy_ratio >= 0.99) + 1 ## Modes accounting for over 99% energy
print(f"Dominant mode number is {r_domi}")
U_r_combined = U_combined[:, :np.min((r, all_u_combined.shape[1]))] ## Take a certain number of modes

# ===== Read test data (multi-height) =====
print('Reading multi-height test data')
test_data_combined = None
original_u_list = []

for iz, h_m in enumerate(test_heights):
    test_file = os.path.join(root_folder, test_fnames[iz])
    test_data = pd.read_csv(test_file, header=None, skiprows=1, dtype=np.float64).values
    
    if test_data.shape[0] != num_pts:
        raise ValueError(f'Test file {test_fnames[iz]} line count mismatch with grid point count {num_pts}')
    
    # Calculate original wind speed magnitude
    original_u = np.linalg.norm(test_data[:, 1:4], axis=1) if test_data.shape[1] >= 4 else test_data.flatten()
    original_u_list.append(original_u)
    
    # Combine multi-height test data
    if test_data_combined is None:
        test_data_combined = original_u
    else:
        test_data_combined = np.concatenate((test_data_combined, original_u))

# ===== Prepare training data (multi-plane joint mode) =====
print('Preparing training data for multi-height ML model')
'''Random selection'''
###=======###

total_sensors = n_s_per_plane * len(test_heights)

# Select sensor positions across the entire multi-height space
combined_num_pts = num_pts * len(test_heights)
idx_sens_combined = np.random.choice(combined_num_pts, size=total_sensors, replace=False)

'''Spatial uniform spacing'''
'''
###=======###
sensor_indices_temp = []
rows = int(np.ceil(np.sqrt(num_pts)))  # Number of rows per plane
cols = int(np.ceil(np.sqrt(num_pts)))  # Number of columns per plane


for i, n_s in enumerate(sensor_nums):
    if n_s == 0:
        continue
        
    # Dynamically calculate 2D grid dimensions
    grid_size = int(np.ceil(np.sqrt(n_s)))
    actual_n_s = min(grid_size**2, n_s)
    

    # Generate grid coordinates (row-major)
    x0 = np.linspace(0, cols-1, grid_size, dtype=int)
    y0 = np.linspace(0, rows-1, grid_size, dtype=int)
    xx0, yy0 = np.meshgrid(x0, y0, indexing='xy')  # Key: ensure x is column direction, y is row direction
    
    # Flatten and intercept actually needed points
    points = np.column_stack((xx0.ravel(), yy0.ravel()))[:actual_n_s] ## Flatten 2D grid coordinates, combine into point coordinate matrix, and intercept specified number of sampling points (first N points)
    local_indices = points[:,1] * cols + points[:,0]  # Row-major conversion
    
    # Calculate global offset
    height_offset = i * num_pts
    combined_indices = local_indices + height_offset
    
    sensor_indices_temp.extend(combined_indices)

idx_sens_combined = np.array(sensor_indices_temp)
'''

X_train = []
y_train = []

for i in range(all_u_combined.shape[1]):  # Iterate through training samples
    # Get sensor measurements
    sensor_measurements = all_u_combined[idx_sens_combined, i]
    
    # Calculate true POD coefficients
    true_coefficients = pinv(U_r_combined) @ all_u_combined[:, i]
    
    X_train.append(sensor_measurements) ### Input, sensor measurements
    y_train.append(true_coefficients) ### Output, latent variables

X_train = np.array(X_train)
y_train = np.array(y_train)

## Construct validation set
test_data = test_data_combined[:, np.newaxis]

X_val = []
y_val = []
for i in range(test_data.shape[1]):  # Iterate through training samples
    # Get sensor measurements
    sensor_measurements = test_data[idx_sens_combined, i]
    
    # Calculate true POD coefficients
    true_coefficients = pinv(U_r_combined) @ test_data[:, i]
    
    X_val.append(sensor_measurements) ### Input, sensor measurements
    y_val.append(true_coefficients) ### Output, latent variables

X_train = np.array(X_train) 
y_train = np.array(y_train)

# Split training and validation sets (skipping split to avoid missing wind directions)
###X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Data standardization
input_scaler = StandardScaler()
X_train_scaled = input_scaler.fit_transform(X_train)
X_val_scaled = input_scaler.transform(X_val)

output_scaler = StandardScaler()
y_train_scaled = output_scaler.fit_transform(y_train)
y_val_scaled = output_scaler.transform(y_val)

# ===== Create ML model =====
if use_pytorch:
    print('Creating PyTorch model for multi-height reconstruction')
    
    class PODCoeffModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(PODCoeffModel, self).__init__()
            uniform_width = 512
            self.fc1 = nn.Linear(input_dim, uniform_width)
            self.fc2 = nn.Linear(uniform_width, uniform_width)
            self.fc3 = nn.Linear(uniform_width, uniform_width)
            self.fc4 = nn.Linear(uniform_width, output_dim)
            self.dropout = nn.Dropout(0.5) ## Increase to combat overfitting
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    
    input_dim = X_train_scaled.shape[1]
    output_dim = y_train_scaled.shape[1]
    model = PODCoeffModel(input_dim, output_dim)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay= 0.002) ## weight_decay avoids overfitting, very important
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0.00001, last_epoch=-1
        )
    
    
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled)
    
    print('Training PyTorch model')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor) ## Loss: difference between dataset latent variables and predicted latent variables
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')
    
    # Plot training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Multi-height Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_folder, 'Projection_training_loss_POD_NN_n_s_per_plane_%d_latent_%d.png' %(n_s_per_plane, r) ), dpi=300)
    plt.close()
else:
    print('Creating scikit-learn model for multi-height reconstruction')
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train_scaled)
    
    y_pred = model.predict(X_val_scaled)
    mse = mean_squared_error(y_val_scaled, y_pred)
    print(f'Validation MSE: {mse:.6f}')

# ===== Multi-height Joint ML Reconstruction =====
print('Multi-height full field reconstruction using ML model')

# Get sensor measurements
u_test_combined = test_data_combined[idx_sens_combined]

# Prepare input features: sensor measurements
input_features = u_test_combined.reshape(1, -1)
input_features_scaled = input_scaler.transform(input_features)

# Use ML model to predict POD coefficients
if use_pytorch:
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_features_scaled)
        coeff_scaled = model(input_tensor).numpy()
else:
    coeff_scaled = model.predict(input_features_scaled)

coeff = output_scaler.inverse_transform(coeff_scaled).flatten()

# Reconstruct full field (multi-height)
u_rec_combined = U_r_combined @ coeff

# ===== Separate results for each height and analyze =====
sensor_table = pd.DataFrame(columns=['X_m', 'Y_m', 'Z_m'])

# Create plots for all heights
# Create figure
fig = plt.figure(figsize=(10, 2.5 * len(test_heights)))
err_list = []
for iz, h_m in enumerate(test_heights):
    # Extract data for this height
    start_idx = iz * num_pts
    end_idx = (iz + 1) * num_pts
    
    original_u = original_u_list[iz]
    u_rec = u_rec_combined[start_idx:end_idx]
    
    # Calculate error
    err_pct = 100 * np.abs(original_u - u_rec) / 10.0
    err_list.extend( err_pct.tolist() )
    # Extract sensor positions for this height
    height_sensor_indices = []
    for sens_idx in idx_sens_combined:
        if start_idx <= sens_idx < end_idx:
            height_sensor_indices.append(sens_idx - start_idx)
    
    # Ensure each height has enough sensors
    if len(height_sensor_indices) < n_s_per_plane:
        additional_needed = n_s_per_plane - len(height_sensor_indices)
        available_indices = list(set(range(num_pts)) - set(height_sensor_indices))
        additional_indices = np.random.choice(available_indices, size=additional_needed, replace=False)
        height_sensor_indices.extend(additional_indices)
    
    height_sensor_indices = np.array(height_sensor_indices)
    
    # Reshape results
    orig_2d = original_u.reshape(n, n)
    reco_2d = u_rec.reshape(n, n)
    err_2d = err_pct.reshape(n, n)
    
    # Visualization
    '''
    titles = [f'Original @ {h_m}m', f'Reconstruction @ {h_m}m', f'Error @ {h_m}m (%)']
    data_arr = [orig_2d, reco_2d, err_2d]
    
    for i in range(3):
        plt.subplot(len(test_heights), 3, iz*3 + i+1)
        cmap = 'viridis'
        
        im = plt.pcolormesh(x, y, data_arr[i], shading='auto')
        plt.axis('equal')
        plt.title(titles[i])
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.colorbar(im)
        
        if i < 2:
            im.set_clim(0, 15)
        else:
            im.set_clim(0, 60)
        
        # Mark buildings
        bldIdx = orig_2d == 0
        plt.scatter(x[bldIdx], y[bldIdx], s=30, marker='s', color='white')
        
        # Mark sensor locations
        plt.scatter(x_vec[height_sensor_indices], y_vec[height_sensor_indices], 
                   s=15, marker='o', color='red')
    '''
    
    
    # Plot original data
    ax1 = plt.subplot(len(test_heights), 3, iz*3 + 1)
    im1 = ax1.pcolormesh(x, y, orig_2d, shading='auto')
    ax1.set_aspect('equal')
    ax1.set_title(f'Original @ {h_m}m')
    plt.colorbar(im1, ax=ax1)
    im1.set_clim(0, 15)
    
    # Plot reconstructed data
    ax2 = plt.subplot(len(test_heights), 3, iz*3 + 2)
    im2 = ax2.pcolormesh(x, y, reco_2d, shading='auto')
    ax2.set_aspect('equal')
    ax2.set_title(f'Reconstructed @ {h_m}m')
    plt.colorbar(im2, ax=ax2)
    im2.set_clim(0, 15)
    
    # Plot error
    ax3 = plt.subplot(len(test_heights), 3, iz*3 + 3)
    im3 = ax3.pcolormesh(x, y, err_2d, shading='auto')
    ax3.set_aspect('equal')
    ax3.set_title('Error (%)')
    plt.colorbar(im3, ax=ax3)
    im3.set_clim(0, 100)
    
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
        
    # Collect sensor coordinates
    sensor_subtab = pd.DataFrame({
        'X_m': x_vec[height_sensor_indices],
        'Y_m': y_vec[height_sensor_indices],
        'Z_m': [h_m] * len(height_sensor_indices)
    })
    sensor_table = pd.concat([sensor_table, sensor_subtab], ignore_index=True)


np.save(save_folder + "U_error_n_s_per_plane_%d_POD_NN_latent_%d_test_direction_%.2f_uniformGrid.npy" %(n_s_per_plane, r, float(test_direction)), np.array(err_list) )
## Save reconstructed wind field
np.save(save_folder + "U_rec_n_s_per_plane_%d_POD_NN_latent_%d_test_direction_%.2f_uniformGrid.npy" %(n_s_per_plane, r, float(test_direction)), u_rec_combined)
    
# Save image
plt.tight_layout()
out_fig = os.path.join(save_folder, 'U_error_n_s_per_plane_%d_POD_NN_latent_%d_test_direction_%.2f_uniformGrid.png' %(n_s_per_plane, r, float(test_direction)))
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print(f'Image saved: {out_fig}')


### Load new trajectory data
trajectory1 = np.load('trajectory1.npy') ## Spatial 2D coordinates of the trajectory
trajectory2 = np.load('trajectory2.npy')

for iz, h_m in enumerate(test_heights):
    # Extract original data, reconstructed data, and error for this height
    start_idx = iz * num_pts
    end_idx = (iz + 1) * num_pts
    
    original_u = original_u_list[iz]
    u_rec = u_rec_combined[start_idx:end_idx]
    
    "=== Trajectory Visualization ==="
    # Create trajectory coordinate mapping
    original_traj1 = []
    recon_traj1 = []
    original_traj2 = []
    recon_traj2 = []

    if iz == len(test_heights)-1: ## Specific plane
        # Iterate through trajectory points for interpolation extraction
        for pt in trajectory1:
            distance = (x_vec - pt[0])**2 +  (y_vec - pt[1])**2
            pos_idx = np.argmin(distance)
            # Extract data and handle boundaries
            original_traj1.append(original_u[pos_idx])
            recon_traj1.append(u_rec[pos_idx])

        ###===== Trajectory 2
        for pt in trajectory2:
            distance = (x_vec - pt[0])**2 +  (y_vec - pt[1])**2
            pos_idx = np.argmin(distance)

            original_traj2.append(original_u[pos_idx])
            recon_traj2.append(u_rec[pos_idx])


###===================### Trajectory Comparison
# Draw new comparison plot
# Global font settings (optional)
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

# Draw new comparison plot
plt.figure(figsize=(10, 10), dpi=300)  # Set image resolution

# Original trajectory comparison
plt.subplot(2, 1, 1)
# Assume data range is known, can be adjusted based on actual data
x_range = range(len(original_traj1))
y_min, y_max = min(min(original_traj1), min(recon_traj1)), max(max(original_traj1), max(recon_traj1))

plt.plot(x_range, original_traj1, 'bo-', label='Trajectory 1 Original')
plt.plot(x_range, recon_traj1, 'ro-', label='Trajectory 1 Reconstructed')

plt.xlabel('Index', fontsize=20)  # Set x-axis label font
plt.ylabel('Wind Speed (m/s)', fontsize=20)  # Set y-axis label font
plt.legend(loc='upper right')  # Adjust legend position
plt.grid(True, linestyle='--', alpha=0.7)  # Adjust grid line style
plt.xlim(0, len(original_traj1))  # Set x-axis range
plt.ylim(y_min-1, y_max+1)  # Set y-axis range (leave 1 unit margin)

# Reconstructed trajectory comparison
plt.subplot(2, 1, 2)
x_range = range(len(original_traj2))
y_min, y_max = min(min(original_traj2), min(recon_traj2)), max(max(original_traj2), max(recon_traj2))

plt.plot(x_range, original_traj2, 'bo-', label='Trajectory 2 Original')
plt.plot(x_range, recon_traj2, 'ro-', label='Trajectory 2 Reconstructed')

plt.xlabel('Index', fontsize=20)
plt.ylabel('Wind Speed (m/s)', fontsize=20)
plt.legend(loc='upper right', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, len(original_traj2))
plt.ylim(y_min-1, y_max+1)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Adjust subplot spacing
plt.tight_layout()

# Save comparison plot  
plt.savefig(os.path.join(save_folder, 'trajectory_comparison_POD_NN_nSensor_%d_nLatent_%d_nDirc_%d_1.png' %(n_s_per_plane, r, n_wind_dirc)), dpi=300, bbox_inches='tight')
plt.close()

np.save( os.path.join(save_folder, 'original_traj1_POD_NN_1.npy'), original_traj1)
np.save( os.path.join(save_folder, 'recon_traj1_POD_NN_1.npy'), recon_traj1)
np.save( os.path.join(save_folder, 'original_traj2_POD_NN_1.npy'), original_traj2)
np.save( os.path.join(save_folder, 'recon_traj2_POD_NN_1.npy'), recon_traj2)

# ===== Export Results =====
# Sensor coordinates
##sensor_table.to_excel(os.path.join(root_folder, 'multi_height_sensor_coordinates.xlsx'), index=False)

# Error analysis
combined_err_pct = 100 * np.abs(test_data_combined - u_rec_combined) / 10.0
mean_err = np.mean(combined_err_pct)
mean_table = pd.DataFrame({'MeanRelativeError (%)': [mean_err]})
##mean_table.to_excel(os.path.join(root_folder, 'multi_height_mean_relative_error.xlsx'), index=False)

print('Multi-height ML-based reconstruction completed!')
print(f'Mean Relative Error: {mean_err:.2f}%')