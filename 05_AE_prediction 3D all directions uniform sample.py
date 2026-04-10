import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import seaborn as sns
from scipy import stats
import warnings

np.random.seed(42)  # Set global seed
# ===== User-modifiable parameters =====
# When running out of memory, you can try reducing the values of latent_dim (line 25), batch_size (line 26), and uniform_width (line 252).
# However, this will reduce model expressiveness (latent_dim), training stability (batch_size), and feature extraction capability (uniform_width), leading to decreased reconstruction accuracy.
latent_dim = 128 #25, 64 #128, 256, 320                  # Latent variable dimension
batch_size = 32
epochs = 2000
learning_rate = 1e-4 ## Learning rate should not be too high
num_optim_steps = 4000 ## Number of latent variable optimization iterations

##============= Select number of wind directions (for POD, should be greater than number of modes) ===========##
n_wind_dirc = 32 ## 8, 16, 32, 64

n_s_per_plane = 20  # Number of sensors per height: 2, 5, 10 (std), 20
    
# ===== Path settings =====
root_folder = "your path/Urban_flow/"  # Root directory
save_folder = "images/"
log_folder = 'log'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
##test_direction = '11.25'  # Test wind direction (pure number, e.g., '0', '90.0', '180.0', '270.0') 281.25
test_direction = '281.25'
test_heights = [10, 30, 50, 70, 90, 120]  # Test heights (meters)

sensor_nums = [n_s_per_plane] * len(test_heights)  # Total number of sensors

# ========== 2) Read grid file ==========  
grid_file = os.path.join(root_folder, 'grid_250k.csv')
grid_raw = pd.read_csv(grid_file, header=None, skiprows=2, dtype=np.float64).values
grid_data = grid_raw[0:, :]
num_pts = grid_data.shape[0]
# Assuming grid file format is consistent for all directions, with x/y columns fixed at index 2 and 3
x_vec = grid_data[:, 2].astype(float)  # Example: column 2 is x coordinate
y_vec = grid_data[:, 3].astype(float)  # Example: column 3 is y coordinate
n = int(np.sqrt(len(x_vec)))
if n * n != len(x_vec):
    raise ValueError(f'Grid file point count {len(x_vec)} cannot form a square matrix')

x = x_vec.reshape(n, n)
y = y_vec.reshape(n, n)

# ========== 3) Collect test file paths ==========  
test_dir = os.path.join(root_folder, test_direction)
if not os.path.exists(test_dir):
    raise FileNotFoundError(f'Test direction folder does not exist: {test_dir}')

test_fnames = []
for h in test_heights:
    pattern = f'{h}m.csv'  # Filename format: heightm.csv
    file_path = os.path.join(test_dir, pattern)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Test file does not exist: {file_path}')
    test_fnames.append(file_path)

# ========== 4) Build training dataset ==========  
# Traverse root directory and collect all wind direction folders named with pure numbers
training_dirs = []
for dir_name in os.listdir(root_folder): ## Traverse database folders for each wind direction
    dir_path = os.path.join(root_folder, dir_name) #
    if dir_name == test_direction: ## Exclude test wind direction
        continue
    if os.path.isdir(dir_path):
        training_dirs.append(dir_path)

all_u_combined = []
count_read = 0
effective_wind_directions = []
for train_dir in training_dirs:

    wind_angle = os.path.basename(train_dir)  # Split once from the right

    print("Reading data %d of %d" %(count_read, len(training_dirs)))
    direction = os.path.basename(train_dir)  # Wind direction name (pure number string)
    direction_files = []
    print("Reading case %s" %train_dir)
    # Check if this wind direction includes data files for all test heights
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
    
    # Read and combine data for all heights of this wind direction
    direction_data = []

    for file_path in direction_files:
        
        try:
            data = pd.read_csv(file_path, header=None, skiprows=1, dtype=np.float64).values
            if data.shape[0] != n * n:
                warnings.warn(f'File {file_path} row count mismatch, skipping')
                continue
            # Assuming data format is [u, v, w], calculate wind speed magnitude
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
        all_u_combined.append(np.hstack(direction_data))  # Combine across height dimension

##===============###=========###=
## Arrange wind directions from smallest to largest
effective_wind_directions = np.array(effective_wind_directions)

# Get index list for elements sorted from smallest to largest
idx_list_wind_direction = np.argsort(effective_wind_directions)

# Rearrange array using index list
effective_wind_directions = effective_wind_directions[ idx_list_wind_direction]

if not all_u_combined:
    raise ValueError('No valid training data found')

# Convert to final data structure: (spatial points × number of heights) × training snapshots
all_u_combined = np.vstack(all_u_combined).T  # Shape: (n²×6, training snapshots)
n_train = all_u_combined.shape[1]
print(f'Number of training wind directions: {len(training_dirs)}')
print(f'Data dimensions: {all_u_combined.shape}')
all_u_combined = all_u_combined[:, idx_list_wind_direction]

## Select several wind directions
## Note: Since starting angle is 3.75 and final angle is 360 (close to each other), we set endpoint=False for uniform spacing
float_points = np.linspace(0, all_u_combined.shape[1]-1, n_wind_dirc, endpoint=False) ## Equidistant, wind directions sorted numerically

# Round to nearest integer and convert to integer list
selected_cols = np.round(float_points).astype(int)

all_u_combined = all_u_combined[:, selected_cols]
    
# ===== Data standardization =====
scaler = StandardScaler()
all_u_scaled = scaler.fit_transform(all_u_combined.T).T  # Transpose for correct standardization

# ===== Read test data (multiple heights) =====
print('Reading multi-height test data')
test_data_combined = None
original_u_list = []

for iz, h_m in enumerate(test_heights):
    test_file = os.path.join(root_folder, test_fnames[iz])
    test_data = pd.read_csv(test_file, header=None, skiprows=1, dtype=np.float64).values
    
    if test_data.shape[0] != num_pts:
        raise ValueError(f'Test file {test_fnames[iz]} row count mismatch with grid points {num_pts}')
    
    # Calculate original wind speed magnitude
    original_u = np.linalg.norm(test_data[:, 1:4], axis=1) if test_data.shape[1] >= 4 else test_data.flatten()
    original_u_list.append(original_u)
    
    # Combine multi-height test data
    if test_data_combined is None:
        test_data_combined = original_u
    else:
        test_data_combined = np.concatenate((test_data_combined, original_u))

# Standardize test data
test_data_scaled = scaler.transform(test_data_combined.reshape(1, -1)).flatten()

# ===== Key modification 1: Multiple height sensor selection =====
total_sensors = n_s_per_plane * len(test_heights)
combined_num_pts = num_pts * len(test_heights)

# Select sensor positions in the entire multi-height space
'''Random selection'''

random.seed(42)
sensor_idx_combined = random.sample(range(combined_num_pts), total_sensors)
sensor_idx_combined = sorted(sensor_idx_combined)

###=======###
'''Uniform spatial interval'''
'''
###=======###
sensor_indices_temp = []
rows = int(np.ceil(np.sqrt(num_pts)))  # Number of rows in plane
cols = int(np.ceil(np.sqrt(num_pts)))  # Number of columns in plane


for i, n_s in enumerate(sensor_nums):
    if n_s == 0:
        continue
        
    # Dynamically calculate 2D grid size
    grid_size = int(np.ceil(np.sqrt(n_s)))
    actual_n_s = min(grid_size**2, n_s)
    

    # Generate grid coordinates (row-major)
    x0 = np.linspace(0, cols-1, grid_size, dtype=int)
    y0 = np.linspace(0, rows-1, grid_size, dtype=int)
    xx0, yy0 = np.meshgrid(x0, y0, indexing='xy')  # Key: ensure x is column direction, y is row direction
    
    # Flatten and take actual required points
    points = np.column_stack((xx0.ravel(), yy0.ravel()))[:actual_n_s] ## Flatten 2D grid coordinates and combine into point matrix, then take specified number of points
    local_indices = points[:,1] * cols + points[:,0]  # Row-major conversion
    
    # Calculate global offset
    height_offset = i * num_pts
    combined_indices = local_indices + height_offset
    
    sensor_indices_temp.extend(combined_indices)

sensor_idx_combined = np.array(sensor_indices_temp)
'''

# ===== AE model definition (modified to handle multi-height data) =====
class MultiHeightAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MultiHeightAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        uniform_width = 256
        # Encoder 
        '''Wind field data is very large; if network width and depth are too great, it will lead to insufficient memory'''
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, uniform_width),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add Dropout to encoder
            nn.Linear(uniform_width, uniform_width),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add Dropout to encoder
            nn.Linear(uniform_width, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, uniform_width),
            nn.ReLU(),
            ##nn.Dropout(0.3),  # Add Dropout to encoder
            nn.Linear(uniform_width, uniform_width),
            nn.ReLU(),
            ##nn.Dropout(0.3),  # Add Dropout to encoder
            nn.Linear(uniform_width, input_dim)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

# ===== Train AE model =====
print('Training multi-height AE model')

# Prepare data
###X = torch.tensor(all_u_scaled.T, dtype=torch.float32)  # Shape: (samples, features)

##X_train, X_val = train_test_split(X, test_size=0.1, random_state=42) ## Do not perform split

X_train = torch.tensor(all_u_scaled.T, dtype=torch.float32)  # Shape: (samples, features)
X_val  = torch.tensor(test_data_scaled[np.newaxis, :], dtype=torch.float32)  # Shape: (samples, features)

train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size)

# Initialize model and optimizer
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MultiHeightAE(input_dim=X_train.shape[1], latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay= 0.0005) ## weight_decay avoids overfitting, very important
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=0.00001, last_epoch=-1
)

# Loss function
def loss_function(recon_x, x):
    return nn.functional.mse_loss(recon_x, x)
    ##return nn.functional.mse_loss(recon_x, x, reduction='sum')

# Training loop
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
        data = batch[0].to(device)
        ##print("data.shape ", data.shape)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    lr_scheduler.step()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            data = batch[0].to(device)
            recon_batch = model(data)
            val_loss += loss_function(recon_batch, data).item()
    
    avg_train_loss = train_loss / len(train_loader) ## Divide by number of batches
    avg_val_loss = val_loss / len(val_loader)
    
    if epoch > 10: ## Initial values are large, hard to see trend
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
    
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Plot training curve
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Multi-height AE Training Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(log_folder, 'Compression_training_loss_AE_n_s_per_plane_%d_latent_%d.png' %(n_s_per_plane, latent_dim) ), dpi=300)

# ===== Optimize latent variable using gradient descent (multiple heights) =====
print('Optimizing latent variable for multi-height reconstruction')

# Get sensor measurements (standardized)
sensor_measurements_scaled = test_data_scaled[sensor_idx_combined]

# Initialize latent variable
model.eval()
with torch.no_grad():
    test_tensor = torch.tensor(test_data_scaled.reshape(1, -1), dtype=torch.float32).to(device)
    initial_z = model.encode(test_tensor).clone().detach().requires_grad_(True) ## Use the overall wind field data for initialization
    
# Optimizer only targets the latent variable
optimizer = optim.Adam([initial_z], lr=0.01) ## Should be slightly larger

loss_history = []
loss_history1 = []
for step in range(num_optim_steps):
    optimizer.zero_grad()
    
    # Decode entire wind field
    reconstructed_field = model.decode(initial_z)
    
    # Reconstruction error at sensor positions only
    reconstructed_measurements = reconstructed_field[0, sensor_idx_combined]
    true_measurements = torch.tensor(sensor_measurements_scaled, dtype=torch.float32).to(device)
    
    reconstructed_all = reconstructed_field[0, :]
    true_all = torch.tensor(test_data_scaled, dtype=torch.float32).to(device)
    
    # Loss function: MSE at sensor positions
    loss = nn.functional.mse_loss(reconstructed_measurements, true_measurements)
    loss_history.append(loss.item())
    
    loss_overall = nn.functional.mse_loss(reconstructed_all, true_all)
    loss_history1.append(loss_overall.item())
    
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        print(f'Optimization step {step}/{num_optim_steps}, Loss: {loss.item():.6f}')

# Final reconstruction
with torch.no_grad():
    optimized_z = initial_z.detach()
    u_rec_scaled = model.decode(optimized_z).cpu().numpy()
    u_rec_combined = scaler.inverse_transform(u_rec_scaled).flatten()

# Plot loss curve during optimization
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label = 'Measurements')
plt.plot(loss_history1, label = 'Overall')
plt.legend()
plt.xlabel('Optimization Steps')
plt.ylabel('Reconstruction Loss')
plt.title('Multi-height Latent Variable Optimization')
plt.grid(True)

plt.savefig(os.path.join(log_folder, 'Latent_optimization_AE_n_s_per_plane_%d_latent_%d.png' %(n_s_per_plane, latent_dim) ), dpi=300)
    
# ===== Separate results for each height and analyze =====
sensor_table = pd.DataFrame(columns=['X_m', 'Y_m', 'Z_m'])

# Create figure to show all heights
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
    for sens_idx in sensor_idx_combined:
        if start_idx <= sens_idx < end_idx:
            height_sensor_indices.append(sens_idx - start_idx)
    
    # Ensure each height has enough sensors
    if len(height_sensor_indices) < n_s_per_plane:
        additional_needed = n_s_per_plane - len(height_sensor_indices)
        available_indices = list(set(range(num_pts)) - set(height_sensor_indices))
        additional_indices = np.random.choice(available_indices, size=additional_needed, replace=False)
        height_sensor_indices.extend(additional_indices)
    
    height_sensor_indices = np.array(height_sensor_indices)
    
    # Reshape the results
    orig_2d = original_u.reshape(n, n)
    reco_2d = u_rec.reshape(n, n)
    err_2d = err_pct.reshape(n, n)
    
    # Visualization
    titles = [f'Original @ {h_m}m', f'Reconstruction @ {h_m}m', f'Error @ {h_m}m (%)']
    data_arr = [orig_2d, reco_2d, err_2d]   
    
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
    
    # Mark buildings (wind speed = 0) as white squares
    bld_mask = (orig_2d == 0)
    for ax in [ax1, ax2, ax3]:

        ax.scatter(x[bld_mask], y[bld_mask], s=1, marker='s', 
                facecolors='white', edgecolors='none', alpha=0.5)  # [6,8](@ref)
        
    # Mark sensor positions as red dots
    x_sens = x_vec[height_sensor_indices]
    y_sens = y_vec[height_sensor_indices]
    for ax in [ax1, ax2, ax3]:
        ax.scatter(x_sens, y_sens, s=5, marker='o', 
                    facecolors='r', edgecolors='r')
        
    '''
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
        
        # Mark sensor positions
        plt.scatter(x_vec[height_sensor_indices], y_vec[height_sensor_indices], 
                   s=15, marker='o', color='red')
    '''  
    # Collect sensor coordinates
    sensor_subtab = pd.DataFrame({
        'X_m': x_vec[height_sensor_indices],
        'Y_m': y_vec[height_sensor_indices],
        'Z_m': [h_m] * len(height_sensor_indices)
    })
    sensor_table = pd.concat([sensor_table, sensor_subtab], ignore_index=True)

np.save(save_folder + "U_error_n_s_per_plane_%d_AE_latent_%d_test_direction_%.2f_uniformGrid.npy" %(n_s_per_plane, latent_dim, float(test_direction)), np.array(err_list) )
## Save reconstructed wind field
np.save(save_folder + "U_rec_n_s_per_plane_%d_AE_latent_%d_test_direction_%.2f_uniformGrid.npy" %(n_s_per_plane, latent_dim, float(test_direction)), u_rec_combined)
    
##plt.suptitle(f'Multi-Height Wind Speed Comparison — {test_direction} (AE Method)')
plt.tight_layout()

out_fig = os.path.join(save_folder, 'U_error_n_s_per_plane_%d_AE_latent_%d_test_direction_%.2f_uniformGrid.png' %(n_s_per_plane, latent_dim, float(test_direction)))
plt.savefig(out_fig, dpi=300)

### Load trajectory data
trajectory1 = np.load('trajectory1.npy') ## 2D coordinates of the trajectory
trajectory2 = np.load('trajectory2.npy')

for iz, h_m in enumerate(test_heights):
    # Extract original, reconstructed data and error for this height
    start_idx = iz * num_pts
    end_idx = (iz + 1) * num_pts
    
    original_u = original_u_list[iz]
    u_rec = u_rec_combined[start_idx:end_idx]
    
    "=== Trajectory Visualization =="
    # Create trajectory coordinate mapping
    original_traj1 = []
    recon_traj1 = []
    original_traj2 = []
    recon_traj2 = []

    if iz == len(test_heights)-1: ## Specific plane
        # Interpolate and extract points along the trajectory
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
# Plot comparison for trajectories
# Global font settings (optional)
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

# Plot comparison chart
plt.figure(figsize=(10, 10), dpi=300)  # Set image resolution

# Original trajectory comparison
plt.subplot(2, 1, 1)
# Assuming data range is known, adjust based on actual data
x_range = range(len(original_traj1))
y_min, y_max = min(min(original_traj1), min(recon_traj1)), max(max(original_traj1), max(recon_traj1))

plt.plot(x_range, original_traj1, 'bo-', label='Trajectory 1 Original')
plt.plot(x_range, recon_traj1, 'ro-', label='Trajectory 1 Reconstructed')

plt.xlabel('Index', fontsize=20)  # Set x-axis label font
plt.ylabel('Wind Speed (m/s)', fontsize=20)  # Set y-axis label font
plt.legend(loc='upper right')  # Adjust legend location
plt.grid(True, linestyle='--', alpha=0.7)  # Adjust grid line style
plt.xlim(0, len(original_traj1))  # Set x-axis range
plt.ylim(y_min-1, y_max+1)  # Set y-axis range (with 1-unit margin)

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
plt.savefig(os.path.join(save_folder, 'trajectory_comparison_AE_nSensor_%d_nLatent_%d_nDirc_%d_1.png' %(n_s_per_plane, latent_dim, n_wind_dirc)), dpi=300, bbox_inches='tight')
plt.close()

np.save( os.path.join(save_folder, 'original_traj1_AE_1.npy'), original_traj1)
np.save( os.path.join(save_folder, 'recon_traj1_AE_1.npy'), recon_traj1)

np.save( os.path.join(save_folder, 'original_traj2_AE_1.npy'), original_traj2)
np.save( os.path.join(save_folder, 'recon_traj2_AE_1.npy'), recon_traj2)

# ===== Export results =====
# Sensor coordinates
##sensor_table.to_excel(os.path.join(root_folder, 'multi_height_sensor_coordinates_ae.xlsx'), index=False)

# Error analysis
combined_err_pct = 100 * np.abs(test_data_combined - u_rec_combined) / 10.0
mean_err = np.mean(combined_err_pct)

mean_table = pd.DataFrame({'MeanRelativeError (%)': [mean_err]})
##mean_table.to_excel(os.path.join(root_folder, 'multi_height_mean_relative_error_ae.xlsx'), index=False)

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimized_z': optimized_z.cpu(),
    'scaler': scaler,
    'sensor_idx_combined': sensor_idx_combined,
    'test_heights': test_heights
}, os.path.join(save_folder, 'multi_height_ae_wind_model.pth'))

print('Multi-height AE-based reconstruction completed!')
print(f'Mean Relative Error: {mean_err:.2f}%')