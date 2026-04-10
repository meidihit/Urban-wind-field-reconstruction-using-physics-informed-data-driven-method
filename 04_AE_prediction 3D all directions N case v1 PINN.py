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
import re

np.random.seed(42)  # Set global seed
# ===== User-modifiable parameters =====
# When running out of memory, you can try reducing the values of latent_dim (line 26), batch_size (line 27), and uniform_width (line 217).
# However, this will reduce model expressiveness (latent_dim), training stability (batch_size), and feature extraction capability (uniform_width), leading to decreased reconstruction accuracy.
latent_dim = 128  # Latent variable dimension
batch_size = 32
epochs = 5000
learning_rate = 1e-4 # Learning rate (too large may cause instability)
n_s_per_plane = 10  # Number of sensors per plane
num_optim_steps = 5000 # Number of latent variable optimization iterations
physics_loss_weight = 500.0  # Weight of physics loss (0 = disabled, larger values enforce stronger constraints)
#============= Reduce number of wind directions ===========
n_wind_dirc = 31 ## 16, 32, 48

# ===== Path settings =====
root_folder = "F:/test/urban_flow"  # Root directory
save_folder = "images/"
log_folder = 'log'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# 测试风向角度（直接使用数字）
test_direction_angle = '281.25'  # 测试风向的角度值

# Test wind direction (without height information)
test_heights = [10, 30, 70, 90, 120, 300]  # Test heights (meters)
sensor_nums = [n_s_per_plane] * len(test_heights)  # Total number of sensors
n_wind_dirc = 3  # Number of selected wind directions

# ========== 2) Read grid file ==========
grid_file = os.path.join(root_folder, "grid_250k.csv")

grid_raw = pd.read_csv(grid_file, header=None, skiprows=2, dtype=np.float64).values
grid_data = grid_raw[0:, :]
num_pts = grid_data.shape[0]

# 使用第1、2列作为x,y坐标（索引0和1）
x_vec = grid_data[:, 0].astype(float)
y_vec = grid_data[:, 1].astype(float)

n = int(np.sqrt(num_pts))
if n * n != num_pts:
    raise ValueError(f'Grid file point count {num_pts} cannot form a square matrix')

x = x_vec.reshape(n, n)
y = y_vec.reshape(n, n)

# ========== 3) Collect test file paths ==========
# 文件名格式：{高度}m.csv
test_fnames = []
for h_m in test_heights:
    pattern = os.path.join(root_folder, test_direction_angle, f'{h_m}m.csv')
    
    if not os.path.exists(pattern):
        raise FileNotFoundError(f'Test file does not exist: {pattern}')
    
    test_fnames.append(pattern)
    print(f"找到测试文件: {pattern}")

# ========== 4) Build training dataset ==========
# 遍历根目录下的所有文件夹，获取训练风向
training_directions = []

for item in os.listdir(root_folder):
    item_path = os.path.join(root_folder, item)
    if os.path.isdir(item_path) and item != 'CSV':
        try:
            angle = float(item)
            if str(angle) != test_direction_angle:
                training_directions.append(item)
                print(f"找到训练风向文件夹: {item}")
        except ValueError:
            continue

print(f'Found training wind directions: {training_directions}')

# Build training dataset
all_u_combined = []
effective_wind_directions = []

for direction in training_directions:
    direction_files = []
    direction_heights = []
    
    for h_m in test_heights:
        # 文件名格式：{高度}m.csv
        file_path = os.path.join(root_folder, direction, f'{h_m}m.csv')
        
        if os.path.exists(file_path):
            direction_files.append(file_path)
            direction_heights.append(h_m)
        else:
            print(f'Warning: Wind direction {direction} is missing data file for height {h_m}m')
            break
    
    if len(direction_files) == len(test_heights):
        direction_data = []
        
        for file_path in direction_files:
            try:
                data = pd.read_csv(file_path, header=None, skiprows=1, dtype=np.float64).values
                if data.shape[0] != num_pts:
                    warnings.warn(f'File {file_path} row count mismatch, skipping')
                    continue
                
                UV_field = np.zeros(num_pts * 2)
                if data.shape[1] >= 4:
                    mag = np.sqrt(np.sum(data[:, 1:4]**2, axis=1))
                else:
                    mag = data.flatten()
                
                UV_field[0:num_pts] = data[:, 1]  # u velocity
                UV_field[num_pts:] = data[:, 2]   # v velocity
                
                direction_data.append(UV_field)
            except Exception as e:
                warnings.warn(f'Error reading file {file_path}: {str(e)}')
                continue
        
        if len(direction_data) == len(test_heights):
            combined_data = np.hstack(direction_data)
            all_u_combined.append(combined_data)
            effective_wind_directions.append(float(direction))

if not all_u_combined:
    raise ValueError('No valid training data found')

# Sort wind directions
effective_wind_directions = np.array(effective_wind_directions)
idx_list_wind_direction = np.argsort(effective_wind_directions)
effective_wind_directions = effective_wind_directions[idx_list_wind_direction]

# Convert to final data structure
all_u_combined = np.vstack(all_u_combined).T
n_train = all_u_combined.shape[1]
print(f'Number of training wind directions: {n_train}')
print(f'Data dimensions: {all_u_combined.shape}')

# Select wind directions
float_points = np.linspace(0, all_u_combined.shape[1]-1, n_wind_dirc)
selected_cols = np.round(float_points).astype(int)
all_u_combined = all_u_combined[:, selected_cols]

rng = np.random.default_rng(0)
selected_cols = rng.choice(all_u_combined.shape[1], size=n_wind_dirc, replace=False)
float_points = np.linspace(0, all_u_combined.shape[1]-1, n_wind_dirc)
selected_cols = np.round(float_points).astype(int)
all_u_combined = all_u_combined[:, selected_cols]
    
# Standardize data
scaler = StandardScaler()
all_u_scaled = scaler.fit_transform(all_u_combined.T).T

# ===== Read test data (multiple heights) =====
print('Reading multi-height test data')
test_data_combined = None
original_u_list = []

for iz, h_m in enumerate(test_heights):
    test_file = test_fnames[iz]
    test_data = pd.read_csv(test_file, header=None, skiprows=1, dtype=np.float64).values
    
    if test_data.shape[0] != num_pts:
        raise ValueError(f'Test file {test_fnames[iz]} row count does not match grid points {num_pts}')
    
    original_u = np.zeros(num_pts * 2)
    original_u[0:num_pts] = test_data[:, 1] if test_data.shape[1] >= 4 else test_data.flatten()
    original_u[num_pts:] = test_data[:, 2] if test_data.shape[1] >= 4 else test_data.flatten()
    
    original_u_list.append(original_u)
    
    if test_data_combined is None:
        test_data_combined = original_u
    else:
        test_data_combined = np.concatenate((test_data_combined, original_u))

# Standardize test data
test_data_scaled = scaler.transform(test_data_combined.reshape(1, -1)).flatten()

# ===== Sensor selection =====
total_sensors = n_s_per_plane * len(test_heights)
combined_num_pts = num_pts * len(test_heights)

random.seed(42)
sensor_idx_combined = random.sample(range(combined_num_pts), total_sensors)
sensor_idx_combined = sorted(sensor_idx_combined)

# ===== AE model definition =====
class MultiHeightAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MultiHeightAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        uniform_width = 128
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, uniform_width),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(uniform_width, uniform_width),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(uniform_width, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, uniform_width),
            nn.ReLU(),
            nn.Linear(uniform_width, uniform_width),
            nn.ReLU(),
            nn.Linear(uniform_width, input_dim)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

# Compute 2D divergence
def calculate_physics_residual(u_field, v_field, n, dx, dy, residual):
    flag_batch = False
    
    if flag_batch:
        batch_size = u_field.shape[0]
        u_2d = u_field.view(batch_size, n, n)
        v_2d = v_field.view(batch_size, n, n)
    else:
        u_2d = u_field.view(n, n)
        v_2d = v_field.view(n, n)
        
    if flag_batch:
        dudx = torch.gradient(u_2d, dim=2, spacing=dx)[0]
        dvdy = torch.gradient(v_2d, dim=1, spacing=dy)[0]
    else:
        dudx = torch.gradient(u_2d, dim=1, spacing=dx)[0]
        dvdy = torch.gradient(v_2d, dim=0, spacing=dy)[0]
    
    divergence = dudx + dvdy
    divergence = divergence - residual
    return divergence
    
# ===== Train AE model =====
print('Training multi-height AE model')

X_train = torch.tensor(all_u_scaled.T, dtype=torch.float32)
X_val = torch.tensor(test_data_scaled[np.newaxis, :], dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MultiHeightAE(input_dim=X_train.shape[1], latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=0.00001, last_epoch=-1
)

def loss_function(recon_x, x):
    return nn.functional.mse_loss(recon_x, x)
    
x_tensor = torch.FloatTensor(x).to(device)
y_tensor = torch.FloatTensor(y).to(device)
dx = torch.abs(x_tensor[1, 0] - x_tensor[0, 0]).detach()
dy = torch.abs(y_tensor[0, 1] - y_tensor[0, 0]).detach()

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
        data = batch[0].to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    lr_scheduler.step()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            data = batch[0].to(device)
            recon_batch = model(data)
            val_loss += loss_function(recon_batch, data).item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    if epoch > 10:
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
plt.savefig(os.path.join(log_folder, 'Compression_training_loss_AE_wind_direction_%d_weighPhy_%.2f.png' %(n_wind_dirc, physics_loss_weight)), dpi=300)

# ===== Optimize latent variable =====
print('Optimizing latent variable for multi-height reconstruction')

sensor_measurements_scaled = np.zeros(total_sensors * 2)
test_data_scaled_u = np.zeros(combined_num_pts)
test_data_scaled_v = np.zeros(combined_num_pts) 

for h in range(len(test_heights)):
    test_data_scaled_u[h * num_pts: (h + 1) * num_pts] = test_data_scaled[h * (2 * num_pts): h * (2 * num_pts) + num_pts]
    test_data_scaled_v[h * num_pts: (h + 1) * num_pts] = test_data_scaled[h * (2 * num_pts) + num_pts: h * (2 * num_pts) + 2 * num_pts]
    
sensor_measurements_scaled[0:total_sensors] = test_data_scaled_u[sensor_idx_combined]
sensor_measurements_scaled[total_sensors:2 * total_sensors] = test_data_scaled_v[sensor_idx_combined]

model.eval()
with torch.no_grad():
    test_tensor = torch.tensor(test_data_scaled.reshape(1, -1), dtype=torch.float32).to(device)
    initial_z = model.encode(test_tensor).clone().detach().requires_grad_(True)
    
optimizer = optim.Adam([initial_z], lr=0.3)

def inverse_transform_gpu(tensor, scaler):
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        params = (scaler.mean_, scaler.scale_)
        op = lambda x, p: x * p[1] + p[0]
    elif hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
        params = (scaler.data_min_, scaler.data_max_)
        op = lambda x, p: x * (p[1] - p[0]) + p[0]
    else:
        raise ValueError("Unsupported scaler type")
    
    param_tensors = [
        torch.tensor(p, dtype=tensor.dtype, device=tensor.device) 
        for p in params
    ]
    
    return op(tensor, param_tensors)

loss_history = []
loss_history1 = []
physics_loss_history = []

for step in range(num_optim_steps):
    optimizer.zero_grad()
    
    reconstructed_field = model.decode(initial_z)
    reconstructed_field_orgScale = inverse_transform_gpu(reconstructed_field, scaler).flatten()
    
    reconstructed_measurements = torch.zeros(total_sensors * 2, dtype=torch.float32, device=device, requires_grad=False)
    
    rec_data_scaled_u = torch.zeros(combined_num_pts, dtype=torch.float32, device=device, requires_grad=False)
    rec_data_scaled_v = torch.zeros(combined_num_pts, dtype=torch.float32, device=device, requires_grad=False)
    
    rec_data_orgScale_u = torch.zeros(combined_num_pts, dtype=torch.float32, device=device, requires_grad=False)
    rec_data_orgScale_v = torch.zeros(combined_num_pts, dtype=torch.float32, device=device, requires_grad=False)
    
    for h in range(len(test_heights)):
        rec_data_scaled_u[h * num_pts: (h + 1) * num_pts] = reconstructed_field[0, h * (2 * num_pts): h * (2 * num_pts) + num_pts]
        rec_data_scaled_v[h * num_pts: (h + 1) * num_pts] = reconstructed_field[0, h * (2 * num_pts) + num_pts: h * (2 * num_pts) + 2 * num_pts]
        rec_data_orgScale_u[h * num_pts: (h + 1) * num_pts] = reconstructed_field_orgScale[h * (2 * num_pts): h * (2 * num_pts) + num_pts]
        rec_data_orgScale_v[h * num_pts: (h + 1) * num_pts] = reconstructed_field_orgScale[h * (2 * num_pts) + num_pts: h * (2 * num_pts) + 2 * num_pts]
    
    reconstructed_measurements[0:total_sensors] = rec_data_scaled_u[sensor_idx_combined]
    reconstructed_measurements[total_sensors:2 * total_sensors] = rec_data_scaled_v[sensor_idx_combined]
    
    true_measurements = torch.tensor(sensor_measurements_scaled, dtype=torch.float32).to(device)
    rec_loss = nn.functional.mse_loss(reconstructed_measurements, true_measurements)
    
    reconstructed_all = reconstructed_field[0, :]
    true_all = torch.tensor(test_data_scaled, dtype=torch.float32).to(device)
    rec_loss_overall = nn.functional.mse_loss(reconstructed_all, true_all)
    
    num_heights = len(test_heights)
    test_data_combined_tensor = torch.tensor(test_data_combined, dtype=torch.float32, device=device, requires_grad=False)
    
    physics_loss = 0.0
    for h in range(num_heights):
        residual_CFD = calculate_physics_residual(
            test_data_combined_tensor[h * (2 * num_pts): h * (2 * num_pts) + num_pts],
            test_data_combined_tensor[h * (2 * num_pts) + num_pts: h * (2 * num_pts) + 2 * num_pts],
            n, dx, dy, 0.0
        )
        
        residual_temp = calculate_physics_residual(
            rec_data_orgScale_u[h * num_pts: (h + 1) * num_pts],
            rec_data_orgScale_v[h * num_pts: (h + 1) * num_pts],
            n, dx, dy, residual_CFD
        )
        
        physics_loss += torch.mean(residual_temp ** 2)
        
    physics_loss /= num_heights
    
    total_loss = rec_loss + physics_loss_weight * physics_loss
    total_loss.backward()
    optimizer.step()
    
    loss_history.append(rec_loss.item())
    loss_history1.append(rec_loss_overall.item())
    physics_loss_history.append(physics_loss.item())
    
    if step % 100 == 0:
        print(f'Optimization step {step}/{num_optim_steps}, Rec Loss: {rec_loss.item():.6f}, Physics Loss: {physics_loss.item():.6f}')

# Global font parameter settings
plt.rcParams.update({
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 25,
    'font.family': 'sans-serif',
    'font.weight': 'semibold'
})

# Plot loss curve
plt.figure(figsize=(9, 6))
plt.plot(loss_history, label='Sensor Reconstruction Error')
plt.plot(loss_history1, label='Full-field Reconstruction Error')
plt.plot(physics_loss_history, label='Physics Loss')
plt.yscale('log')
plt.xlabel('Optimization Steps')
plt.ylabel('Loss')
plt.title('Latent Variable Optimization Process')
plt.legend(fontsize=20, loc='best')
plt.grid(True, which="both", ls="-")
plt.ylim(0.0001, 0.2)
plt.savefig(os.path.join(log_folder, 'latent_optimization_loss_weight_physics_%.2f.png' % physics_loss_weight), dpi=300)

# Final reconstruction
with torch.no_grad():
    optimized_z = initial_z.detach()
    u_rec_scaled = model.decode(optimized_z).cpu().numpy()
    u_rec_combined = scaler.inverse_transform(u_rec_scaled).flatten()

# ===== Separate results for each height and analyze =====
fig = plt.figure(figsize=(10, 2.5 * len(test_heights)))
err_list = []

for iz, h_m in enumerate(test_heights):
    start_idx = iz * num_pts * 2
    end_idx = (iz + 1) * num_pts * 2
    
    original_u = original_u_list[iz]
    u_rec = u_rec_combined[start_idx:end_idx]
    
    err_pct = 100 * np.abs(original_u - u_rec) / 10.0
    err_list.extend(err_pct.tolist())
    
    height_sensor_indices = []
    for sens_idx in sensor_idx_combined:
        if int(start_idx / 2) <= sens_idx < int(start_idx / 2) + num_pts:
            height_sensor_indices.append(sens_idx - int(start_idx / 2))
    
    if len(height_sensor_indices) < n_s_per_plane:
        additional_needed = n_s_per_plane - len(height_sensor_indices)
        available_indices = list(set(range(num_pts)) - set(height_sensor_indices))
        additional_indices = np.random.choice(available_indices, size=additional_needed, replace=False)
        height_sensor_indices.extend(additional_indices)
    
    height_sensor_indices = np.array(height_sensor_indices)
    
    original_u = np.sqrt((original_u[0:num_pts] ** 2 + original_u[num_pts:2 * num_pts] ** 2) / 2)
    u_rec = np.sqrt((u_rec[0:num_pts] ** 2 + u_rec[num_pts:2 * num_pts] ** 2) / 2)
    err_pct = np.sqrt((err_pct[0:num_pts] ** 2 + err_pct[num_pts:2 * num_pts] ** 2) / 2)
    
    orig_2d = original_u.reshape(n, n)
    reco_2d = u_rec.reshape(n, n)
    err_2d = err_pct.reshape(n, n)
    
    ax1 = plt.subplot(len(test_heights), 3, iz*3 + 1)
    im1 = ax1.pcolormesh(x, y, orig_2d, shading='auto')
    ax1.set_aspect('equal')
    ax1.set_title(f'Original @ {h_m}m')
    plt.colorbar(im1, ax=ax1)
    im1.set_clim(0, 15)
    
    ax2 = plt.subplot(len(test_heights), 3, iz*3 + 2)
    im2 = ax2.pcolormesh(x, y, reco_2d, shading='auto')
    ax2.set_aspect('equal')
    ax2.set_title(f'Reconstructed @ {h_m}m')
    plt.colorbar(im2, ax=ax2)
    im2.set_clim(0, 15)
    
    ax3 = plt.subplot(len(test_heights), 3, iz*3 + 3)
    im3 = ax3.pcolormesh(x, y, err_2d, shading='auto')
    ax3.set_aspect('equal')
    ax3.set_title('Error (%)')
    plt.colorbar(im3, ax=ax3)
    im3.set_clim(0, 100)
    
    bld_mask = (orig_2d == 0)
    for ax in [ax1, ax2, ax3]:
        ax.scatter(x[bld_mask], y[bld_mask], s=1, marker='s', facecolors='white', edgecolors='none', alpha=0.5)
    
    x_sens = x_vec[height_sensor_indices]
    y_sens = y_vec[height_sensor_indices]
    for ax in [ax1, ax2, ax3]:
        ax.scatter(x_sens, y_sens, s=5, marker='o', facecolors='r', edgecolors='r')

np.save(save_folder + "U_error_n_s_per_plane_%d_AE_nCase_%d_weighPhy_%.2f.npy" % (n_s_per_plane, n_wind_dirc, physics_loss_weight), np.array(err_list))
np.save(save_folder + "U_rec_n_s_per_plane_%d_AE_nCase_%d_weighPhy_%.2f.npy" % (n_s_per_plane, n_wind_dirc, physics_loss_weight), u_rec_combined)

plt.tight_layout()
out_fig = os.path.join(save_folder, 'U_error_n_s_per_plane_%d_AE_nCase_%d_weighPhy_%.2f.png' % (n_s_per_plane, n_wind_dirc, physics_loss_weight))
plt.savefig(out_fig, dpi=300)

# ===== Export results =====
# Error analysis
combined_err_pct = np.abs(test_data_combined - u_rec_combined)
mean_err = np.mean(combined_err_pct)
max_err = np.max(combined_err_pct)
std_err = np.std(combined_err_pct)
median_err = np.median(combined_err_pct)
q95_err = np.percentile(combined_err_pct, 95)

# Global font parameter settings
plt.rcParams.update({
    'axes.labelsize': 20,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'font.family': 'sans-serif',
    'font.weight': 'semibold'
})

plt.figure(figsize=(8, 6))
plt.hist(combined_err_pct, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(mean_err, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.2f} m/s')
plt.axvline(median_err, color='green', linestyle='--', linewidth=2, label=f'Median: {median_err:.2f} m/s')
plt.xlabel('Absolute Error (m/s)')
plt.ylabel('Probability Density')
plt.title('Error statistics distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.0, 1.5)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, 'Error_statistics_weighPhy_%.2f.png' % physics_loss_weight), dpi=300)

print('Multi-height AE-based reconstruction completed!')
print(f'Mean Error: {mean_err:.2f} m/s')