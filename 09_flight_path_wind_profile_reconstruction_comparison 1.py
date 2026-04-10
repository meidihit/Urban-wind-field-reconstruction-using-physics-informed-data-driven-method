import numpy as np
import matplotlib.pyplot as plt
import os

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Set folder path for saving data
save_folder = "images"

# Load saved data (profile 1)
original_traj1_POD = np.load(os.path.join(save_folder, 'original_traj1_POD_1.npy'))

# Reconstruction results of four methods - corrected filename errors
recon_traj1_POD = np.load(os.path.join(save_folder, 'recon_traj1_POD_1.npy'))
recon_traj1_POD_NN = np.load(os.path.join(save_folder, 'recon_traj1_POD_NN_1.npy'))
recon_traj1_AE = np.load(os.path.join(save_folder, 'recon_traj1_AE_1.npy'))  # Corrected filename
recon_traj1_AE_NN = np.load(os.path.join(save_folder, 'recon_traj1_AE_NN_1.npy'))  # Corrected filename

# Load saved data (profile 2)
original_traj2_POD = np.load(os.path.join(save_folder, 'original_traj2_POD_1.npy'))

# Reconstruction results of four methods - corrected filename errors
recon_traj2_POD = np.load(os.path.join(save_folder, 'recon_traj2_POD_1.npy'))
recon_traj2_POD_NN = np.load(os.path.join(save_folder, 'recon_traj2_POD_NN_1.npy'))
recon_traj2_AE = np.load(os.path.join(save_folder, 'recon_traj2_AE_1.npy'))
recon_traj2_AE_NN = np.load(os.path.join(save_folder, 'recon_traj2_AE_NN_1.npy'))

# Create canvas
plt.figure(figsize=(25, 10), dpi=300)

# First subplot: Trajectory 1 comparison
plt.subplot(1, 2, 1)
x_range1 = range(len(original_traj1_POD))

# Calculate data range for y-axis boundaries
all_data1 = np.concatenate([
    original_traj1_POD, 
    recon_traj1_POD, 
    recon_traj1_POD_NN, 
    recon_traj1_AE, 
    recon_traj1_AE_NN
])
y_min1, y_max1 = np.min(all_data1), np.max(all_data1)

# Plot each curve with different styles
plt.plot(x_range1, original_traj1_POD, 'k-o', linewidth=2.5, markersize=6, label='Original')
plt.plot(x_range1, recon_traj1_POD, 'r--s', linewidth=2, markersize=5, label='POD')
plt.plot(x_range1, recon_traj1_POD_NN, 'g-.^', linewidth=2, markersize=5, label='POD + NN')
plt.plot(x_range1, recon_traj1_AE, 'b:*', linewidth=2, markersize=6, label='AE')
plt.plot(x_range1, recon_traj1_AE_NN, 'm--d', linewidth=2, markersize=5, label='AE + NN')

plt.xlabel('Index', fontsize=35)
plt.ylabel('Wind Speed (m/s)', fontsize=35)
plt.legend(loc='best', fontsize=30)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, len(original_traj1_POD)-1)
plt.ylim(y_min1-0.5, y_max1+0.5)

# Unified setting for tick font
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

# Second subplot: Trajectory 2 comparison
plt.subplot(1, 2, 2)
x_range2 = range(len(original_traj2_POD))

# Calculate data range for y-axis boundaries
all_data2 = np.concatenate([
    original_traj2_POD, 
    recon_traj2_POD, 
    recon_traj2_POD_NN, 
    recon_traj2_AE, 
    recon_traj2_AE_NN
])
y_min2, y_max2 = np.min(all_data2), np.max(all_data2)

# Plot each curve with different styles
plt.plot(x_range2, original_traj2_POD, 'k-o', linewidth=2.5, markersize=6, label='Original')
plt.plot(x_range2, recon_traj2_POD, 'r--s', linewidth=2, markersize=5, label='POD')
plt.plot(x_range2, recon_traj2_POD_NN, 'g-.^', linewidth=2, markersize=5, label='POD + NN')
plt.plot(x_range2, recon_traj2_AE, 'b:*', linewidth=2, markersize=6, label='AE')
plt.plot(x_range2, recon_traj2_AE_NN, 'm--d', linewidth=2, markersize=5, label='AE + NN')

plt.xlabel('Index', fontsize=35)
plt.ylabel('Wind Speed (m/s)', fontsize=35)
plt.legend(loc='best', fontsize=30)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, len(original_traj2_POD)-1)
plt.ylim(y_min2-0.5, y_max2+0.5)

# Unified setting for tick font
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

# Adjust layout and save
plt.tight_layout()
output_path = os.path.join(save_folder, 'enhanced_wind_profile_comparison_n_1.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Enhanced comparison plot saved to: {output_path}")
