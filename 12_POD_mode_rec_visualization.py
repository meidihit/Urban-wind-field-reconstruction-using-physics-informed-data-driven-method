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
import seaborn as sns
from scipy import stats

# ===== User-modifiable parameters =====
root_folder = 'your path/Urban_flow'
# ===== Path settings =====
csv_folder = os.path.join(root_folder, 'CSV')
grid_file = os.path.join(csv_folder, 'grid_250k.csv')
test_filename = 'northwest_281.25_120m.csv'

# ===== Read grid coordinates =====
grid_data = pd.read_csv(grid_file, header=None, skiprows=2, dtype=np.float64).values
n = int(np.sqrt(grid_data.shape[0]))
assert n * n == grid_data.shape[0], "Grid file row count must be a square matrix"

# ===== Read all data to build POD basis =====
all_files = glob.glob(os.path.join(csv_folder, '*.csv'))
all_u = []
test_path = ''

for file in all_files:
    fname = os.path.basename(file)
    if fname == 'grid_250k.csv':
        continue
    if fname == test_filename:
        test_path = file
        continue
    
    data = pd.read_csv(file, header=None, skiprows=1, dtype=np.float64).values
    all_u.append(np.linalg.norm(data[:, 1:4], axis=1))

all_u = np.column_stack(all_u)

# ===== Read test case =====
test_data = pd.read_csv(test_path, header=None, skiprows=1, dtype=np.float64).values
original_u = np.linalg.norm(test_data[:, 1:4], axis=1)

# ===== Coordinate processing =====
name = os.path.splitext(test_filename)[0]
test_dir = name.split('_')[0]
coord_columns = [0,1] if test_dir in ['south','north'] else [2,3]
x_vec = grid_data[:, coord_columns[0]]
y_vec = grid_data[:, coord_columns[1]]

x = x_vec.reshape((n, n), order='F')
y = y_vec.reshape((n, n), order='F')

# ===== POD decomposition =====
U, s, VT = svd(all_u, full_matrices=False)
energy_ratio = np.cumsum(s**2) / np.sum(s**2)
r = 5  # Keep the first 5 modes
U_r = U[:, :r]

# ===== Mode visualization function =====
def plot_psi_mode(mode_idx, x, y, U_r, n, output_path):
    """Save the vector field distribution of a single POD mode"""
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    
    psi = U_r[:, mode_idx].reshape((n, n), order='F')
    
    # Create color map
    norm = Normalize(vmin=-np.max(psi), vmax=np.max(psi))
    cmap = plt.get_cmap('coolwarm')  # Blue-white-red gradient
    
    # Plot mode distribution
    im = ax.pcolormesh(x, y, psi, shading='auto', cmap=cmap, norm=norm)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add energy labels
    energy = 100 * s[mode_idx]**2 / np.sum(s**2)
    #ax.text(0.05, 0.95, f'{energy:.1f}%', transform=ax.transAxes,
    #        fontsize=10, verticalalignment='top', color='white')
    
    # Add color bar
    #cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    #cbar.set_label('Amplitude', fontsize=10, labelpad=5)
    
    # Save as vector image
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# ===== Save the first 5 POD modes =====
output_dir = os.path.join(root_folder, 'pod_modes')
os.makedirs(output_dir, exist_ok=True)

for mode_idx in range(r):
    output_path = os.path.join(output_dir, f'pod_mode_{mode_idx+1}.png')
    plot_psi_mode(mode_idx, x, y, U_r, n, output_path)

# ===== Gappy POD reconstruction =====
num_sensors = 30
total_numbers = n ** 2
sampled_numbers = random.sample(range(total_numbers), num_sensors)
idx = sorted(sampled_numbers)

u_test = original_u[idx]
coeff = pinv(U_r[idx, :]) @ u_test
u_rec = U_r @ coeff

# ===== Error analysis =====
err_pct = 100 * np.abs(original_u - u_rec) / 10

# ===== Main field comparison visualization =====
Xq, Yq = np.meshgrid(np.linspace(x.min(), x.max(), n),
                     np.linspace(y.min(), y.max(), n))
err_int = griddata((x_vec, y_vec), err_pct, (Xq, Yq), method='linear')

fig = plt.figure(figsize=(18, 6))
titles = ['Original Field', 'Reconstructed Field', 'Relative Error']
data = [original_u.reshape(n,n,order='F'), 
        u_rec.reshape(n,n,order='F'), 
        err_int]

for i in range(3):
    ax = fig.add_subplot(131 + i)
    cmap = plt.get_cmap('coolwarm')  # Blue-white-red gradient
    im = ax.pcolormesh(x, y, data[i], shading='auto', cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titles[i], fontsize=14)
    
    # Mark building areas
    bldg_mask = (original_u.reshape(n,n,order='F') == 0)
    ax.scatter(x[bldg_mask], y[bldg_mask], s=20, color='gray', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'field_comparison.png'), dpi=300)
plt.close()