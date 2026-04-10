import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ==========================================================
# 1) Basic parameters
# ==========================================================
root_folder = "your path/Urban_flow"
save_folder = "images"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
test_direction = '281.25'                  # Direction excluded from training
test_heights = [10, 30, 50, 70, 90, 120]  # Heights (m)

# Optional: manually specify wind directions
# If None -> use all available directions
selected_dirs = None  # selected_dirs = ['0.0', '30.0', '60.0']

# Ensure output folder exists
os.makedirs(save_folder, exist_ok=True)

# ==========================================================
# 2) Read grid data
# ==========================================================
grid_file = os.path.join(root_folder, 'grid_250k.csv')

grid_raw = pd.read_csv(
    grid_file,
    header=None,
    skiprows=2,
    dtype=np.float64
).values

grid_data = grid_raw[:, :]

# Extract coordinates (assumed columns)
x_vec = grid_data[:, 2].astype(float)
y_vec = grid_data[:, 3].astype(float)

# Apply 90-degree counterclockwise rotation
x_vec, y_vec = -y_vec, x_vec

# Ensure grid is square
n = int(np.sqrt(len(x_vec)))
if n * n != len(x_vec):
    raise ValueError("Grid points cannot form a square matrix")

x = x_vec.reshape(n, n)
y = y_vec.reshape(n, n)

# ==========================================================
# 3) Scan valid wind directions
# ==========================================================
training_dirs = []

for d in os.listdir(root_folder):
    dir_path = os.path.join(root_folder, d)
    if os.path.isdir(dir_path) and d != test_direction:
        training_dirs.append(dir_path)

effective_dirs = []
effective_dirs_str = []

for train_dir in training_dirs:
    direction = os.path.basename(train_dir)
    valid = True
    for h in test_heights:
        file_path = os.path.join(train_dir, f"{h}m.csv")
        if not os.path.exists(file_path):
            valid = False
            break
    if valid:
        effective_dirs.append(float(direction))
        effective_dirs_str.append(direction)

# Sort directions
effective_dirs = np.array(effective_dirs)
idx_sort = np.argsort(effective_dirs)
effective_dirs = effective_dirs[idx_sort]
effective_dirs_str_sorted = [effective_dirs_str[i] for i in idx_sort]

# ==========================================================
# 4) Select wind directions
# ==========================================================
if selected_dirs is None:
    selected_dirs = effective_dirs_str_sorted
else:
    selected_dirs = [str(d) for d in selected_dirs]
    valid_set = set(effective_dirs_str_sorted)
    invalid = [d for d in selected_dirs if d not in valid_set]
    if invalid:
        raise ValueError(f"Invalid directions: {invalid}")

print(f"Selected directions: {selected_dirs}")

# ==========================================================
# 5) Load wind field data
# ==========================================================
speed_data = []
uv_data = []

for direction in selected_dirs:
    dir_path = os.path.join(root_folder, direction)

    direction_speeds = []
    direction_uv = []

    for h in test_heights:
        file_path = os.path.join(dir_path, f"{h}m.csv")
        if not os.path.exists(file_path):
            warnings.warn(f"{direction} missing {h}m")
            continue

        data = pd.read_csv(
            file_path,
            header=None,
            skiprows=1,
            dtype=np.float64
        ).values

        if data.shape[0] != x.size:
            warnings.warn(f"{file_path} size mismatch")
            continue

        if data.shape[1] < 4:
            raise ValueError(f"{file_path} missing velocity components")

        # Extract velocity components
        u = data[:, 1]
        v = data[:, 2]

        # Compute wind speed magnitude
        speed = np.sqrt(np.sum(data[:, 1:4]**2, axis=1))

        # Reshape to grid
        direction_speeds.append(speed.reshape(x.shape))
        direction_uv.append((u.reshape(x.shape), v.reshape(x.shape)))

    # Only keep complete directions
    if len(direction_speeds) == len(test_heights):
        speed_data.append(np.stack(direction_speeds))
        uv_data.append(direction_uv)

speed_data = np.array(speed_data)

# ==========================================================
# 6) Visualization in batches (24 directions per figure)
# ==========================================================
n_dirs_total = len(selected_dirs)
n_heights = len(test_heights)
batch_size = 24  # 24 rows per figure

for batch_start in range(0, n_dirs_total, batch_size):
    batch_end = min(batch_start + batch_size, n_dirs_total)
    batch_dirs = selected_dirs[batch_start:batch_end]

    n_batch = len(batch_dirs)

    fig, axes = plt.subplots(
        n_batch,
        n_heights,
        figsize=(2 * n_heights, 2 * n_batch)
    )

    axes = np.atleast_2d(axes)

    for d_idx, direction in enumerate(batch_dirs):
        print(f"Plotting {batch_start + d_idx +1}/{n_dirs_total} ({direction}°)")
        for h_idx, h in enumerate(test_heights):
            ax = axes[d_idx, h_idx]
            speed = speed_data[batch_start + d_idx, h_idx]
            u, v = uv_data[batch_start + d_idx][h_idx]

            # Plot wind speed
            im = ax.pcolormesh(x, y, speed.astype(np.float32), cmap='viridis', shading='auto')

            # Mark building regions
            bld_mask = speed < 1e-6
            if np.any(bld_mask):
                ax.scatter(
                    x[bld_mask],
                    y[bld_mask],
                    s=1,
                    c='white',
                    alpha=0.5
                )

            ax.set_aspect('equal')

            # Titles and labels
            if d_idx == 0:
                ax.set_title(f"{h} m", fontsize=10)
            if h_idx == 0:
                ax.set_ylabel(f"{direction}°", fontsize=10)

            # Colorbar per subplot
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            plt.colorbar(im, cax=cax)

    plt.tight_layout()
    save_path = os.path.join(save_folder, f'Wind_field_of_selected_directions_{batch_start//batch_size +1}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")