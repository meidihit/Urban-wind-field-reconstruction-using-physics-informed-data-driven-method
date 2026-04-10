import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def remove_nan_rows(arr: np.ndarray) -> np.ndarray:
    """
    Remove rows containing NaN in a 2D NumPy array.
    
    Parameters:
        arr (np.ndarray): Input array, shape should be [N, 2] or [N, M].
    
    Returns:
        np.ndarray: New array after removing NaN rows.
    """
    # Input validation
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy array")
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D")
    
    # Use ~np.isnan().any(axis=1) to filter out rows containing NaN
    mask = ~np.isnan(arr).any(axis=1)
    return arr[mask]

# 1. Read Excel file
file_path = "your path/2path_coordinate_120m_3.xlsx"  # Please ensure the file path is correct
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 2. Extract coordinates for two trajectories
# Assuming column names are 'path1_X', 'path1_Y', 'path2_X', 'path2_Y', consistent with documentation
trajectory1 = df[['path1_X', 'path1_Y']].to_numpy()
trajectory2 = df[['path2_X', 'path2_Y']].to_numpy()
trajectory1 = remove_nan_rows(trajectory1)
trajectory2 = remove_nan_rows(trajectory2)

# 3. Save as .npy files
np.save('trajectory1.npy', trajectory1)
np.save('trajectory2.npy', trajectory2)
print("Trajectory data saved as 'trajectory1.npy' and 'trajectory2.npy'.")

# 4. Plot trajectories
plt.figure(figsize=(10, 8))
plt.plot(trajectory1[:, 0], trajectory1[:, 1], 'b-', label='Trajectory 1', linewidth=2)
plt.plot(trajectory2[:, 0], trajectory2[:, 1], 'r-', label='Trajectory 2', linewidth=2)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Two Trajectories Visualization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')  # Ensure consistent scaling for x and y axes to prevent distortion
plt.show()
