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
def gappy_pod_5heights_combined():
    # ========== 1) User-modifiable parts ==========
    # ========== 1) Basic parameter settings ==========  
    root_folder = "F:/data/Urban_wind_field_reconstruction_github/Urban_flow"  # Root directory
    save_folder = "images/"
    if not os.path.exists(save_folder):
           os.makedirs(save_folder)

    ##test_direction = '11.25'  # Test wind direction (pure number, e.g., '0', '90.0', '180.0', '270.0') 281.25
    test_direction = '281.25'
    test_heights = [10, 30, 50, 70, 90, 120]  # Test heights (meters)
    n_s_per_plane = 20 # Number of sensors per height: 2, 5, 10 (std), 20
    sensor_nums = [n_s_per_plane] * len(test_heights)  # Total number of sensors
    r = 10  # Number of POD modes: 2, 5, 10 (std), 20, 25
    ##============= Select number of wind directions (for POD, should be greater than number of modes) ===========##
    n_wind_dirc = 32 ## 8, 16, 32, 64, baseline 32
    
    # ========== 2) Read grid file ==========  
    grid_file = os.path.join(root_folder, 'grid_250k.csv')
    grid_raw = pd.read_csv(grid_file, header=None, skiprows=2, dtype=np.float64).values
    grid_data = grid_raw[0:, :]
    num_pts = grid_data.shape[0]
    # Assuming grid file format is consistent for all directions, with x/y columns fixed at index 2 and 3, 4 columns total
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
            if len(direction_data) < len(test_heights):
                print("Direction float(wind_angle) %f is problematic" %float(wind_angle))
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
    all_u_combined = np.vstack(all_u_combined).T  # Shape: (n²×5, training snapshots)
    n_train = all_u_combined.shape[1]
    print(f'Number of training wind directions: {len(training_dirs)}')
    print(f'Data dimensions: {all_u_combined.shape}')
    all_u_combined = all_u_combined[:, idx_list_wind_direction] ## All wind directions
    
    ## Select several wind directions
    ## Note: Since starting angle is 3.75 and final angle is 360 (close to each other), we set endpoint=False for uniform spacing
    float_points = np.linspace(0, all_u_combined.shape[1]-1, n_wind_dirc, endpoint=False) ## Equidistant, wind directions sorted numerically

    # Round to nearest integer and convert to integer list
    selected_cols = np.round(float_points).astype(int)

    all_u_combined = all_u_combined[:, selected_cols]
    
    # ========== 5) Multi-plane combined POD decomposition ==========
    # all_u_combined: (num_pts * num_heights) x (#snapshots)
    U_combined, S_combined, Vh_combined = la.svd(all_u_combined, full_matrices=False)
    U_r_combined = U_combined[:, :np.min((r, all_u_combined.shape[1]))] ## Take certain modes as basis
    
    energy_ratio = np.cumsum(S_combined**2) / np.sum(S_combined**2)
    r_domi = np.argmax(energy_ratio >= 0.99) + 1 ## Modes exceeding 99% energy
    print(f"Dominant mode number is {r_domi}")
    
    print(f'POD mode shape: {U_r_combined.shape}')    
    
    # ===== New energy curve plotting function =====
    try:
        # Ensure S_combined is not empty
        if S_combined.size == 0:
            raise ValueError("Singular values array is empty")
        
        # Calculate energy distribution
        total_energy = np.sum(S_combined**2)
        energy_spectrum = (S_combined**2) / total_energy  # Single mode energy ratio
        cumulative_energy = np.cumsum(energy_spectrum)    # Cumulative energy ratio
        
        # Create visualization chart
        plt.figure(figsize=(8, 6))
        
        # Plot single mode energy distribution (bar chart)
        plt.bar(range(1, len(energy_spectrum)+1), energy_spectrum, 
                width=0.6, color='blue', alpha=0.7, label='Single Mode Energy')
        
        # Plot cumulative energy curve (line chart)
        plt.plot(range(1, len(cumulative_energy)+1), cumulative_energy, 
                'r-o', markersize=5, linewidth=2, label='Cumulative Energy')
        
        # Add key reference line
        plt.axhline(y=0.99, color='#2ca02c', linestyle='--', 
                label='99% Energy Threshold')
        
        # Chart beautification
        ##plt.title('POD Modal Energy Spectrum Analysis', fontsize=14)
        plt.xlabel('Mode Index', fontsize=16)
        plt.ylabel('Normalized Energy Content', fontsize=16)
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.legend(loc='lower right', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        # Auto-adjust layout
        plt.tight_layout()
        
        # Save image (create directory if not exists)
        save_path = os.path.join('images', 'pod_energy_spectrum.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                transparent=True, pad_inches=0.1)
        
        print(f"Energy spectrum plot saved to: {save_path}")

    except Exception as e:
        print(f"Error generating energy spectrum plot: {str(e)}")
    
    # ========== 6) Multi-plane Gappy POD reconstruction for test wind direction ==========
    # Create sensor coordinates table
    sensor_table = pd.DataFrame(columns=['X_m', 'Y_m', 'Z_m'])
    
    # Create figure
    fig = plt.figure(figsize=(10, 2.5 * len(test_heights)))
    
    # Read and combine data for all heights of the test wind direction
    test_snapshot_combined = None
    test_data_list = []
    
    for iz, h_m in enumerate(test_heights):
        test_file = os.path.join(root_folder, test_fnames[iz])
        test_data = pd.read_csv(test_file, header=None, skiprows=1, dtype=np.float64).values
        
        if test_data.shape[0] != num_pts:
            raise ValueError(f'Test file {test_fnames[iz]} row count mismatch with grid points {num_pts}')
        
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
    
    # Select sensor positions in the entire multi-height space
    '''Random selection'''
    ###=======###
    
    total_sensors = sum(sensor_nums)
    combined_num_pts = num_pts * len(test_heights)
    
    # Randomly select sensor positions (in the entire combined space)
    idx_sens_combined = np.random.choice(combined_num_pts, size=total_sensors, replace=False)
    
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
    
    idx_sens_combined = np.array(sensor_indices_temp)
    '''
    #==================###
    # Gappy POD reconstruction (multi-plane combined)
    U_sub_combined = U_r_combined[idx_sens_combined, :]
    y_obs_combined = test_snapshot_combined[idx_sens_combined]
    a_combined = np.linalg.pinv(U_sub_combined) @ y_obs_combined
    u_rec_combined = U_r_combined @ a_combined
    
    # Calculate overall error
    err_pct_combined = 100 * np.abs(u_rec_combined - test_snapshot_combined) / 10.0
    ##err_pct_combined = 100 * np.abs(u_rec_combined - test_snapshot_combined) / (np.abs(test_snapshot_combined) + 0.0000001)
    print("err_pct_combined.shape ", err_pct_combined.shape)
    # ========== 7) Separate results for each height and visualize ==========

   
    for iz, h_m in enumerate(test_heights):
        # Extract original, reconstructed data and error for this height
        start_idx = iz * num_pts
        end_idx = (iz + 1) * num_pts
        
        original_u = test_data_list[iz]
        u_rec = u_rec_combined[start_idx:end_idx]
        err_pct = err_pct_combined[start_idx:end_idx]

        ##==============###
        # Extract sensor positions for this height
        height_sensor_indices = []
        for sens_idx in idx_sens_combined:
            if start_idx <= sens_idx < end_idx:
                height_sensor_indices.append(sens_idx - start_idx)
        
        # If the number of sensors for this height is insufficient, supplement with random selection
        num_sens = sensor_nums[iz]
        if len(height_sensor_indices) < num_sens:
            additional_needed = num_sens - len(height_sensor_indices)
            available_indices = list(set(range(num_pts)) - set(height_sensor_indices))
            additional_indices = np.random.choice(available_indices, size=additional_needed, replace=False)
            height_sensor_indices.extend(additional_indices)
        elif len(height_sensor_indices) > num_sens:
            height_sensor_indices = height_sensor_indices[:num_sens]
        
        height_sensor_indices = np.array(height_sensor_indices)
        
        # Reshape the results
        orig_2d = original_u.reshape(n, n)
        reco_2d = u_rec.reshape(n, n)
        err_2d = err_pct.reshape(n, n)
        
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
        
        # Collect sensor coordinates into table (including z)
        sensor_subtab = pd.DataFrame({
            'X_m': x_vec[height_sensor_indices],
            'Y_m': y_vec[height_sensor_indices],
            'Z_m': [h_m] * len(height_sensor_indices)
        })
        sensor_table = pd.concat([sensor_table, sensor_subtab], ignore_index=True)

    ##plt.suptitle(f'Multi-Height Gappy POD for {test_direction} (Combined POD)')
    plt.tight_layout()
    
    np.save(save_folder + "U_error_n_s_per_plane_%d_POD_latent_%d_test_direction_%.2f_uniformGrid.npy" %(n_s_per_plane, r, float(test_direction)), err_pct_combined)
    ## Save reconstructed wind field
    np.save(save_folder + "U_rec_n_s_per_plane_%d_POD_latent_%d_test_direction_%.2f_uniformGrid.npy" %(n_s_per_plane, r, float(test_direction)), u_rec_combined)
    
    # Save image
    out_fig = os.path.join(save_folder, 'U_error_n_s_per_plane_%d_POD_latent_%d_test_direction_%.2f_uniformGrid.png' %(n_s_per_plane, r, float(test_direction)))
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f'Image saved: {out_fig}')
    
    mean_err = np.mean(err_pct_combined)
    print(f'Mean Relative Error: {mean_err:.2f}%')
    plt.close()
    
    ### Load trajectory data
    trajectory1 = np.load('trajectory1.npy') ## 2D coordinates of the trajectory
    trajectory2 = np.load('trajectory2.npy')

    for iz, h_m in enumerate(test_heights):
        # Extract original, reconstructed data and error for this height
        start_idx = iz * num_pts
        end_idx = (iz + 1) * num_pts
        original_u = test_data_list[iz]
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

    # Create velocity field contour plot for the trajectory plane
    # Create velocity field contour plot
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    fig = plt.figure(figsize=(12, 10))
    
    # Plot velocity field contour (using wind speed magnitude)
    speed_mag = np.sqrt(original_u**2 + np.roll(original_u, 1, axis=0)**2)  # Assuming u is the east component
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(x, y, speed_mag.reshape(n, n), shading='auto', cmap='viridis', vmin=0, vmax=15)
    ##fig.colorbar(im, ax=ax, label='Wind Speed (m/s)')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Wind Speed (m/s)', fontsize=30, fontname='Times New Roman', fontweight='bold')  # Increase title font size
    cbar.ax.tick_params(labelsize=25)  # Increase tick label font size

    # Extract trajectory coordinates (convert to grid indices)
    traj1_x = []
    traj1_y = []
    traj2_x = []
    traj2_y = []
    
    for pt in trajectory1:
        distance = (x_vec - pt[0])**2 +  (y_vec - pt[1])**2
        pos_idx = np.argmin(distance)
        traj1_x.append(x_vec[pos_idx])
        traj1_y.append(y_vec[pos_idx])
    
    for pt in trajectory2:
        distance = (x_vec - pt[0])**2 +  (y_vec - pt[1])**2
        pos_idx = np.argmin(distance)
        traj2_x.append(x_vec[pos_idx])
        traj2_y.append(y_vec[pos_idx])
    
    # Plot trajectory curves
    ax.plot(traj1_x, traj1_y, color='blue', linestyle='-', linewidth=2, 
            marker='o', markersize=5, label='Trajectory 1')
    
    ax.plot(traj2_x, traj2_y, color='red', linestyle='--', linewidth=2, 
            marker='x', markersize=5, label='Trajectory 2')
    
    # Set graph properties
    ax.set_aspect('equal')
    ##ax.set_title(f'Wind Speed Field at {h_m}m with Trajectories')
    ax.set_xlabel('X Coordinate (m)', fontsize=30)
    ax.set_ylabel('Y Coordinate (m)', fontsize=30)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    # Save contour plot
    plt.savefig(os.path.join(save_folder, f'wind_speed_field_last_height_trajectories_1.png'), dpi=300)
    plt.close(fig)

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
    plt.savefig(os.path.join(save_folder, 'trajectory_comparison_POD_nSensor_%d_nLatent_%d_nDirc_%d_1.png' %(n_s_per_plane, r, n_wind_dirc)), dpi=300, bbox_inches='tight')
    plt.close()
    np.save( os.path.join(save_folder, 'original_traj1_POD_1.npy'), original_traj1)
    np.save( os.path.join(save_folder, 'recon_traj1_POD_1.npy'), recon_traj1)
    np.save( os.path.join(save_folder, 'original_traj2_POD_1.npy'), original_traj2)
    np.save( os.path.join(save_folder, 'recon_traj2_POD_1.npy'), recon_traj2)

    '''Preserve original wind field data'''
    # Initialize sensor coordinates table 
    sensor_table = pd.DataFrame(columns=['X_m', 'Y_m', 'Z_m'])

    for iz, h_m in enumerate(test_heights):
        # Extract data for this height (keep original logic unchanged)
        start_idx = iz * num_pts
        end_idx = (iz + 1) * num_pts
        original_u = test_data_list[iz]
        u_rec = u_rec_combined[start_idx:end_idx]
        err_pct = err_pct_combined[start_idx:end_idx]
        
        # Sensor position extraction (optimize calculation method)
        height_sensor_indices = np.where(
            np.isin(np.arange(start_idx, end_idx), idx_sens_combined)
        )[0]
        
        # Supplement random sensors (keep original logic)
        if len(height_sensor_indices) < sensor_nums[iz]:
            available = list(set(range(num_pts)) - set(height_sensor_indices))
            add_indices = np.random.choice(available, sensor_nums[iz]-len(height_sensor_indices), replace=False)
            height_sensor_indices = np.concatenate([height_sensor_indices, add_indices])
        height_sensor_indices = height_sensor_indices[:sensor_nums[iz]]

        # Generate grid data
        orig_2d = original_u.reshape(n, n)
        reco_2d = u_rec.reshape(n, n)
        err_2d = err_pct.reshape(n, n)

        # ========== Save original wind field plot ==========
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Plot original wind field
        im = ax.pcolormesh(x, y, orig_2d, shading='auto', cmap='rainbow', vmin=0, vmax=15)
        cb = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        
        # Set colorbar tick font (key modification)
        cb.ax.tick_params(axis='y', labelsize=18)  # Vertical direction ticks
        # Add geographic labels
        ax.scatter(x[bld_mask], y[bld_mask], s=2, 
                facecolors='white', edgecolors='k', label='Buildings')
        
        # Add sensor positions (red dots)
        ax.scatter(x_sens, y_sens, s=400, c='red', 
                edgecolors='black', label='Sensors')
        
        ax.tick_params(axis='both',          # Adjust both x/y axes
               which='both',         # Adjust both major/minor ticks
               labelsize=20,         # Tick label font size
               length=6,             # Tick line length
               width=1.5,            # Tick line width
               direction='out')      # Tick lines pointing out
        
        # Chart decoration
        ax.set_title(f'Original Wind Field @ {h_m}m', fontsize=18)
        ax.set_xlabel('X Coordinate (m)', fontsize=18)
        ax.set_ylabel('Y Coordinate (m)', fontsize=18)
        ax.legend(loc='upper right', fontsize=18)
        plt.savefig(os.path.join(save_folder, f'original_field_with_uniform_sensor_{h_m:.1f}m.png'), 
                dpi=300, bbox_inches='tight')
        
        plt.close(fig)

# Run function
if __name__ == "__main__":
    gappy_pod_5heights_combined()
# 运行函数
if __name__ == "__main__":
    gappy_pod_5heights_combined()