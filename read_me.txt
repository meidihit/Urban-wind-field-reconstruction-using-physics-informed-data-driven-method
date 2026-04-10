Instructions for Running the Code

(Code files are named in the format "number\_main content". In the following description, code files are referred to by their number only.)

Note: Before running Code 05, 06, 07, or 08, Code 1 must be executed to generate the two required trajectory .npy files. After running Code 05, 06, 07, and 08, Code 09, 10, 11, and 12 can be executed for analysis.

Additionally, pay attention to the root directory and output paths. The directory separators for Linux and Windows systems are / and \, respectively. The default output directory is the log and images folders created under the directory where the code files are located.

Below are the purpose, output content, and notes for Code 01 through Code 12.

Code 01 (Excel → Trajectory Plot)
   Output: Saves .npy files (trajectory1.npy, trajectory2.npy) and visualizes two trajectories (X–Y plane plots).
   Use: Provides trajectory data for subsequent wind field analysis or sampling.

Code 02 (Collect Training Wind Fields → POD Preparation → Wind Direction Correlation Matrix)
   Output: Saves images/wind\_correlation\_matrix.png (correlation heatmap of wind directions).
   Use: Processes wind field data and checks correlations between different wind directions to guide selection of effective directions.

Code 03 (Wind Field Data Reading → Batch Visualization)
   Output: Saves images/Wind\_field\_of\_selected\_directions\_1.png, \_2.png, etc. (saved in batches of 24 wind directions).
   Use: Visualizes the wind fields of the entire training set or selected directions, providing an intuitive check before model training.

Codes 04, 05, and 06 are based on the low-dimensional manifold assumption. They use autoencoders（AE） to learn the latent space of urban multi-height wind fields, and then invert the latent space using a small number of sensor measurements to reconstruct the full 3D wind field. 
      Code 04 reconstructs the directional u and v velocity components and explicitly incorporates a physics-based constraint for mass conservation.
      Code 05 simplifies the reconstruction to the scalar wind speed magnitude without any physical constraints.
Code 06 extends this by adding a sensor‑to‑latent mapping network (end‑to‑end regression).
      Code 06, building on Code 05, additionally trains a neural network to directly predict the initial latent variables from sensor measurements, making it more suitable for real-world deployment scenarios where full-field data is unavailable to initialize the latent variables.

Code 07 and 08 – These implement, respectively, the classical Gappy POD reconstruction method (low‑rank linear subspace + least‑squares / pseudo‑inverse solution) and the Gappy POD + machine learning regression (POD‑NN) method.

Code 09 – Compares the original data of two trajectories (Trajectory 1 and Trajectory 2) with reconstructions from the four different methods.

Code 10 and 11 – These visualize the reconstruction results of the AE‑NN method (deep learning) and the POD method (linear dimensionality reduction), respectively. The parameters can be modified as needed.

Code 12 – Implements the Gappy POD method, which reconstructs the entire urban wind field using a small number of sensor measurements (e.g., 30 points) and visualizes the spatial distribution of the first r POD modes.

