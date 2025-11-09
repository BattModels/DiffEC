import jax.numpy as jnp
import csv
import matplotlib.pyplot as plt

def generate_mesh(N, L):
    """
    Generate a finite volume mesh for the 1D domain.
    Args:
        N (int): Number of cells.
        L (float): Length of the domain.
    Returns:
        x (ndarray): Coordinates of cell centers.
    """
    dx = L / N
    x = (jnp.arange(N)+0.5) * dx
    return x

def csv_to_numpy(filename):
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Strip spaces and convert to float
            data.append([float(item.strip()) for item in row])
    return jnp.array(data)

def load_c_data():
    filenames = ['t=0020.csv', 't=0151.csv', 't=0314.csv', 't=0412.csv', 't=0511.csv', 't=0609.csv', 't=0707.csv', 't=0871.csv', 't=1100.csv']
    c_data = []
    for filename in filenames:
        c_data.append(csv_to_numpy('data/' + filename))

    return c_data

def interp_c_data(c_data, N, L):
    time_indices = jnp.array([20, 151, 314, 412, 511, 609, 707, 871, 1100]) * 600

    x = 10 * generate_mesh(N, L) # Length in mm
    condition = jnp.logical_and(x >= 0.6, x <=2.4)
    space_indices = jnp.where(condition)[0]
    x_interp = x[condition]

    c_data_interp = []
    for i in range(len(c_data)):
        raw_data = c_data[i]
        c_data_interp.append(jnp.interp(x_interp, raw_data[:, 0], raw_data[:, 1]))
    c_data_interp = jnp.stack(c_data_interp)

    return c_data_interp, x_interp, time_indices, space_indices

def load_v0_data():
    filenames = ['x=0.95.csv', 'x=1.30.csv', 'x=1.65.csv', 'x=2.00.csv', 'x=2.35.csv']
    vel_data = []
    for filename in filenames:
        vel_data.append(csv_to_numpy('data/' + filename))
    
    return vel_data

def get_pos_indices(dx):
    pos_indices = ((jnp.array([0.95, 1.30, 1.65, 2.00, 2.35]) * 0.1 - dx / 2) / dx).astype(jnp.int32)
    return pos_indices

def plot_results(X, T, c_arr, v0_arr_c):
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    contour_c = ax[0].contourf(X, T, c_arr, cmap='viridis')
    ax[0].set_xlabel('x (mm)')
    ax[0].set_ylabel('Time (h)')
    ax[0].set_title('Concentration Distribution Over Time')
    ax[0].set_aspect(1/4)

    ax[1].contourf(X, T, v0_arr_c * 1e7, cmap='rainbow')
    ax[1].set_xlabel('x (mm)')
    ax[1].set_ylabel('Time (h)')
    ax[1].set_title('Solvent Velocity Over Time')
    ax[1].set_aspect(1/4)

    plt.tight_layout()  # Adjust layout to prevent overlap
    cbar = plt.colorbar(contour_c, ax=ax[0], label='c (M)')
    plt.colorbar(ax[1].collections[0], ax=ax[1], label='v (nm/s)')
    fig.savefig('simulation_results.png')
    plt.close()