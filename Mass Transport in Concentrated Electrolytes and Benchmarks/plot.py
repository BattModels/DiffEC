import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils import generate_mesh, load_c_data, load_v0_data
from solver import i_func, tp0, D, factor

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 12 }
plt.rc('font', **font)

# ---------- Global Variables ----------
Faraday = 96485.0                      # Faraday constant (C/mol)
c_avg = 1.87                           # Average concentration (mol/L)
MLi = 6.94                             # Molar mass of Li (g/mol)
rho_Li = 0.534                         # g/cm^3
L = 3                                  # Length of the electrolyte thickness (mm)
N = 50                                 # Number of cells
dt = 0.1                               # Time step size (s)
t_end_in_minutes = 1100

num_steps = int(t_end_in_minutes * 60 / dt)
x = generate_mesh(N, L)
t = jnp.arange(num_steps+1) * dt / 3600  # Convert time to hours

# ---------- Load Data ----------
directory = 'results/'
p0_history = jnp.load(directory + 'p0_history.npy')
p1_history = jnp.load(directory + 'p1_history.npy')
loss_history = jnp.load(directory + 'loss_history.npy')

c_sim = jnp.load(directory + 'c_sim.npy')
v0_sim = jnp.load(directory + 'v0_sim.npy')

params = {
    'p0': p0_history[-1],
    'p1': p1_history[-1]
}

# ---------- Plot Results ----------
# Plot parameter history and loss history
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(p0_history, alpha=0.7, label=r'$p_0$',lw=2.5)
ax.plot(p1_history, alpha=0.7, label=r'$p_1$',lw=2.5)
ax.set_xlabel('Iteration', fontsize='large',fontweight='bold')
ax.set_ylabel(r'$p_0$ or $p_1$', fontsize='large',fontweight='bold')
ax.legend()
twin = ax.twinx()
twin.plot(loss_history, color='k', alpha=0.5, ls='--', label='Loss',lw=3)
twin.set_ylabel('Loss', fontsize='large', fontweight='bold')
twin.legend()
fig.savefig('parameter_and_loss_history.png', bbox_inches='tight',dpi=250)
plt.close()

# Plot concentration comparison
# Load experimental data
c_data = load_c_data()

fig, ax = plt.subplots(3, 3, figsize=(9, 9))
for i in range(3):
    for j in range(3):
        ax[i, j].scatter(c_data[i*3+j][:, 0], c_data[i*3+j][:, 1], label='Experimental Data')
        ax[i, j].plot(x, c_sim[i*3+j], label='Simulation Result')

        ax[i, j].set_xticks(jnp.linspace(0, 3, 4))
        ax[i, j].set_yticks(jnp.array([0.6, 1.2, 1.9, 2.6, 3.2]))
        
        # Only set ylabel on the leftmost column
        if j == 0:
            ax[i, j].set_ylabel(r"$c$ (M)", fontsize='large',fontweight='bold')
        # Only set xlabel on the bottom row
        if i == 2:
            ax[i, j].set_xlabel(r"$x$ (mm)", fontsize='large',fontweight='bold')
plt.tight_layout()
fig.savefig('concentration_comparison.png', bbox_inches='tight', dpi=250)

# Plot velocity comparison
# Read experimental data
vel_data = load_v0_data()

# Calculate interface velocity
v_interface = -i_func(t * 3600) / Faraday * MLi / rho_Li

num_steps = len(t)
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
for i in range(2):
    for j in range(3):
        if i*3+j >= len(vel_data):  
            break
        ax[i, j].scatter(vel_data[i*3+j][:, 0], vel_data[i*3+j][:, 1], label='Experimental Data')
        ax[i, j].plot(t, (v0_sim[:, i*3+j] + v_interface) * 1e7, label='Simulation Result')
        ax[i, j].set_xticks(jnp.array([0, 5, 10, 15, 20]))
        ax[i, j].set_yticks(jnp.array([0, 5, 10, 15]))
        
        # Only set ylabel on the leftmost column
        if j == 0:
            ax[i, j].set_ylabel(r"$v^\prime$ (nm/s)")
        # Only set xlabel on the bottom row and the last column on the top row
        if i == 1 or (i == 0 and j == 2):
            ax[i, j].set_xlabel(r"$t$ (h)")

# Hide the unused subplot in second row, last column
fig.delaxes(ax[1, 2])  # Deletes the last axis
fig.savefig('velocity_comparison.png', bbox_inches='tight', dpi=250)

# Plot transference number
fig, ax = plt.subplots()
c = jnp.linspace(0.6, 3, 100)
ax.scatter(jnp.array([0.8584, 1.1896, 1.5823, 1.8564, 2.10376, 2.3733, 2.5724, 2.7570]), 
           jnp.array([0.4058, 0.3376, 0.4330, 0.2025, 0.0837, -0.0740, -0.3760, 0.1052]), 
           label='Experiment')
ax.plot(c, tp0(c, params['p0'], params['p1'], c_avg), label='Estimation')
ax.set_xticks(jnp.array([0.6, 1.2, 1.9, 2.6, 3.2]))
ax.set_yticks(jnp.array([-1.0, -0.5, 0, 0.5, 1.0]))
ax.set_xlim(0.5, 3.3)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel(r"$c$ (M)")
ax.set_ylabel(r"$t^0_+$")
ax.legend()
fig.savefig('tp0.png', bbox_inches='tight', dpi=250)
plt.close()

# Plot diffusion coefficient
fig, ax = plt.subplots()
c = jnp.linspace(0.6, 2.97, 100)
ax.scatter(jnp.array([0.8612, 1.1874, 1.5827, 1.8578, 2.0984, 2.3691, 2.5707, 2.7511]), 
           jnp.array([1.0038e-7, 1.3062e-7, 1.1053e-7, 8.4410e-8, 7.0545e-8, 5.8875e-8, 9.4628e-8, 9.0685e-8]), 
           label='Experiment')
ax.plot(c, D(c, tp0(c, params['p0'], params['p1'], c_avg), factor(c)), label='Estimation')
ax.set_xticks(jnp.array([0.6, 1.2, 1.9, 2.6, 3.2]))
ax.set_yticks(jnp.array([0, 1, 2, 3, 4]) * 1e-7)
ax.set_xlim(0.5, 3.3)
ax.set_ylim(-0.1e-7, 4.1e-7)
ax.set_xlabel(r"$c$ (M)")
ax.set_ylabel(r"$D$ (cm$^2$/s)")
ax.legend()
fig.savefig('D.png', bbox_inches='tight', dpi=250)
plt.close()