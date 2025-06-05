import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import config, grad, jit, vmap, lax
import jaxopt
from functools import partial

config.update("jax_enable_x64", True) 

from utils import generate_mesh, csv_to_numpy, load_c_data, interp_c_data, get_pos_indices

# ---------- Global Variables ----------
Faraday = 96485.0                      # Faraday constant (C/mol)
M0 = 44.05                             # Molar mass of PEO (g/mol)
M = 6.94 + 280.15                      # Molar mass of salt (g/mol)

# Read experimental data
current_density_data = csv_to_numpy('data/current_density.csv')
D_tp0_relation_data = csv_to_numpy('data/D_tp0_relation.csv')

# ---------- Finite Volume Solver ----------
@jit
def step(p0, p1, c, v0, i, dt, dx, c_avg):
    """
    Compute the next time step using the finite volume method.
    Args:
        p0 (float): Coefficient for constant term.
        p1 (float): Coefficient for linear term.
        c (ndarray): Current concentration, stored at cell centroids.
        v0 (ndarray): Current solvent velocity in the moving electrode reference frame, stored at cell interfaces, including boundary.
        i (float): Ionic current.
        dt (float): Time step size.
        dx (float): Cell size.
        c_avg (float): Average concentration at initial time.
    Returns:
        c_next (ndarray): Concentration at the next time step.
        v0_next (ndarray): Solvent velocity at the next time step.
    """
    # Compute the concentration fluxes at the cell interfaces as well as the boundary fluxes
    F = flux(p0, p1, c, v0, i, dx, c_avg)

    # Update the concentration using the first-order Euler method
    c_next = c - dt / dx * (F[1:] - F[:-1]) * 1e3
    
    # Update the solvent velocity for next time step
    F = flux(p0, p1, c_next, jnp.zeros_like(v0), i, dx, c_avg)
    v0_next = update_solvent_vel(F, c_next)

    return c_next, v0_next

@jit
def flux(p0, p1, c, v0, i, dx, c_avg):
    cf = (c[:-1] + c[1:]) / 2                 # Concentration at the cell interfaces (mol/L)
    dcdx = (c[1:] - c[:-1]) / dx * 1e-3       # Gradient of concentration at the cell interfaces (mol/cm^4)
    factorf = factor(cf)                      # unitless 

    # Tansference number at the cell interfaces (unitless)
    tp0f = tp0(cf, p0, p1, c_avg)
    # Diffusion coefficient at the cell interfaces (cm^2/s)
    Df = D(cf, tp0f, factorf)

    # Compute the fluxes at the interior cell interfaces
    F_int = -Df * factorf * dcdx + i / Faraday * tp0f + cf * v0[1:-1] * 1e-3

    # Compute the fluxes at the boundary cell interfaces
    F_left = i / Faraday
    F_right = i / Faraday
    F = jnp.concatenate((jnp.array([F_left]), F_int, jnp.array([F_right])))

    return F

@jit
def update_solvent_vel(F, c):
    V_bar_value = V_bar(c)                   # Partial molar volume at the cell interfaces

    cur = 0.0                                # Record value at current cell interface
    v0_next = jnp.zeros_like(F)
    for j in range(1, len(F)-1):
        v0_next = v0_next.at[j].set(cur - V_bar_value[j-1] * (F[j] - F[j-1]))
        cur = v0_next[j]

    return v0_next

# ---------- Physics Functions ----------
@jit
def tp0(c, p0, p1, c_avg):
    """
    Polynomial function for transference number.
    Args:
        c (ndarray): Concentration.
        p0 (float): Coefficient for constant term.
        p1 (float): Coefficient for linear term.
        c_avg (float): Average concentration.
    Returns:
        ndarray: Transference number.
    """
    return p0 + p1 * (c - c_avg) / c_avg

@jit
def rho(c):
    return 1.123276 + 0.106822 * c + 0.007606 * c**2                    # g/cm^3

@jit
def drhodc(c):
    return vmap(grad(rho))(c)

@jit
def V_bar(c):
    drhodc_value = drhodc(c)
    return (M - drhodc_value * 1e3) / (rho(c) - c * drhodc_value)       # cm^3/mol

@jit
def factor(c):
    rho_value = rho(c)
    c0 = 1 / M0 * (rho_value * 1e3 - M * c)                             # Molarity: mol/L
    V0_bar = M0 / (rho_value - c * drhodc(c))                           # Partial Molar volume of solvent
    return 1 / (c0 * V0_bar) * 1e3

@jit
def D(c, tp0, factor):
    """
    Compute the diffusion coefficient.
    Args:
        c (ndarray): Concentration.
        tp0 (float): Transference number.
        factor (float): Factor for the diffusion.
    Returns:
        ndarray: Diffusion coefficient.
    """
    xp = D_tp0_relation_data[:, 0]
    fp = D_tp0_relation_data[:, 1]
    relation_coef = jnp.interp(c, xp, fp)

    return (1 - tp0) * relation_coef / factor

@jit
def i_func(t):
    """
    Interpolate the current density.
    Args:
        t (ndarray): Time in seconds.
    Returns:
        ndarray: Current density (A/cm^2).
    """
    xp = current_density_data[:, 0]
    fp = current_density_data[:, 1]
    return jnp.interp(t / 3600, xp, fp) * 1e-3

# ---------- Simulator and Loss Function ----------
@partial(jit, static_argnames=('N', 'L', 'dt', 't_in_minutes'))
def simulate(p0, p1, c_avg, N, L, dt, t_in_minutes):
    # Generate mesh
    dx = L / N
    x = generate_mesh(N, L)

    # Initial conditions
    c = c_avg * jnp.ones(N)
    cf = c_avg * jnp.ones(N+1)
    v0 = V_bar(cf) * (1 - tp0(cf, p0, p1, c_avg)) * i_func(0) / Faraday       # cm/s

    # Time-stepping loop
    t_end = t_in_minutes * 60
    num_steps = int(t_end / dt)
    cur_time = 0.0        # seconds

    def step_wrapper(carry, _):
        c, v0, cur_time = carry
        i = i_func(cur_time)
        c, v0 = step(p0, p1, c, v0, i, dt, dx, c_avg)
        cur_time += dt
        return (c, v0, cur_time), (c, v0)
    
    # Initial carry: (c, v0, cur_time)
    carry_init = (c, v0, cur_time)

    (c_final, v0_final, _), (c_history, v0_history) = lax.scan(
        step_wrapper,
        carry_init,
        xs=None,  # No sequence to iterate over, just loop num_steps times
        length=num_steps
    )

    # Prepend the initial conditions
    c_arr = jnp.concatenate([jnp.expand_dims(c, axis=0), c_history], axis=0)
    v0_arr = jnp.concatenate([jnp.expand_dims(v0, axis=0), v0_history], axis=0)

    # Interpolate velocity to cell centers
    v0_arr_c = (v0_arr[:, :-1] + v0_arr[:, 1:]) / 2  

    X, T = jnp.meshgrid(10 * x, jnp.arange(num_steps+1) * dt / 3600)  # Length in mm, Time in hours

    return X, T, c_arr, v0_arr_c

def loss_func(p0, p1, c_data, time_indices, space_indices, c_avg, N, L, dt, t_in_minutes):
    _, _, c_arr, _ = simulate(p0, p1, c_avg, N, L, dt, t_in_minutes)
    loss = jnp.sqrt(jnp.sum((c_arr[jnp.ix_(time_indices, space_indices)] - c_data)**2) / len(time_indices) / len(space_indices)) / c_avg
    
    return loss

if __name__ == "__main__":
    # Get parameters from command line
    import argparse
    parser = argparse.ArgumentParser(description='Get parameters for the simulation.')
    parser.add_argument('--p0', type=float, default=0.0, help='Coefficient for constant term')
    parser.add_argument('--p1', type=float, default=0.0, help='Coefficient for linear term')
    parser.add_argument('--lbd', type=float, default=5e-3, help='Regularization parameter')
    args = parser.parse_args()

    # Initialize polynomial coefficients that need to be learnt
    params = {'p0': args.p0, 'p1': args.p1}

    # Parameters
    c_avg = 1.87
    N = 50
    L = 0.3                 # Length in cm
    dt = 0.1                # Time step in seconds
    
    # Load experimental data
    c_data = load_c_data()

    # Interpolate experimental data
    c_data_interp, x_interp, time_indices, space_indices = interp_c_data(c_data, N, L)

    num_train = 9
    t_in_minutes = time_indices[num_train-1].item() / 600
    
    # Define loss function with regularization and optimizer
    def loss_with_reg(params):
        p0 = params['p0']
        p1 = params['p1']

        loss_physics = loss_func(params['p0'], params['p1'], c_data_interp[:num_train], time_indices[:num_train], space_indices, c_avg, N, L, dt, t_in_minutes)
        loss_reg = args.lbd * (p0**2 + p1**2)
        return loss_physics + loss_reg
    
    solver = jaxopt.BFGS(fun=loss_with_reg, tol=1e-8)
    state = solver.init_state(params)
    print(f"Regularization lambda = {args.lbd}")

    # Begin optimization
    loss_history = []
    p0_history = []
    p1_history = []
    for i in range(50):  # max iterations
        print(f"Iteration {i}, params = {params}")
        print(f"state = {state}")

        loss_history.append(state.value)
        p0_history.append(params['p0'])
        p1_history.append(params['p1'])

        if state.error < solver.tol:
            print(f"Converged at iteration {i}, error = {state.error}")
            break

        # Update parameters
        params, state = solver.update(params, state)
    
    # Final simulation with the learned parameters
    _, _, c_arr, v0_arr_c = simulate(params['p0'], params['p1'], c_avg, N, L, dt, 1100)

    # If the directory does not exist, create it
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save the loss history, p0 history, and p1 history
    jnp.save('results/loss_history.npy', jnp.array(loss_history))
    jnp.save('results/p0_history.npy', jnp.array(p0_history))
    jnp.save('results/p1_history.npy', jnp.array(p1_history))

    # Save the simulation results for selected positions and times
    pos_indices = get_pos_indices(L / N)
    jnp.save('results/c_sim.npy', c_arr[time_indices, :])
    jnp.save('results/v0_sim.npy', v0_arr_c[:, pos_indices])

    