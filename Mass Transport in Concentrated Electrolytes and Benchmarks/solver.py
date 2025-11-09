import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import config, jit, lax
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

I_xp = jnp.asarray(current_density_data[:, 0])
I_fp = jnp.asarray(current_density_data[:, 1])
D_xp = jnp.asarray(D_tp0_relation_data[:, 0])
D_fp = jnp.asarray(D_tp0_relation_data[:, 1])

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
    V_bar_value = V_bar(c)                   # Partial molar volume at the cell centroids

    # increments for faces j=1..N-1 (uses F[j]-F[j-1])
    inc = -V_bar_value[:-1] * (F[1:-1] - F[:-2])
    cum = jnp.cumsum(inc)
    v0_next = jnp.zeros_like(F)
    v0_next = v0_next.at[1:-1].set(cum)

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
    return 0.106822 + 2 * 0.007606 * c

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

    relation_coef = jnp.interp(c, D_xp, D_fp)

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

    return jnp.interp(t / 3600, I_xp, I_fp) * 1e-3

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
