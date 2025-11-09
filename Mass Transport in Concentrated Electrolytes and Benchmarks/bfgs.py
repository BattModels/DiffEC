import os
os.environ["JAX_PLATFORMS"] = "cpu"
import time

import jax
import jax.numpy as jnp
from jax import config, jit, grad
import jaxopt

config.update("jax_enable_x64", True)

from solver import loss_func, simulate
from utils import load_c_data, interp_c_data, get_pos_indices

def sync(tree):
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )

# Forward-mode (JVP) value+grad wrapper (fast for 2 params)
def make_value_and_grad_forward(loss_scalar_fun):
    @jit
    def f_val_and_grad(params):
        # flatten params to [p0, p1]
        p = jnp.array([params['p0'], params['p1']])
        def f_flat(pvec):
            return loss_scalar_fun({'p0': pvec[0], 'p1': pvec[1]})

        val = f_flat(p)
        e1, e2 = jnp.array([1., 0.]), jnp.array([0., 1.])
        _, g1 = jax.jvp(f_flat, (p,), (e1,))
        _, g2 = jax.jvp(f_flat, (p,), (e2,))
        grad_dict = {'p0': g1, 'p1': g2}
        return val, grad_dict
    return f_val_and_grad

def safe(x):
    return jnp.nan_to_num(x, nan=1e3, posinf=1e3, neginf=1e3)

if __name__ == "__main__":
    # Get parameters from command line
    import argparse
    parser = argparse.ArgumentParser(description='Get parameters for the simulation.')
    parser.add_argument('--p0', type=float, default=-3.0, help='Coefficient for constant term')
    parser.add_argument('--p1', type=float, default=-3.0, help='Coefficient for linear term')
    parser.add_argument('--lbd', type=float, default=5e-3, help='Regularization parameter')
    parser.add_argument('--save', action='store_true', help='Save results to disk')
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

    # Define loss function with regularization
    def loss_with_reg(params):
        p0 = params['p0']
        p1 = params['p1']

        loss_physics = loss_func(
            params['p0'], params['p1'], 
            c_data_interp[:num_train], time_indices[:num_train], space_indices, 
            c_avg, N, L, dt, t_in_minutes
        )
        loss_reg = args.lbd * (p0**2 + p1**2)
        return safe(loss_physics + loss_reg)

    loss_vg_fwd = make_value_and_grad_forward(loss_with_reg)

    # Record history
    loss_history, p0_history, p1_history = [], [], []
    # Append initial values
    p0_history.append(float(params['p0']))
    p1_history.append(float(params['p1'])) 

    def cb(xk):
        p0_history.append(float(xk['p0']))
        p1_history.append(float(xk['p1']))

    # Define optimizer
    method = "BFGS"
    solver = jaxopt.ScipyMinimize(
        fun=loss_vg_fwd, 
        value_and_grad=True, 
        method=method, 
        callback=cb,
        options={"gtol": 1e-6, "disp": True},
    )
    t0 = time.perf_counter()
    best_p, info = solver.run(params)
    t1 = time.perf_counter()

    # Check final gradient norm
    final_grad = grad(loss_with_reg)(best_p)
    error = jnp.linalg.norm(jnp.array([final_grad['p0'], final_grad['p1']]))

    print(f"Method = {method}")
    print(f"Initial guess = ({params['p0']}, {params['p1']})")
    print(f"Optimization took {t1 - t0} seconds")
    print(f"Best (p0,p1): ({best_p['p0']}, {best_p['p1']})")
    print(f"Final gradient norm = {error}")

    if args.save == True:
        # If the directory does not exist, create it
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # Reevaluate the loss history with the learned parameters
        for i in range(len(p0_history)):
            loss_history.append(float(loss_with_reg({'p0': p0_history[i], 'p1': p1_history[i]})))

        # Save the loss history, p0 history, and p1 history
        jnp.save('results/loss_history.npy', jnp.array(loss_history))
        jnp.save('results/p0_history.npy', jnp.array(p0_history))
        jnp.save('results/p1_history.npy', jnp.array(p1_history))

        # Save the simulation results for selected positions and times with the learned parameters
        _, _, c_arr, v0_arr_c = simulate(best_p['p0'], best_p['p1'], c_avg, N, L, dt, 1100)
        pos_indices = get_pos_indices(L / N)
        jnp.save('results/c_sim.npy', c_arr[time_indices, :])
        jnp.save('results/v0_sim.npy', v0_arr_c[:, pos_indices])
