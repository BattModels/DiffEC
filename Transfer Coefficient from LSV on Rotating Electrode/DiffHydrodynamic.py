import numpy as np 
import json
import jax
import jax.numpy as jnp
from main import mse,simulation,calcSigma
from helper import load_experimental_flux
import optax
import sys

SLURM_ARRAY_TASK_ID = 0# int(sys.argv[1])

def compute_loss(simulation_params,sigma,experimental_flux,theta_i,theta_v):
    alpha_guess = simulation_params[0]
    beta_guess = simulation_params[1]
    log10K0_guess = simulation_params[2]
    K0 = jnp.power(10,log10K0_guess)
    fluxes_predicted = simulation(sigma=sigma,K0=K0,alpha=alpha_guess,beta=beta_guess,theta_i=theta_i,theta_v=theta_v)

    loss = mse(experimental_flux[:1398],fluxes_predicted[:1398])

    return loss 



if __name__ == "__main__":
    for index,rel_noise_level in enumerate([0.0]):
        if index == SLURM_ARRAY_TASK_ID:
            nu = 1e-6 #Kinematic viscosity of water m^2/s 
            D = 9.311e-9 #Diffusion coefficients m^2/s 
            c_bulk = 1e3 #Bulk concentration of H+ in solutionmol/m^3 
            rot_freq  = 2500/60 # Rotatingal frequency RMP to Hz
            omega = rot_freq*np.pi*2 #Rad/second
            scan_rate = 2e-3 #V/s 
            sigma = calcSigma(omega=omega,nu=nu,D=D,scan_rate=scan_rate)
            theta_i = 0.755379149453578 #Start potential of LSV in Volt 
            theta_v = -2.188620850546423 #Stop potential of LSV in Volt

            lr = 1e-2

            alpha_initial_guess = 1.0
            beta_initial_guess = 1.0
            log10K0_initial_guess = -3



            experimental_flux = load_experimental_flux(r'KoperExperimentDimensionless.csv',rel_noise_level=rel_noise_level)

            epochs = 500
            
            alpha_guess = alpha_initial_guess
            beta_guess = beta_initial_guess 
            log10K0_guess = log10K0_initial_guess

            alpha_guess_history = []
            beta_guess_history = []
            log10K0_guess_history = []
            loss_history = []

            grad_history = []

            params = jnp.array([alpha_guess,beta_guess,log10K0_guess])


            value_and_grd = jax.value_and_grad(compute_loss,argnums=0)

            optimizer = optax.adam(learning_rate=lr)
            opt_state = optimizer.init(params=params)


            for epoch in range(epochs):
                loss, grad = value_and_grd(params,sigma,experimental_flux,theta_i=theta_i,theta_v=theta_v)
                updates,opt_state = optimizer.update(grad,opt_state,params)

                params = optax.apply_updates(params,updates)


                alpha_guess_history.append(float(params[0]))
                beta_guess_history.append(float(params[1]))
                log10K0_guess_history.append(float(params[2]))
                loss_history.append(float(loss))

                diff_dict = {'rel_noise_level':rel_noise_level,'alpha_guess_history':alpha_guess_history,'beta_guess_history':beta_guess_history,'log10K0_guess_history':log10K0_guess_history,'loss_history':loss_history,'grad_history':grad_history}

                with open(f"./histories/rel_noise_levl={rel_noise_level:.2E}.json",'w') as f:
                    json.dump(diff_dict,f)

        