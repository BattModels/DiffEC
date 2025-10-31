import numpy as np 
import json
import jax
import jax.numpy as jnp
from main import mse,simulation,calcSigma
from helper import load_experimental_flux
import optax
import sys
import time
import os
from SimulationHyperParameters import nu,D,c_bulk,rot_freq,omega,scan_rate,theta_i,theta_v,lr,log10K0_initial,log10K0_initial_range,alpha_initial,alpha_initial_range,epochs,exp_dimensionless_file,optimizer_name
import subprocess
from matplotlib import pyplot as plt



SLURM_ARRAY_TASK_ID = int(sys.argv[1])
time.sleep(np.random.uniform(1,20))

def compute_loss(simulation_params,sigma,experimental_flux,theta_i,theta_v):
    alpha_guess = simulation_params[0]
    log10K0_guess = simulation_params[1]
    K0 = jnp.power(10,log10K0_guess)
    theta_simulated, fluxes_predicted = simulation(sigma=sigma,K0=K0,alpha=alpha_guess,theta_i=theta_i,theta_v=theta_v)

    loss = mse(experimental_flux[:1398],fluxes_predicted[:1398])

    return loss 



if __name__ == "__main__":

    if not os.path.exists("./history_folder"):
        os.mkdir('./history_folder')

    if not os.path.exists('./Figures'):
        os.mkdir('./Figures')

    if not os.path.exists('./opt_state_folder'):
        os.mkdir('./opt_state_folder')


    log10K0_initial = np.random.uniform(log10K0_initial-log10K0_initial_range,log10K0_initial+log10K0_initial_range)
    alpha_initial = np.random.uniform(alpha_initial-alpha_initial_range,alpha_initial+alpha_initial_range)

    log10K0_guess_history = []
    alpha_guess_history = []
    loss_history = []


    sigma = calcSigma(omega=omega,nu=nu,D=D,scan_rate=scan_rate)


    diffECDict = {
        "epochs":epochs,
        "log10K0_initial":log10K0_initial,
        "alpha_initial":alpha_initial,
        "sigma":sigma,
        "exp_dimensionless_file":exp_dimensionless_file,
        "log10K0_guess_history":log10K0_guess_history,
        "alpha_guess_history":alpha_guess_history,
        "loss_history":loss_history,
        "lr":lr,
        "optimizer_name":optimizer_name 
    }

    if not os.path.exists("./history_folder"):
        os.mkdir('./history_folder')

    history_dict_name = f"./history_folder/DiffECDict_{SLURM_ARRAY_TASK_ID}.json"

    if not os.path.exists(history_dict_name):
        with open(history_dict_name,'w') as outfile: 
            json.dump(diffECDict,outfile)

    for i in range(epochs):
        if os.path.exists(history_dict_name):
            with open(history_dict_name,'r') as infile:
                diffECDict = json.load(infile)
            if len(diffECDict['log10K0_guess_history']) >= diffECDict['epochs']:
                #If there are enough optimization results, the master program exits. 
                sys.exit()
        subprocess.run(['sbatch','--wait','submit_cpu_worker.sh',f'{SLURM_ARRAY_TASK_ID}'])




        