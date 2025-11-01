import numpy as np 
import jax.numpy as jnp
import jax.scipy.linalg as linag
import os
import jax
jax.config.update('jax_enable_x64',True)

import json
import optax
import pickle
import pandas as pd

from helper import expParameters
import subprocess


from SimulationHyperParameters import scan_rates, exp_dimensionless_files, epochs, logK0_initial,logK0_initial_range,alpha_initial,alpha_initial_range,beta_initial,beta_initial_range,dA_initial,dA_initial_range, lr, optimizer_name

import sys

import time 

if not os.path.exists("./history_folder"):
    os.mkdir("./history_folder")

if not os.path.exists("./opt_state_folder"):
    os.mkdir("./opt_state_folder")

if not os.path.exists("./Figures"):
    os.mkdir("")



time.sleep(np.random.uniform(1,20))

SLRUM_ARRAY_TASK_ID = 0#int(sys.argv[1])

logK0_initial = np.random.uniform(logK0_initial-logK0_initial_range,logK0_initial+logK0_initial_range)
alpha_initial = np.random.uniform(alpha_initial-alpha_initial_range,alpha_initial+alpha_initial_range)
beta_initial = np.random.uniform(beta_initial-beta_initial_range,beta_initial+beta_initial_range)
dA_initial = np.random.uniform(dA_initial-dA_initial_range,dA_initial+dA_initial_range)

sigmas = []


for exp_dimensionless_file in exp_dimensionless_files:
    sigma,theta_i,theta_v,dA = expParameters(exp_dimensionless_file)
    sigmas.append(sigma)

logK0_guess_history = []
alpha_guess_history = []
beta_guess_history = []
dA_guess_history = []
loss_history = []

diffECDict = {
    "epochs":epochs,
    "logK0_initial":logK0_initial,
    "alpha_initial":alpha_initial,
    "beta_initial":beta_initial,
    "dA_initial":dA_initial,
    "sigmas":sigmas,
    "exp_dimensionless_files":exp_dimensionless_files,
    "logK0_guess_history":logK0_guess_history,
    "alpha_guess_history":alpha_guess_history,
    "beta_guess_history":beta_guess_history,
    "dA_guess_history":dA_guess_history,
    "lr":lr,
    "optimizer_name":optimizer_name,
    "loss_history":loss_history,
}

history_dict_name = f"./history_folder/DiffECDict_{SLRUM_ARRAY_TASK_ID}.json"

if not os.path.exists(history_dict_name):
    with open(history_dict_name,'w') as outfile: 
        json.dump(diffECDict,outfile)



if __name__ == "__main__":
    for i in range(epochs):
        if os.path.exists(history_dict_name):
            with open(history_dict_name,'r') as infile:
                diffECDict = json.load(infile)
            if len(diffECDict['logK0_guess_history']) >= diffECDict['epochs']:
                #If there are enough optimization results, the master program exits. 
                sys.exit()
        #This is subject a slurm script to run the diffEC worker 
        subprocess.run(['sbatch','--wait','submit_cpu_worker.sh',f'{SLRUM_ARRAY_TASK_ID}'])





    


    

