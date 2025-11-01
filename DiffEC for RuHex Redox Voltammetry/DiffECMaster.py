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
import subprocess
from SimulationHyperParameters import scan_rates,sigmas, exp_dimensionless_files, epochs,theta_corr_initial,theta_corr_initial_range,dA_initial,dA_initial_range, lr, optimizer_name
    
import sys

import time 

time.sleep(np.random.uniform(1,20))

SLRUM_ARRAY_TASK_ID = int(sys.argv[1])

theta_corr_initial = np.random.uniform(theta_corr_initial-theta_corr_initial_range,theta_corr_initial+theta_corr_initial_range)
dA_initial = np.random.uniform(dA_initial-dA_initial_range,dA_initial+dA_initial_range)


theta_corr_history = []
dA_guess_history = []
loss_history = []


diffECDict = {
    "epochs":epochs,
    "theta_corr_initial":theta_corr_initial,
    "dA_initial":dA_initial,
    "theta_corr_history":theta_corr_history,
    "dA_guess_history":dA_guess_history,
    "lr":lr,
    "optimizer_name":optimizer_name,
    "loss_history":loss_history,
}



history_dict_name = f"./history_folder/DiffECDict_{SLRUM_ARRAY_TASK_ID}.json"

if not os.path.exists("./history_folder"):
    os.mkdir("./history_folder")

if not os.path.exists("./opt_state_folder"):
    os.mkdir("./opt_state_folder")

if not os.path.exists("./Figures"):
    os.mkdir("./Figures")

if not os.path.exists(history_dict_name):
    with open(history_dict_name,'w') as outfile: 
        json.dump(diffECDict,outfile)



if __name__ == "__main__":
    for i in range(epochs):
        if os.path.exists(history_dict_name):
            with open(history_dict_name,'r') as infile:
                diffECDict = json.load(infile)
            if len(diffECDict['theta_corr_history']) >= diffECDict['epochs']:
                #If there are enough optimization results, the master program exits. 
                sys.exit()
        subprocess.run(['sbatch','--wait','submit_cpu_worker.sh',f'{SLRUM_ARRAY_TASK_ID}'])



    


    

