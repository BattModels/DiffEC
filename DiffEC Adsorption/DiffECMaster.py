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

from SimulationHyperParameters import sigmas,voltammograms_ground_truth,epochs,lr,optimizer_name,logK0_initial,alpha_initial,beta_initial,logK0_ads_initial,alpha_ads_initial,beta_ads_initial,logK_A_ads_initial,logK_B_ads_initial,logK_A_des_initial,logK_B_des_initial,logK0_initial_range,alpha_initial_range,beta_initial_range,logK0_ads_initial_range,alpha_ads_initial_range,beta_ads_initial_range,logK_A_ads_initial_range,logK_B_ads_initial_range,logK_A_des_initial_range,logK_B_des_initial_range

import sys

import time 

time.sleep(np.random.uniform(1,20))

if not os.path.exists("./history_folder"):
    os.mkdir("./history_folder")

if not os.path.exists("./Figures"):
    os.mkdir("./Figures")

if not os.path.exists("./opt_state_folder"):
    os.mkdir("./opt_state_folder")



SLRUM_ARRAY_TASK_ID = int(sys.argv[1])

logK0_initial = np.random.uniform(logK0_initial-logK0_initial_range,logK0_initial+logK0_initial_range)
alpha_initial = np.random.uniform(alpha_initial-alpha_ads_initial_range,alpha_initial+alpha_initial_range)
beta_initial = np.random.uniform(beta_initial-beta_ads_initial_range,beta_initial+beta_ads_initial_range)

logK0_ads_initial = np.random.uniform(logK0_ads_initial-logK0_ads_initial_range,logK0_ads_initial+logK0_ads_initial_range)
alpha_ads_initial = np.random.uniform(alpha_ads_initial-alpha_ads_initial_range,alpha_ads_initial+alpha_ads_initial_range)
beta_ads_initial = np.random.uniform(beta_ads_initial-beta_ads_initial_range,beta_ads_initial+beta_ads_initial_range)

logK_A_ads_initial  = np.random.uniform(logK_A_ads_initial-logK_A_ads_initial_range,logK_A_ads_initial+logK_A_ads_initial_range)
logK_B_ads_initial = np.random.uniform(logK_B_ads_initial-logK_B_ads_initial_range,logK_B_ads_initial+logK_B_ads_initial_range)

logK_A_des_initial = np.random.uniform(logK_A_des_initial-logK_A_des_initial_range,logK_A_des_initial+logK_A_des_initial_range)
logK_B_des_initial = np.random.uniform(logK_B_des_initial-logK_B_des_initial_range,logK_B_des_initial+logK_B_des_initial_range)

logK0_guess_history = []
alpha_guess_history = []
beta_guess_history = []

logK0_ads_guess_history = []
alpha_ads_guess_history = []
beta_ads_guess_history  = []

logK_A_ads_guess_history = []
logK_B_ads_guess_history = []
logK_A_des_guess_history = []
logK_B_des_guess_history = []

loss_history = []



    


diffECDict = {
    "epochs":epochs,
    "optimizer_name":optimizer_name,
    "lr":lr,
    "loss_history":loss_history,
    "logK0_initial":logK0_initial,
    "alpha_initial":alpha_initial,
    "beta_initial":beta_initial,
    "logK0_ads_initial":logK0_ads_initial,
    "alpha_ads_initial":alpha_ads_initial,
    "beta_ads_initial":beta_ads_initial,
    "logK_A_ads_initial":logK_A_ads_initial,
    "logK_B_ads_initial":logK_B_ads_initial,
    "logK_A_des_initial":logK_A_des_initial,
    "logK_B_des_initial":logK_B_des_initial,
    "logK0_guess_history":logK0_guess_history,
    "alpha_guess_history":alpha_guess_history,
    "beta_guess_history":beta_guess_history,
    "logK0_ads_guess_history": logK0_ads_guess_history,
    "alpha_ads_guess_history":alpha_ads_guess_history,
    "beta_ads_guess_history":beta_ads_guess_history,
    "logK_A_ads_guess_history":logK_A_ads_guess_history,
    "logK_B_ads_guess_history":logK_B_ads_guess_history,
    "logK_A_des_guess_history":logK_A_des_guess_history,
    "logK_B_des_guess_history":logK_B_des_guess_history,
    "loss_history":loss_history
}

if not os.path.exists("./history_folder"):
    os.mkdir("./history_folder")
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
        subprocess.run(['sbatch','--wait','submit_cpu_worker.sh',f'{SLRUM_ARRAY_TASK_ID}'])