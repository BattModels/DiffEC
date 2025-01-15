import numpy as np 
import pandas as pd
import json
import os
import subprocess
import random
import sys 


epochs = 30

experimental_currents = [-30.5e-9, -55.2e-9,-102e-9,-238e-9] #Steady state current in nA at 10 mM, 20mM, 40mM, 100 mm of bulk concentration of acetic acid 

lr_initial = 1e-4
optimizer_name = 'sgd'
for i in range(0,50):


    while True:

        logdimkeq_initial = random.random()*4.0 - 7.0 #The range of keq search is
        logdimKf_initial = random.random()*6.0 + 2.0  #The range of kf search is

        if logdimKf_initial - logdimkeq_initial <13.0:
            break
    


    logdimKeq_guess_history = []
    logdimKf_guess_history = []
    loss_history = []
    grad_history_1 = []
    grad_history_2 = []


    json_file_name = f'TrainingLog/DiffCE_trial2_{optimizer_name}_epochs={epochs}_lr={lr_initial:.2E}_{i}.json'

    diffCEDict ={
        "epochs":epochs,
        "logdimkeq_initial":logdimkeq_initial,
        "logdimKf_initial":logdimKf_initial,
        "lr_initial":lr_initial,
        'lr_effective':lr_initial,
        "logdimKeq_guess_history":logdimKeq_guess_history,
        "logdimKf_guess_history":logdimKf_guess_history,
        "loss_history":loss_history,
        "grad_history_1":grad_history_1,
        "grad_history_2":grad_history_2,
        'experimental_currents':experimental_currents,
        "optimizer_name":optimizer_name
    }

    # Convert and write JSON object to file
    with open(json_file_name, "w") as outfile: 
        json.dump(diffCEDict, outfile)

    for i in range(epochs):
        subprocess.run(['sbatch','--wait','submit_cpu.sh',json_file_name])
