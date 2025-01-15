import numpy as np 
import pandas as pd
import json
import os
import subprocess



epochs=500
alpha_initial=0.5
logK0_initial = -2
sigma = 10.0
C_sup =5.0
lr=0.5
alpha_guess_history = []
logK0_guess_history = []
loss_history = []
grad_history_1 = []
grad_history_2 = []
optimizer_name = "sgd"
experimental_flux_name = './Data/sigma=1.00E+01 logK0=5.00E-01 alpha=4.50E-01 C_sup=5.csv'


diffMigrationDict ={
    "epochs":epochs,
    "alpha_initial":alpha_initial,
    "logK0_initial":logK0_initial,
    "sigma":sigma,
    "C_sup":C_sup,
    "lr":lr,
    "alpha_guess_history":alpha_guess_history,
    "logK0_guess_history":logK0_guess_history,
    "loss_history":loss_history,
    "grad_history_1":grad_history_1,
    "grad_history_2":grad_history_2,
    "optimizer_name":optimizer_name,
    "experimental_flux_name":experimental_flux_name
}

# Convert and write JSON object to file
with open("DiffMigrationDict.json", "w") as outfile: 
    json.dump(diffMigrationDict, outfile)

for i in range(epochs):
    subprocess.run(['sbatch','--wait','submit_cpu.sh'])
