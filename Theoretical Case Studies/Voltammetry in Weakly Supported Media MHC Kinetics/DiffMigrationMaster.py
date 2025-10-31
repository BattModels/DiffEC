import numpy as np 
import pandas as pd
import json
import os
import subprocess



epochs=600
logK0_initial = -3
reorg_e_initial=0.5
sigma = 10.0
C_sup =10.0
lr=0.5
logK0_guess_history = []
reorg_e_guess_history = []
loss_history = []
grad_history_1 = []
grad_history_2 = []
optimizer_name = "sgd"
experimental_flux_name = './Data/logK0=5.00E-01 reorg=9.50E-01 sigma=1.00E+01 C_sup=1.00E+01.csv'


diffMigrationDict ={
    "epochs":epochs,
    "reorg_e_initial":reorg_e_initial,
    "logK0_initial":logK0_initial,
    "sigma":sigma,
    "C_sup":C_sup,
    "lr":lr,
    "reorg_e_guess_history":reorg_e_guess_history,
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
