import numpy as np 
import jax.numpy as jnp
import os
import json
import pandas as pd
from matplotlib import pyplot as plt
from DiffECHyperParameters import optimizer_name,lr,epochs
from KineticModels import MH_current,MHC_current

linewidth = 4
fontsize = 10

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs

system_names = [r"10% FEC",r'EC DEC',r'DEC',r'PC'][::-1]
system_labels = [r'EC:DEC w. 10% FEC',r'EC:DC',r'DEC',r'PC'][::-1]




for ii, (system_name,system_label) in enumerate(zip(system_names,system_labels)):
    project_history_folder = f'./history_folder_{system_name}'

    losses_all_MH = []
    j0_all_MH = []
    reogr_e_all_MH = []

    losses_all_MHC = []
    j0_all_MHC = []
    reogr_e_all_MHC = []

    losses_all_MHC_approx = []
    j0_all_MHC_approx = []
    reogr_e_all_MHC_approx = []


    for SLRUM_ARRAY_TASK_ID in range(1):
        with open(f"{project_history_folder}/DiffECDict_MH_{SLRUM_ARRAY_TASK_ID}.json") as f:
            diffECDict = json.load(f)
        
        j0_initial_guess = diffECDict['j0_initial_guess']
        reorg_e_initial_guess = diffECDict['reorg_e_initial_guess']
        j0_history = diffECDict['j0_history']
        reorg_e_history = diffECDict['reorg_e_history']
        rmse_loss_history = diffECDict['rmse_loss_history']


        j0_all_MH.append(j0_history)
        reogr_e_all_MH.append(reorg_e_history)
        losses_all_MH.append(rmse_loss_history)



        fig,axs = plt.subplots(figsize=(8,13.5),nrows=3)

        ax = axs[0]
        ax.plot(rmse_loss_history)
        ax.set_yscale('log')
        ax.set_xlabel('Epochs',fontsize='large',fontweight='bold')
        ax.set_ylabel(r'RMSE, $mA/cm^2$',fontsize='large',fontweight='bold')

        ax = axs[1]
        ax.plot(j0_history)
        ax.set_xlabel('Epochs',fontsize='large',fontweight='bold')
        ax.set_ylabel(r'$j_0$, $mA/cm^2$',fontsize='large',fontweight='bold')

        ax = axs[2]
        ax.plot(reorg_e_history)
        ax.set_xlabel('Epochs',fontsize='large',fontweight='bold')
        ax.set_ylabel('reorg_e, eV',fontsize='large',fontweight='bold')
        


        #ax.legend()

        fig.savefig(f"{project_history_folder}/DiffECDict_MH_{SLRUM_ARRAY_TASK_ID}.png",dpi=250,bbox_inches='tight') 


        with open(f"{project_history_folder}/DiffECDict_MHC_{SLRUM_ARRAY_TASK_ID}.json") as f:
            diffECDict = json.load(f)
        
        j0_initial_guess = diffECDict['j0_initial_guess']
        reorg_e_initial_guess = diffECDict['reorg_e_initial_guess']
        j0_history = diffECDict['j0_history']
        reorg_e_history = diffECDict['reorg_e_history']
        rmse_loss_history = diffECDict['rmse_loss_history']


        j0_all_MHC.append(j0_history)
        reogr_e_all_MHC.append(reorg_e_history)
        losses_all_MHC.append(rmse_loss_history)



        fig,axs = plt.subplots(figsize=(8,13.5),nrows=3)

        ax = axs[0]
        ax.plot(rmse_loss_history)
        ax.set_yscale('log')
        ax.set_xlabel('Epochs',fontsize='large',fontweight='bold')
        ax.set_ylabel(r'RMSE, $mA/cm^2$',fontsize='large',fontweight='bold')

        ax = axs[1]
        ax.plot(j0_history)
        ax.set_xlabel('Epochs',fontsize='large',fontweight='bold')
        ax.set_ylabel(r'$j_0$, $mA/cm^2$',fontsize='large',fontweight='bold')

        ax = axs[2]
        ax.plot(reorg_e_history)
        ax.set_xlabel('Epochs',fontsize='large',fontweight='bold')
        ax.set_ylabel(r'$\lambda$, eV',fontsize='large',fontweight='bold')
        


        #ax.legend()

        fig.savefig(f"{project_history_folder}/DiffECDict_MHC_{SLRUM_ARRAY_TASK_ID}.png",dpi=250,bbox_inches='tight') 



        with open(f"{project_history_folder}/DiffECDict_MH_approx_{SLRUM_ARRAY_TASK_ID}.json") as f:
            diffECDict = json.load(f)
        
        j0_initial_guess = diffECDict['j0_initial_guess']
        reorg_e_initial_guess = diffECDict['reorg_e_initial_guess']
        j0_history = diffECDict['j0_history']
        reorg_e_history = diffECDict['reorg_e_history']
        rmse_loss_history = diffECDict['rmse_loss_history']


        j0_all_MHC_approx.append(j0_history)
        reogr_e_all_MHC_approx.append(reorg_e_history)
        losses_all_MHC_approx.append(rmse_loss_history)



        fig,axs = plt.subplots(figsize=(8,13.5),nrows=3)

        ax = axs[0]
        ax.plot(rmse_loss_history)
        ax.set_yscale('log')
        ax.set_xlabel('Epochs',fontsize='large',fontweight='bold')
        ax.set_ylabel(r'RMSE, $mA/cm^2$',fontsize='large',fontweight='bold')

        ax = axs[1]
        ax.plot(j0_history)
        ax.set_xlabel('Epochs',fontsize='large',fontweight='bold')
        ax.set_ylabel(r'$j_0$, $mA/cm^2$',fontsize='large',fontweight='bold')

        ax = axs[2]
        ax.plot(reorg_e_history)
        ax.set_xlabel('Epochs',fontsize='large',fontweight='bold')
        ax.set_ylabel('reorg_e, eV',fontsize='large',fontweight='bold')
        ax.legend()

        fig.savefig(f"{project_history_folder}/DiffECDict_MH_approx_{SLRUM_ARRAY_TASK_ID}.png",dpi=250,bbox_inches='tight')
        plt.close('all')




