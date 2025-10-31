import numpy as np 
import json
import jax
import jax.numpy as jnp 
import optax
import sys
import pandas as pd
from matplotlib import pyplot as plt
from KineticModels import BV_current,MH_current,MHC_current,MHC_current_approx
from DiffECHyperParameters import j0_initial_guess_central,reorg_e_initial_guess_central,optimizer_name,lr,epochs,j0_initial_guess_range,reorg_e_initial_guess_range,j0_initial_guess_central_MHC_approx
import os
jax.config.update("jax_enable_x64", True) #It's very important to enable float64 in JAX computation; The default float 32 does not satisfy the precision requirement. 
SLRUM_ARRAY_TASK_ID = 0#int(sys.argv[1])


F = 96485 # C/mol 
R = 8.314 # J/(mol * k)
T = 298 #K



system_names = [r"10% FEC",r'EC DEC',r'DEC',r'PC']
df = pd.read_csv("Boyle Figure 4a.csv",skiprows=1)


def mse(y_true,y_pred):
    loss = jnp.mean((y_true - y_pred) ** 2)
    return loss


def compute_loss_MH(params,expt_eta,expt_current_density,):
    j0 = params[0]
    reorg_e = params[1]

    j_MH = MH_current(j0=j0,reorg_e=reorg_e,eta=expt_eta)

    loss = mse(expt_current_density,j_MH)

    return loss 

def compute_loss_MHC(params,expt_eta,expt_current_density):
    j0 = params[0]
    reorg_e = params[1]
    
    j_MHC = MHC_current(j0=j0,reorg_e=reorg_e,eta=expt_eta)

    loss = mse(expt_current_density,j_MHC)

    return loss 

def compute_loss_MHC_approx(params,expt_eta,expt_current_density):
    j0 = params[0]
    reorg_e = params[1]
    
    j_MHC = MHC_current_approx(j0=j0,reorg_e=reorg_e,eta=expt_eta)

    loss = mse(expt_current_density,j_MHC)

    return loss 

for index, system_name in enumerate(system_names):

    project_history_folder = f'./history_folder_{system_name}'
    if not os.path.exists(project_history_folder):
        os.mkdir(project_history_folder)

    expt_eta = jnp.array(df.iloc[:,index*2].dropna())
    expt_current_density = jnp.array(df.iloc[:,index*2+1].dropna())

    j0_initial_guess = np.random.uniform(j0_initial_guess_central-j0_initial_guess_range,j0_initial_guess_central+j0_initial_guess_range)
    reorg_e_initial_guess = np.random.uniform(reorg_e_initial_guess_central-reorg_e_initial_guess_range,reorg_e_initial_guess_central+reorg_e_initial_guess_range)
    
    
    ##########################################################
    print(f"{system_name} MH")
    params = jnp.array([j0_initial_guess,reorg_e_initial_guess]) 
    if optimizer_name == "sgd":
        optimizer = optax.sgd(lr)
    elif optimizer_name =='adam':
        optimizer = optax.adam(lr)
    else:
        raise ValueError
    opt_state = optimizer.init(params)

    value_and_grad_MH = jax.value_and_grad(compute_loss_MH,argnums=0)

    j0_history = []
    reorg_e_history = []
    rmse_loss_history = []

    for i in range(epochs):
        loss,grads = value_and_grad_MH(params,expt_eta,expt_current_density)
        #print('State before optimization',type(opt_state),opt_state)
        #print('The grads of errors',grads)
        updates,opt_state = optimizer.update(grads, opt_state,params)
        #print('State after optinmization',opt_state)
        #print('Optimizer Updates',updates)
        #print('Params before applying updates',params)
        params = optax.apply_updates(params,updates)
        #print('Params after applying updates',params)

        rmse_loss = jnp.sqrt(loss)
        #print(f"The rmse loss is {rmse_loss:.2f}")

        j0_history.append(float(params[0]))
        reorg_e_history.append(float(params[1]))
        rmse_loss_history.append(float(rmse_loss))

        if np.isnan(rmse_loss):
            print(f'Nan value identified, {j0_initial_guess} {reorg_e_initial_guess}')

    
    diffECDict = {
        "j0_initial_guess":j0_initial_guess,
        "reorg_e_initial_guess":reorg_e_initial_guess,
        "j0_history":j0_history,
        "reorg_e_history":reorg_e_history,
        "rmse_loss_history":rmse_loss_history
    }

    with open(f"{project_history_folder}/DiffECDict_MH_{SLRUM_ARRAY_TASK_ID}.json",'w') as outfile: 
        json.dump(diffECDict,outfile)


    ####################################################
    print(f"{system_name} MHC")
    params = jnp.array([j0_initial_guess,reorg_e_initial_guess])
    if optimizer_name == "sgd":
        optimizer = optax.sgd(lr)
    elif optimizer_name =='adam':
        optimizer = optax.adam(lr)
    else:
        raise ValueError
    opt_state = optimizer.init(params)

    j0_history = []
    reorg_e_history = []
    rmse_loss_history = []

    value_and_grad_MH = jax.value_and_grad(compute_loss_MHC,argnums=0)
    for i in range(epochs):
        loss,grads = value_and_grad_MH(params,expt_eta,expt_current_density)

        #print('State before optimization',type(opt_state),opt_state)
        #print('The grads of errors',grads)
        updates,opt_state = optimizer.update(grads, opt_state,params)
        #print('State after optinmization',opt_state)
        #print('Optimizer Updates',updates)
        #print('Params before applying updates',params)
        params = optax.apply_updates(params,updates)
        #print('Params after applying updates',params)

        rmse_loss = jnp.sqrt(loss)
        #print(f"The rmse loss is {rmse_loss:.2f}")


        j0_history.append(float(params[0]))
        reorg_e_history.append(float(params[1]))
        rmse_loss_history.append(float(rmse_loss))

        if np.isnan(rmse_loss):
            print(f'Nan value identified, {j0_initial_guess} {reorg_e_initial_guess}')



    diffECDict = {
        "j0_initial_guess":j0_initial_guess,
        "reorg_e_initial_guess":reorg_e_initial_guess,
        "j0_history":j0_history,
        "reorg_e_history":reorg_e_history,
        "rmse_loss_history":rmse_loss_history
    }

    with open(f"{project_history_folder}/DiffECDict_MHC_{SLRUM_ARRAY_TASK_ID}.json",'w') as outfile: 
        json.dump(diffECDict,outfile)
    




    
    ####################################################### 
    print(f"{system_name} MHC Approx")

    params = jnp.array([j0_initial_guess_central_MHC_approx,reorg_e_initial_guess])
    if optimizer_name == "sgd":
        optimizer = optax.sgd(lr,momentum=0.5)
    elif optimizer_name =='adam':
        optimizer = optax.adam(lr)
    else:
        raise ValueError
    opt_state = optimizer.init(params)

    value_and_grad_MH = jax.value_and_grad(compute_loss_MHC_approx,argnums=0)

    j0_history = []
    reorg_e_history = []
    rmse_loss_history = []

    for i in range(epochs):
        loss,grads = value_and_grad_MH(params,expt_eta,expt_current_density)

        #print('State before optimization',type(opt_state),opt_state)
        #print('The grads of errors',grads)
        updates,opt_state = optimizer.update(grads, opt_state,params)
        #print('State after optinmization',opt_state)
        #print('Optimizer Updates',updates)
        #print('Params before applying updates',params)
        params = optax.apply_updates(params,updates)
        #print('Params after applying updates',params)

        rmse_loss = jnp.sqrt(loss)
        #print(f"The rmse loss is {rmse_loss:.2f}")
        j0_history.append(float(params[0]))
        reorg_e_history.append(float(params[1]))
        rmse_loss_history.append(float(rmse_loss))

        if np.isnan(rmse_loss):
            print(f'Nan value identified, {j0_initial_guess} {reorg_e_initial_guess}')


    diffECDict = {
        "j0_initial_guess":j0_initial_guess,
        "reorg_e_initial_guess":reorg_e_initial_guess,
        "j0_history":j0_history,
        "reorg_e_history":reorg_e_history,
        "rmse_loss_history":rmse_loss_history
    }

    with open(f"{project_history_folder}/DiffECDict_MH_approx_{SLRUM_ARRAY_TASK_ID}.json",'w') as outfile: 
        json.dump(diffECDict,outfile)
    