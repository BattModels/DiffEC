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

from SimulationHyperParameters import cycles,deltaX,deltaTheta,expanding_grid_factor,SimulationSpaceMultiple,Theta_i,Theta_v,scan_rates,sigmas,exp_dimensionless_files

from JAX_FD_Simulation_Unequal_D.simulation import simulation
from helper import flux_sampling,expParameters
from matplotlib.cm import viridis
colors = viridis(np.linspace(0,1,len(scan_rates)))



from matplotlib import pyplot as plt


import sys
SLRUM_ARRAY_TASK_ID = int(sys.argv[1])
opt_state_file_name = f"./opt_state_folder/opt_state_{SLRUM_ARRAY_TASK_ID}.pkl"



def load_experimental_flux(sigma,file_name):
    
    df = pd.read_csv(file_name)

    nTimeSteps = jnp.round(2*jnp.fabs(Theta_v-Theta_i)/deltaTheta*cycles)
    deltaT = deltaTheta/sigma
    maxT = 2.0*jnp.abs(Theta_v-Theta_i)/sigma * cycles
    time_array = jnp.arange(0,maxT,deltaT)

    flux_sampled = flux_sampling(time_array=time_array,df_FD=df,maxT=maxT)

    flux_sampled = jnp.array(flux_sampled)

    return flux_sampled

def mse(y_true,y_pred):
    loss = jnp.mean((y_true - y_pred) ** 2)
    return loss

def compute_loss(params,sigmas,experimental_fluxes):
    theta_corr = params[0]
    dA_guess = params[1]

    losses = 0

    for i,sigma in enumerate(sigmas):
        theta_simulated, flux_simulated = simulation(sigma=sigma,theta_corr=theta_corr,theta_i=Theta_i,theta_v=Theta_v,dA=dA_guess,dB=dA_guess,kinetics='Nernst')

        loss = mse(experimental_fluxes[i],flux_simulated)

        losses += loss

    return losses





if __name__ == "__main__":
    with open(f"./history_folder/DiffECDict_{SLRUM_ARRAY_TASK_ID}.json") as f:
        diffECDict = json.load(f)
    

    epochs = diffECDict['epochs']
    epochs = 603
    theta_corr_initial = diffECDict["theta_corr_initial"]
    dA_initial = diffECDict['dA_initial']
    theta_corr_history = diffECDict['theta_corr_history']
    dA_guess_history = diffECDict['dA_guess_history']
    lr = diffECDict['lr']
    optimizer_name = diffECDict['optimizer_name']
    loss_history = diffECDict['loss_history']


    
    if len(theta_corr_history) == 0:

        theta_corr_guess = theta_corr_initial
        dA_guess = dA_initial
    elif len(theta_corr_history) == epochs:
        print(f'There are {epochs} optimization records')
        sys.exit() 
    else:
        theta_corr_guess = theta_corr_history[-1]
        dA_guess = dA_guess_history[-1]

    if optimizer_name == "sgd":
        optimizer = optax.sgd(lr,momentum=0.5)
    elif optimizer_name =='adam':
        optimizer = optax.adam(lr)
    else:
        raise ValueError

    experimental_fluxes = []
    for sigma,experimental_file in zip(sigmas,exp_dimensionless_files):
        flux_sampled = load_experimental_flux(sigma,experimental_file)
        experimental_fluxes.append(flux_sampled)

    experimental_fluxes = jnp.array(experimental_fluxes)

    params = jnp.array([theta_corr_guess,dA_guess])

    #losses = compute_loss(params=params,sigmas=sigmas,experimental_fluxes=experimental_fluxes)

    #print(losses)

    if optimizer_name == "sgd":
        optimizer = optax.sgd(lr,momentum=0.5)
    elif optimizer_name =='adam':
        optimizer = optax.adam(lr)
    else:
        raise ValueError

    opt_state = optimizer.init(params)

    step = len(theta_corr_history)
    if  step == 0:
        opt_state = opt_state
    elif len(theta_corr_history) == epochs:
        print(f'There are {epochs} optimization records')
        sys.exit() 
    else:
        with open(opt_state_file_name, "rb") as f:
            opt_state = pickle.load(f)


    value_and_grad = jax.value_and_grad(compute_loss,argnums=0)

    loss,grads = value_and_grad(params,sigmas,experimental_fluxes)





    print('State before optimization',type(opt_state),opt_state)
    print('The grads of errors',grads)
    updates,opt_state = optimizer.update(grads, opt_state,params)
    print('State after optinmization',opt_state)
    print('Optimizer Updates',updates)
    print('Params before applying updates',params)
    params = optax.apply_updates(params,updates)
    print('Params after applying updates',params)

    #checkpoint_manager.save(step, opt_state)
    with open(opt_state_file_name, "wb") as f:
        pickle.dump(opt_state, f)
    jax.debug.print('Loss is {loss}',loss=loss)


    theta_corr_guess = params[0]
    dA_guess = params[1]

    loss_history.append(float(loss))
    theta_corr_history.append(float(theta_corr_guess))
    dA_guess_history.append(float(dA_guess))
    

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

    with open(f"./history_folder/DiffECDict_{SLRUM_ARRAY_TASK_ID}.json",'w') as outfile: 
        json.dump(diffECDict,outfile)



    fig, ax = plt.subplots(figsize=(8,4.5))




    theta_corr_guess = params[0]
    dA_guess = params[1]

    for index, exp_dimensionless_file in enumerate(exp_dimensionless_files):
        scan_rate = scan_rates[index]
        sigma = sigmas[index]

        df_exp = pd.read_csv(exp_dimensionless_file)
        ax.plot(df_exp.iloc[:,0],df_exp.iloc[:,1],label=f'Expt $\\nu={scan_rate:.3f} V/s$',alpha=0.8,color=tuple(colors[index]),ls='--')

        theta_simulated, flux_simulated = simulation(sigma=sigma,theta_corr=theta_corr_guess,theta_i=Theta_i,theta_v=Theta_v,dA=dA_guess,dB=dA_guess)
        ax.plot(theta_simulated,flux_simulated,color=tuple(colors[index]),ls='-',label=f'Simulation $\\nu={scan_rate:.3f} V/s$')


    ax.legend()
    #ax.title(f'logK0={float(logK0_guess):.3E},alpha={float(alpha_guess):.3E},beta={float(beta_guess):.3E},dA={float(dA_guess):.3E}')

    fig.savefig(f"./Figures/epoch={len(theta_corr_history)} {SLRUM_ARRAY_TASK_ID}.png",dpi=250,bbox_inches='tight')









