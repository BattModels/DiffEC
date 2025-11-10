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
from matplotlib import pyplot as plt
from helper import flux_sampling
from SimulationHyperParameters import sigmas,voltammograms_ground_truth,cycles,deltaX,deltaTheta,expanding_grid_factor,SimulationSpaceMultiple,Theta_i,Theta_v,sigmas
from simulation import simulation
from matplotlib.cm import viridis
colors = viridis(np.linspace(0,1,len(sigmas)))

import sys
SLRUM_ARRAY_TASK_ID = int(sys.argv[1])

opt_state_file_name = f"./opt_state_folder/opt_state_{SLRUM_ARRAY_TASK_ID}.pkl"

def load_experimental_flux(sigma,file_name):
    
    df = pd.read_csv(file_name)
    """
    nTimeSteps = jnp.round(2*jnp.fabs(Theta_v-Theta_i)/deltaTheta*cycles)
    deltaT = deltaTheta/sigma
    maxT = 2.0*jnp.abs(Theta_v-Theta_i)/sigma * cycles
    time_array = jnp.arange(0,maxT,deltaT)
    flux_sampled = flux_sampling(time_array=time_array,df_FD=df,maxT=maxT)
    
    flux_sampled = jnp.array(flux_sampled)
    """

    flux_sampled = jnp.array(df.iloc[:,1])

    return flux_sampled

def mse(y_true,y_pred):
    loss = jnp.mean((y_true - y_pred) ** 2)
    return loss

def compute_loss(params,sigmas,experimental_fluxes):
    logK0_guess = params[0]
    K0_guess = jnp.power(10,logK0_guess)
    alpha_guess = params[1]
    beta_guess = params[2]

    logK0_ads_guess = params[3]
    K0_ads_guess = jnp.power(10,logK0_ads_guess)
    alpha_ads_guess = params[4]
    beta_ads_guess = params[5]

    logK_A_ads_guess = params[6]
    K_A_ads_guess = jnp.power(10,logK_A_ads_guess)
    logK_B_ads_guess = params[7]
    K_B_ads_guess = jnp.power(10,logK_B_ads_guess)
    logK_A_des_guess = params[8]
    K_A_des_guess = jnp.power(10,logK_A_des_guess)
    logK_B_des_guess = params[9]
    K_B_des_guess = jnp.power(10,logK_B_des_guess)

    losses = 0

    for i,sigma in enumerate(sigmas):
        theta_simulated,flux_simulated = simulation(sigma=sigma,K0=K0_guess,alpha=alpha_guess,beta=beta_guess,K0_ads=K0_ads_guess,alpha_ads=alpha_ads_guess,beta_ads=beta_ads_guess,K_A_ads=K_A_ads_guess,K_B_ads=K_B_ads_guess,K_A_des=K_A_des_guess,K_B_des=K_B_des_guess,theta_i=Theta_i,theta_v=Theta_v)

        loss = mse(experimental_fluxes[i]/jnp.sqrt(sigma),flux_simulated/jnp.sqrt(sigma))

        losses += loss 

    return losses 



if __name__ == "__main__":
    with open(f"./history_folder/DiffECDict_{SLRUM_ARRAY_TASK_ID}.json") as f:
        diffECDict = json.load(f)

    epochs = diffECDict['epochs']
    optimizer_name = diffECDict['optimizer_name']
    lr = diffECDict['lr']
    loss_history = diffECDict['loss_history']

    logK0_initial = diffECDict["logK0_initial"]
    alpha_initial = diffECDict["alpha_initial"]
    beta_initial = diffECDict["alpha_initial"]

    logK0_ads_initial = diffECDict["logK0_ads_initial"]
    alpha_ads_initial = diffECDict["alpha_ads_initial"]
    beta_ads_initial = diffECDict["beta_ads_initial"]

    logK_A_ads_initial = diffECDict["logK_A_ads_initial"]
    logK_B_ads_initial = diffECDict["logK_B_ads_initial"]
    logK_A_des_initial = diffECDict["logK_A_des_initial"]
    logK_B_des_initial = diffECDict["logK_B_des_initial"]

    logK0_guess_history = diffECDict["logK0_guess_history"]
    alpha_guess_history = diffECDict["alpha_guess_history"]
    beta_guess_history = diffECDict["beta_guess_history"]

    logK0_ads_guess_history = diffECDict["logK0_ads_guess_history"]
    alpha_ads_guess_history = diffECDict["alpha_ads_guess_history"]
    beta_ads_guess_history = diffECDict["beta_ads_guess_history"]

    logK_A_ads_guess_history = diffECDict["logK_A_ads_guess_history"]
    logK_B_ads_guess_history = diffECDict["logK_B_ads_guess_history"]
    logK_A_des_guess_history = diffECDict["logK_A_des_guess_history"]
    logK_B_des_guess_history = diffECDict["logK_B_des_guess_history"]



    if len(logK0_guess_history) == 0:
        logK0_guess = logK0_initial
        alpha_guess = alpha_initial
        beta_guess = beta_initial

        logK0_ads_guess = logK0_ads_initial
        alpha_ads_guess = alpha_ads_initial
        beta_ads_guess = alpha_ads_initial

        logK_A_ads_guess = logK_A_ads_initial
        logK_B_ads_guess = logK_B_ads_initial
        logK_A_des_guess = logK_A_des_initial
        logK_B_des_guess = logK_B_des_initial

    elif len(logK0_guess_history) >= epochs:
        print(f'There are {epochs} optimization records')
        sys.exit() 

    else:
        logK0_guess = logK0_guess_history[-1]
        alpha_guess = alpha_guess_history[-1]
        beta_guess = beta_guess_history[-1]

        logK0_ads_guess = logK0_ads_guess_history[-1]
        alpha_ads_guess = alpha_ads_guess_history[-1]
        beta_ads_guess = alpha_ads_guess_history[-1]

        logK_A_ads_guess = logK_A_ads_guess_history[-1]
        logK_B_ads_guess = logK_B_ads_guess_history[-1]
        logK_A_des_guess = logK_A_des_guess_history[-1]
        logK_B_des_guess = logK_B_des_guess_history[-1]

    if optimizer_name == "sgd":
        optimizer = optax.sgd(lr,momentum=0.5)
    elif optimizer_name =='adam':
        optimizer = optax.adam(lr)
    else:
        raise ValueError
    
    experimental_fluxes = []
    
    for sigma,experimental_file in zip(sigmas,voltammograms_ground_truth):
        flux_sampled = load_experimental_flux(sigma,experimental_file)
        experimental_fluxes.append(flux_sampled)

    
    experimental_fluxes = jnp.array(experimental_fluxes)

    params = jnp.array([
        logK0_guess,
        alpha_guess,
        beta_guess,
        logK0_ads_guess,
        alpha_ads_guess,
        beta_ads_guess,
        logK_A_ads_guess,
        logK_B_ads_guess,
        logK_B_ads_guess,
        logK_B_des_guess
    ])
    value_and_grad = jax.value_and_grad(compute_loss,argnums=0)

    loss,grads = value_and_grad(params,sigmas,experimental_fluxes)


    opt_state = optimizer.init(params)

    step = len(logK0_guess_history)
    if  step == 0:
        opt_state = opt_state
    elif len(logK0_guess_history) == epochs:
        print(f'There are {epochs} optimization records')
        sys.exit() 
    else:
        with open(opt_state_file_name, "rb") as f:
            opt_state = pickle.load(f)


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

    logK0_guess = params[0]
    K0_guess = jnp.power(10,logK0_guess)
    alpha_guess = params[1]
    beta_guess = params[2]

    logK0_ads_guess = params[3]
    K0_ads_guess = jnp.power(10,logK0_ads_guess)
    alpha_ads_guess = params[4]
    beta_ads_guess = params[5]

    logK_A_ads_guess = params[6]
    K_A_ads_guess = jnp.power(10,logK_A_ads_guess)
    logK_B_ads_guess = params[7]
    K_B_ads_guess = jnp.power(10,logK_B_ads_guess)
    logK_A_des_guess = params[8]
    K_A_des_guess = jnp.power(10,logK_A_des_guess)
    logK_B_des_guess = params[9]
    K_B_des_guess = jnp.power(10,logK_B_des_guess)

    loss_history.append(float(loss))
    logK0_guess_history.append(float(logK0_guess))
    alpha_guess_history.append(float(alpha_guess))
    beta_guess_history.append(float(beta_guess))

    logK0_ads_guess_history.append(float(logK0_ads_guess))
    alpha_ads_guess_history.append(float(alpha_ads_guess))
    beta_ads_guess_history.append(float(beta_ads_guess))

    logK_A_ads_guess_history.append(float(logK_A_ads_guess))
    logK_B_ads_guess_history.append(float(logK_B_ads_guess))
    logK_A_des_guess_history.append(float(logK_A_des_guess))
    logK_B_des_guess_history.append(float(logK_B_des_guess))

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

    with open(f"./history_folder/DiffECDict_{SLRUM_ARRAY_TASK_ID}.json",'w') as outfile: 
        json.dump(diffECDict,outfile)


    fig, ax = plt.subplots(figsize=(8,4.5))

    for index,file_name in enumerate(voltammograms_ground_truth):
        sigma = sigmas[index]
        df_exp = pd.read_csv(file_name)
        ax.plot(df_exp.iloc[:,0],df_exp.iloc[:,1]/np.sqrt(sigma),alpha=0.8,color=tuple(colors[index]),ls='--')

        theta_simulated, flux_simulated  = simulation(sigma=sigma,K0=K0_guess,alpha=alpha_guess,beta=beta_guess,K0_ads=K0_ads_guess,alpha_ads=alpha_ads_guess,beta_ads=beta_ads_guess,K_A_ads=K_A_ads_guess,K_B_ads=K_B_ads_guess,K_A_des=K_A_des_guess,K_B_des=K_B_des_guess,theta_i=Theta_i,theta_v=Theta_v)
        ax.plot(theta_simulated,flux_simulated/np.sqrt(sigma),color=tuple(colors[index]),ls='-',label=f'sigma={sigma:.2E}')


    ax.set_xlabel(r'Potential, $\theta$')
    ax.set_ylabel(r'Normalized Flux, $J/\sqrt{\sigma}$')
    ax.legend()
    fig.savefig(f"./Figures/epoch={len(logK0_guess_history)} {SLRUM_ARRAY_TASK_ID}.png",dpi=250,bbox_inches='tight')