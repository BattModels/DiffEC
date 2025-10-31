import numpy as np 
import json
import jax
import jax.numpy as jnp
from main import mse,simulation,calcSigma
from helper import load_experimental_flux
import optax
import sys
from SimulationHyperParameters import nu,D,c_bulk,rot_freq,omega,scan_rate,theta_i,theta_v,lr,log10K0_initial,log10K0_initial_range,alpha_initial,alpha_initial_range,epochs,exp_dimensionless_file,optimizer_name
from matplotlib import pyplot as plt
import pickle

SLURM_ARRAY_TASK_ID =  int(sys.argv[1])
opt_state_file_name = f"./opt_state_folder/opt_state_{SLURM_ARRAY_TASK_ID}.pkl"

def compute_loss(simulation_params,sigma,experimental_flux,theta_i,theta_v):
    alpha_guess = simulation_params[0]
    log10K0_guess = simulation_params[1]
    K0 = jnp.power(10,log10K0_guess)
    theta_simulated, fluxes_predicted = simulation(sigma=sigma,K0=K0,alpha=alpha_guess,theta_i=theta_i,theta_v=theta_v)

    loss = mse(experimental_flux[:],fluxes_predicted[:])

    return loss 



if __name__ == "__main__":

    with open(f"./history_folder/DiffECDict_{SLURM_ARRAY_TASK_ID}.json") as f:
        diffECDict = json.load(f)
    
    epochs = diffECDict['epochs']
    log10K0_initial = diffECDict['log10K0_initial']
    alpha_initial = diffECDict['alpha_initial']
    sigma = diffECDict['sigma']
    exp_dimensionless_file = diffECDict['exp_dimensionless_file']
    log10K0_guess_history = diffECDict['log10K0_guess_history']
    alpha_guess_history = diffECDict['alpha_guess_history']
    loss_history = diffECDict['loss_history']
    lr = diffECDict['lr']
    optimizer_name = diffECDict['optimizer_name']

    


    experimental_flux = load_experimental_flux(exp_dimensionless_file)

    if len(log10K0_guess_history) == 0:
    
        alpha_guess = alpha_initial
        log10K0_guess = log10K0_initial
    elif len(log10K0_guess_history) >=epochs:
        print(f'There are {epochs} optimization records')
        sys.exit() 
    else:
        alpha_guess = alpha_guess_history[-1]
        log10K0_guess = log10K0_guess_history[-1]

    params = jnp.array([alpha_guess,log10K0_guess])


    value_and_grd = jax.value_and_grad(compute_loss,argnums=0)

    if optimizer_name == "sgd":
        optimizer = optax.sgd(lr,momentum=0.5)
    elif optimizer_name =='adam':
        optimizer = optax.adam(lr)
    else:
        raise ValueError
    

    opt_state = optimizer.init(params=params)
    step = len(log10K0_guess_history)
    if  step == 0:
        opt_state = opt_state
    elif len(log10K0_guess_history) >= epochs:
        print(f'There are {epochs} optimization records')
        sys.exit() 
    else:
        with open(opt_state_file_name, "rb") as f:
            opt_state = pickle.load(f)


    loss, grads = value_and_grd(params,sigma,experimental_flux,theta_i=theta_i,theta_v=theta_v)
    
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



    alpha_guess_history.append(float(params[0]))
    log10K0_guess_history.append(float(params[1]))
    loss_history.append(float(loss))

    if len(log10K0_guess_history) >= epochs:
        #If there are enough optimization results, the master program exits. 
        sys.exit()

    diffECDict = {
        "epochs":epochs,
        "log10K0_initial":log10K0_initial,
        "alpha_initial":alpha_initial,
        "sigma":sigma,
        "exp_dimensionless_file":exp_dimensionless_file,
        "log10K0_guess_history":log10K0_guess_history,
        "alpha_guess_history":alpha_guess_history,
        "loss_history":loss_history,
        "lr":lr,
        "optimizer_name":optimizer_name 
    }

    with open(f"./history_folder/DiffECDict_{SLURM_ARRAY_TASK_ID}.json",'w') as outfile: 
        json.dump(diffECDict,outfile)


    fig,ax = plt.subplots(figsize=(8,4.5))

    alpha_guess = params[0]
    log10K0_guess = params[1]
    K0 = jnp.power(10,log10K0_guess)
    theta_simulated,flux_simulated = simulation(sigma=sigma,K0=K0,alpha=alpha_guess,theta_i=theta_i,theta_v=theta_v)
    
    ax.plot(theta_simulated[:],experimental_flux[:],alpha=0.8,ls='--')
    ax.plot(theta_simulated[:],flux_simulated[:],label='Simulation')
    ax.legend()


    fig.savefig(f"./Figures/epoch={len(log10K0_guess_history)} {SLURM_ARRAY_TASK_ID}.png",dpi=250,bbox_inches='tight')