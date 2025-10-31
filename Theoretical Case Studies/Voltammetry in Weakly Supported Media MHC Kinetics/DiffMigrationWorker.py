import numpy as np 
import jax.numpy as jnp
import jax.scipy.linalg as linag
import os
import jax
jax.config.update('jax_enable_x64',True)
import jax.profiler
import pandas as pd
from simulation import simulation 
import json
import optax
import pickle

opt_state_file_name = "opt_state.pkl"

def load_experimental_flux(file_name):

    df = pd.read_csv(file_name)

    flux = jnp.array(df.iloc[:,1].to_numpy())

    return flux
def mse(y_true,y_pred):
    loss = jnp.mean((y_true - y_pred) ** 2)
    return loss

def compute_loss(params,sigma,C_sup,experimenal_flux):
    K0 = jnp.power(10.0,params[0])
    reogr_e = params[1]
    fluxes_predicted = simulation(K0=K0,reorg_e=reogr_e,sigma=sigma,C_sup=C_sup)
    loss = mse(experimenal_flux,fluxes_predicted)
    return loss




if __name__ == "__main__":

    with open('DiffMigrationDict.json') as f:
        diffMigrationDict = json.load(f)

    epochs=diffMigrationDict['epochs']
    logK0_initial = diffMigrationDict['logK0_initial']
    reorg_e_initial = diffMigrationDict['reorg_e_initial']
    sigma = diffMigrationDict['sigma']
    C_sup = diffMigrationDict['C_sup']
    lr=diffMigrationDict['lr']
    logK0_guess_history = diffMigrationDict["logK0_guess_history"]
    reorg_e_guess_history = diffMigrationDict['reorg_e_guess_history']
    loss_history =diffMigrationDict['loss_history']
    grad_history_1 = diffMigrationDict['grad_history_1']
    grad_history_2 = diffMigrationDict['grad_history_2']
    optimizer_name = diffMigrationDict['optimizer_name']
    experimental_flux_name = diffMigrationDict['experimental_flux_name']


    if len(reorg_e_guess_history) == 0:
        K0_guess = logK0_initial
        reorg_e_guess = reorg_e_initial
    else:
        K0_guess = logK0_guess_history[-1]
        reorg_e_guess = reorg_e_guess_history[-1]


    if optimizer_name == "sgd":
        optimizer = optax.sgd(lr,momentum=0.5)
    elif optimizer_name =='adam':
        optimizer = optax.adam(lr)
    else:
        raise ValueError

    experimental_flux =  load_experimental_flux(experimental_flux_name)

    value_and_grad  = jax.value_and_grad(compute_loss,argnums=0)

    params = jnp.array([K0_guess,reorg_e_guess])
    loss,grads = value_and_grad(params,sigma,C_sup,experimental_flux)
    opt_state = optimizer.init(params)


    step = len(logK0_guess_history)
    if  step == 0:
        opt_state = opt_state
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

    K0_guess = params[0]
    reorg_e_guess = params[1]
    


    loss_history.append(float(loss))
    grad_history_1.append(float(grads[0]))
    grad_history_2.append(float(grads[1]))
    logK0_guess_history.append(float(K0_guess))
    reorg_e_guess_history.append(float(reorg_e_guess))

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
