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
import sys 
json_file_name = sys.argv[1]
import optax
import pickle



opt_state_file_name = "opt_state.pkl"


def mse(y_true,y_pred):
    loss = jnp.mean((y_true - y_pred) ** 2)
    return loss



def compute_loss(params,experimental_currents):
    logdimKeq = params[0]
    logdimKf = params[1]
    dimKeq = jnp.power(10,logdimKeq)
    dimKf = jnp.power(10,logdimKf)
    currents_1 = simulation(dimKeq=dimKeq,dimKf=dimKf,cTstar=0.01)
    currents_2 = simulation(dimKeq=dimKeq,dimKf=dimKf,cTstar=0.02)
    currents_3 = simulation(dimKeq=dimKeq,dimKf=dimKf,cTstar=0.04)
    currents_4 = simulation(dimKeq=dimKeq,dimKf=dimKf,cTstar=0.1)

    simulation_currents = jnp.array([currents_1[-1],currents_2[-1],currents_3[-1],currents_4[-1]]) # unit, A 
    experimental_currents = jnp.array(experimental_currents) # Unit, A

    simulation_currents *= 1e9  # Unit, nA 
    experimental_currents *= 1e9 # Unit, nA

    loss = mse(experimental_currents,simulation_currents)

    return loss 


if __name__ == "__main__":

    # Convert and write JSON object to file
    with open(json_file_name) as f: 
        diffCEDict = json.load(f)



    epochs =  diffCEDict['epochs']
    logdimkeq_initial = diffCEDict['logdimkeq_initial']
    logdimKf_initial = diffCEDict['logdimKf_initial']
    lr_initial = diffCEDict['lr_initial']
    lr_effective = diffCEDict['lr_effective']
    logdimKeq_guess_history = diffCEDict['logdimKeq_guess_history']
    logdimKf_guess_history = diffCEDict['logdimKf_guess_history']
    loss_history = diffCEDict['loss_history']
    grad_history_1 = diffCEDict['grad_history_1']
    grad_history_2 = diffCEDict['grad_history_2']
    experimental_currents = diffCEDict['experimental_currents']
    optimizer_name = diffCEDict['optimizer_name']


    if len(logdimKeq_guess_history) == 0:
        logdimKeq_guess = logdimkeq_initial
        logdimKf_guess = logdimKf_initial
    else:
        logdimKeq_guess = logdimKeq_guess_history[-1]
        logdimKf_guess = logdimKf_guess_history[-1]

    print(f'Log Keq is {logdimKeq_guess:.2f} Log Kf is {logdimKf_guess:.2f}  Step is {len(logdimKeq_guess_history)}')

    """
    if len(logdimKeq_guess_history)>50:
        lr_effective = lr_initial * 0.99
    else:
        lr_effective = lr_initial
    """


    value_and_grad = jax.value_and_grad(compute_loss)

    if optimizer_name == "sgd":
        optimizer = optax.sgd(lr_effective,momentum=0.5)
    elif optimizer_name =='adam':
        optimizer = optax.adam(lr_effective)
    else:
        raise ValueError

    params = jnp.array([logdimKeq_guess,logdimKf_guess])
    loss,grads = value_and_grad(params,experimental_currents)
    opt_state = optimizer.init(params)


    step = len(logdimKeq_guess_history)
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


    logdimKeq_guess = params[0]
    logdimKf_guess = params[1]


    #logdimKeq_guess = logdimKeq_guess - lr*grad[0]
    #logdimKf_guess = logdimKf_guess - lr*grad[1]
    
    
    loss_history.append(float(loss))
    grad_history_1.append(float(grads[0]))
    grad_history_2.append(float(grads[1]))


    logdimKeq_guess_history.append(float(logdimKeq_guess))
    logdimKf_guess_history.append(float(logdimKf_guess))
    
    step = len(logdimKeq_guess_history)

    #checkpoint_manager.save(step, opt_state)
    with open(opt_state_file_name, "wb") as f:
        pickle.dump(opt_state, f)


    jax.debug.print('Loss is {loss}',loss=loss)
    jax.debug.print('Grad1 is {grad1}\nGrad2 is {grad2}',grad1=grads[0],grad2=grads[1])


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
