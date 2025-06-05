import numpy as np 
import jax.numpy as jnp
import jax.scipy.linalg as linalg

import jax
import time
import os
from matplotlib import pyplot as plt
from helper import GetInterpW
import pandas as pd
import tqdm

jax.config.update('jax_enable_x64',True)




def mse(y_true,y_pred):
    loss = jnp.mean((y_true - y_pred) ** 2)
    return loss

def calcSigma(omega,nu,D,scan_rate):
    """
    This function returns the dimensionless scan rate as a function of rotational speed, kinematic viscosity 
    omega: Rotational speed in rad/s 
    nu: Kinematic viscosity , m^2 s^-1 
    D: Diffusion Coefficient, m^2 s^-1 
    scan_rate: V/s
    """
    F = 96485 #Faraday Constant C/mol 
    R = 8.314 # Gas Constant, J/(mol * K)
    T = 298 # Temperature, K

    L = 0.51023 * omega **(1.5) * nu**(-0.5)

    sigma = F * scan_rate / (R*T) *(L**2*D)**(-1.0/3.0)


    return sigma

    


def init_c(conc_grid,bulk_conc=1.0,):
    conc_grid = conc_grid.at[:].set(bulk_conc)
    conc_d_grid = jnp.copy(conc_grid)

    return conc_grid,conc_d_grid
"""
def update_d_Nernst(conc_grid,Theta,bulk_conc=1.0):
    conc_d_grid = jnp.copy(conc_grid)
    conc_d_grid = conc_d_grid.at[0].set(1.0/(1.0+jnp.exp(-Theta)))
    conc_d_grid = conc_d_grid.at[-1].set(bulk_conc)
    return conc_d_grid
"""

def update_d_BV(conc_grid,Theta,K0,alpha,beta,deltaY,bulk_conc=1.0):
    Kred = K0*jnp.exp(-alpha*Theta)
    Kox = K0*jnp.exp(beta*Theta)


    conc_d_grid = jnp.copy(conc_grid)
    conc_d_grid = conc_d_grid.at[0].set(deltaY*Kox)
    conc_d_grid = conc_d_grid.at[-1].set(bulk_conc)
    return conc_d_grid

def calcFlux(conc_grid,Y_grid):
    flux = - (conc_grid[1]-conc_grid[0])/(Y_grid[1] - Y_grid[0])
    return flux

def calcBVFlux(conc_grid,Kred,Kox):
    #print(conc_grid[0]*Kred)
    #print((1.0-conc_grid[0])*Kox)
    #input()
    flux_BV = - (conc_grid[0]*Kred - ((1.0-conc_grid[0])*Kox))
    return flux_BV


def Acalc_abc(A_matrix,Theta,K0,alpha,beta,Y_grid,W_grid,n,deltaT,deltaY):

    Kred = K0*jnp.exp(-alpha*Theta)
    Kox = K0*jnp.exp(beta*Theta)


    Arow = []
    Acol = []
    Aval = []

    Arow.append(0)
    Acol.append(0)
    Aval.append(1.0 + deltaY*Kred+deltaY*Kox)

    Arow.append(0)
    Acol.append(1)
    Aval.append(-1.0)


    for i in range(1,n-1):
        W = W_grid[i]
        Arow.append(i)
        Acol.append(i-1)
        Aval.append(-jnp.exp(-2.0/3.0*W**3)/1.65894 * deltaT / (deltaY * deltaY))

        Arow.append(i)
        Acol.append(i)
        Aval.append(2.0 * jnp.exp(-2.0/3.0*W**3)/1.65894 * deltaT / (deltaY * deltaY) + 1.0 )

        Arow.append(i)
        Acol.append(i+1)
        Aval.append(-jnp.exp(-2.0/3.0*W**3)/1.65894 * deltaT / (deltaY * deltaY))


    Arow.append(-1)
    Acol.append(-1)
    Aval.append(1.0)

    A_matrix = A_matrix.at[jnp.array(Arow),jnp.array(Acol)].set(jnp.array(Aval))

    return A_matrix


def modA_abc(A_matrix,Theta,K0,alpha,beta,deltaY):

    Kred = K0*jnp.exp(-alpha*Theta)
    Kox =K0*jnp.exp(beta*Theta)


    Arow = []
    Acol = []
    Aval = []

    Arow.append(0)
    Acol.append(0)
    Aval.append(1.0 + deltaY*Kred+deltaY*Kox)

    Arow.append(0)
    Acol.append(1)
    Aval.append(-1.0)

    A_matrix = A_matrix.at[jnp.array(Arow),jnp.array(Acol)].set(jnp.array(Aval))

    return A_matrix

#@jax.jit
def update(Theta,K0,alpha,beta,A_matrix,conc_grid,conc_d_grid,Y_grid,deltaY):
    
    Kred = K0*jnp.exp(-alpha*Theta)
    Kox = K0*jnp.exp(beta*Theta)


    conc_grid = linalg.solve(A_matrix,conc_d_grid)



    flux = calcFlux(conc_grid,Y_grid)

    flux_BV = calcBVFlux(conc_grid,Kred,Kox)

    return A_matrix,conc_grid,conc_d_grid,Y_grid,flux,flux_BV




def simulation(sigma,K0,alpha,beta,theta_i,theta_v,CV_location=None,save_voltammogram=False):
    #start and end potential of voltammetric scan


    Y_sim = 0.9999
    deltaY = 2e-3
    Y_grid = jnp.arange(0.0,Y_sim,step=deltaY)
    n = Y_grid.shape[0]

    deltaTheta = 2e-3  # potential step of simulation

    deltaT = deltaTheta/sigma
    maxT = abs(theta_v-theta_i)/sigma

    interpW = GetInterpW()
    W_grid = interpW(Y_grid).reshape(-1)
    conc_grid = jnp.zeros(n)

    conc_grid,conc_d_grid = init_c(conc_grid)

    A_matrix = jnp.zeros((n,n))

    
    #simulation steps
    nTimeSteps = int(np.fabs(theta_v-theta_i)/deltaTheta)
    Esteps = np.arange(nTimeSteps)
    E = theta_i-deltaTheta*Esteps

    start_time = time.time()

    A_matrix = Acalc_abc(A_matrix,theta_i,K0,alpha,beta,Y_grid,W_grid,n,deltaT,deltaY)
    conc_d_grid = update_d_BV(conc_grid,theta_i,K0,alpha,beta,deltaY)

    fluxes = []
    fluxes_BV = []
    for index in tqdm.tqdm(range(0,int(len(E)))):

        Theta = E[index]

        A_matrix = modA_abc(A_matrix,Theta,K0,alpha,beta,deltaY)
        conc_d_grid = update_d_BV(conc_grid,Theta,K0,alpha,beta,deltaY)
        A_matrix,conc_grid,conc_d_grid,Y_grid,flux,flux_BV  = update(Theta,K0,alpha,beta,A_matrix,conc_grid,conc_d_grid,Y_grid,deltaY)

        #print(Theta,conc_grid[0],conc_grid[1],conc_grid[-2],conc_grid[-1])


        fluxes.append(flux)
        fluxes_BV.append(flux_BV)

        #Optional: Print concentration profile during training
        """
        if index in [0,1,2,3,4]:
            df_conc = pd.DataFrame({'Y':Y_grid,'C':conc_grid})
            df_conc.to_csv(f'{CV_location} conc {index}.csv',index=False)
        """



    if save_voltammogram:
        df = pd.DataFrame({'Potential':E,'Flux':fluxes})
        df.to_csv(f'{CV_location}.csv',index=False)

    return jnp.array(fluxes)
    



if __name__ =="__main__":
    """
    for sigma in [0.001,0.01,0.1,0.5,1,10,30,100,1000]:
        fluxes = simulation(sigma)
    """

    ground_truth_nu = 1e-6# m^2s^-1
    rot_freq =  20 # Hz
    omega = rot_freq*np.pi*2 #Rad/second
    D = 1e-9 #m^2s^-1
    scan_rate = 1e-3 # V/s
    ground_truth_sigma = calcSigma(omega=omega,nu=ground_truth_nu,D=D,scan_rate=scan_rate)
    
    for K0 in [1e-1]:
        for alpha in [0.45]:
            for beta in [0.40]:
                saving_directory = f"ExperimentalCV"
                if not os.path.exists(saving_directory):
                    os.mkdir(saving_directory)
                CV_location  = f'{saving_directory}/rot_freq={rot_freq:.2E} nu={ground_truth_nu:.2E} sigma={ground_truth_sigma:.2E} K0={K0:.2E} alpha={alpha:.2E} beta={beta:.2E}'

                simulation(ground_truth_sigma,K0=K0,alpha=alpha,beta=beta, CV_location = CV_location,save_voltammogram=True)

