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

from DiffECHyperParameters import Y_sim, deltaY,deltaTheta

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

def update_d_Nernst(conc_grid,Theta,bulk_conc=1.0):
    conc_d_grid = jnp.copy(conc_grid)
    conc_d_grid = conc_d_grid.at[0].set(1.0/(1.0+jnp.exp(-Theta)))
    conc_d_grid = conc_d_grid.at[-1].set(bulk_conc)
    return conc_d_grid


def update_d_BV(conc_grid,Theta,K0,alpha,beta,deltaY,bulk_conc=1.0):
    Kred = K0*jnp.exp(-alpha*Theta)
    Kox = K0*jnp.exp(beta*Theta)


    conc_d_grid = jnp.copy(conc_grid)
    conc_d_grid = conc_d_grid.at[0].set(deltaY*Kox)
    conc_d_grid = conc_d_grid.at[-1].set(bulk_conc)
    return conc_d_grid


def update_d(kinetics,conc_grid,Theta,K0,alpha,beta,deltaY,bulk_conc=1.0):
    if kinetics == "BV":
        Kred = K0*jnp.exp(-alpha*Theta)
        Kox = K0*jnp.exp(beta*Theta)

        conc_d_grid = jnp.copy(conc_grid)
        conc_d_grid = conc_d_grid.at[0].set(deltaY*Kox)
        conc_d_grid = conc_d_grid.at[-1].set(bulk_conc)

    elif kinetics == "Nernst":
        conc_d_grid = jnp.copy(conc_grid)
        conc_d_grid = conc_d_grid.at[0].set(1.0/(1.0+jnp.exp(-Theta)))
        conc_d_grid = conc_d_grid.at[-1].set(bulk_conc)
    else:
        raise ValueError

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



def Acalc_abc(kinetics,A_matrix,Theta,K0,alpha,beta,Y_grid,W_grid,n,deltaT,deltaY):




    Arow = []
    Acol = []
    Aval = []

    if kinetics == "BV":
        Kred = K0*jnp.exp(-alpha*Theta)
        Kox = K0*jnp.exp(beta*Theta)

        Arow.append(0)
        Acol.append(0)
        Aval.append(1.0 + deltaY*Kred+deltaY*Kox)

        Arow.append(0)
        Acol.append(1)
        Aval.append(-1.0)
    elif kinetics == "Nernst":
        Arow.append(0)
        Acol.append(0)
        Aval.append(0.0)

        Arow.append(0)
        Acol.append(1)
        Aval.append(1.0)
    else:
        raise ValueError


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



def modA_abc(kinetics,A_matrix,Theta,K0,alpha,beta,deltaY):

    Arow = []
    Acol = []
    Aval = []


    if kinetics == "BV":
    
        Kred = K0*jnp.exp(-alpha*Theta)
        Kox =K0*jnp.exp(beta*Theta)




        Arow.append(0)
        Acol.append(0)
        Aval.append(1.0 + deltaY*Kred+deltaY*Kox)

        Arow.append(0)
        Acol.append(1)
        Aval.append(-1.0)
        A_matrix = A_matrix.at[jnp.array(Arow),jnp.array(Acol)].set(jnp.array(Aval))
    elif kinetics == "Nernst":
        pass
    else:
        raise ValueError



    return A_matrix



def update(Theta,K0,alpha,beta,A_matrix,conc_grid,conc_d_grid,Y_grid,deltaY):
    


    conc_grid = linalg.solve(A_matrix,conc_d_grid)

    flux = calcFlux(conc_grid,Y_grid)

    #flux_BV = calcBVFlux(conc_grid,Kred,Kox)

    return A_matrix,conc_grid,conc_d_grid,Y_grid,flux #,flux_BV



def simulation(sigma=40,K0=1.0,kinetics='BV',alpha=0.5,beta=0.5,theta_i=20.0,theta_v=-20.0,cycles=1,saving_directory="./Data",save_voltammogram=False):
    """
    sigma: dimensionless scan rate 
    K0: dimensionless standard electrochemical rate constant
    kinetics: BV or Nernst 
    alpha and beta: cathodic and anodic transfer coefficient 
    theta_i and theta_v: the start and reverse potential of voltammetry
    cycles, the number of CV cylcles 
    save_voltammogram. A boolean command to save voltammogram to csv file. Note that saving file will make this program non differentiable. 
    """


    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)

    Y_grid = jnp.arange(0.0,Y_sim,step=deltaY)
    n = Y_grid.shape[0]

    deltaT = deltaTheta/sigma
    maxT = cycles*2.0*abs(theta_v-theta_i)/sigma

    interpW = GetInterpW()
    W_grid = interpW(Y_grid).reshape(-1)
    conc_grid = jnp.zeros(n)

    conc_grid,conc_d_grid = init_c(conc_grid)

    A_matrix = jnp.zeros((n,n))

    
    #simulation steps
    nTimeSteps = int(2*np.fabs(theta_v-theta_i)/deltaTheta)
    Esteps = np.arange(nTimeSteps)
    E = np.where(Esteps<nTimeSteps/2.0,theta_i-deltaTheta*Esteps,theta_v+deltaTheta*(Esteps-nTimeSteps/2.0))
    E = np.tile(E,cycles)
    start_time = time.time()

    A_matrix = Acalc_abc(kinetics,A_matrix,theta_i,K0,alpha,beta,Y_grid,W_grid,n,deltaT,deltaY)
    conc_d_grid = update_d(kinetics,conc_grid,theta_i,K0,alpha,beta,deltaY)

    fluxes = []
    for index in tqdm.tqdm(range(0,int(len(E)))):

        Theta = E[index]

        A_matrix = modA_abc(kinetics,A_matrix,Theta,K0,alpha,beta,deltaY)
        conc_d_grid = update_d(kinetics,conc_grid,Theta,K0,alpha,beta,deltaY)
        A_matrix,conc_grid,conc_d_grid,Y_grid,flux  = update(Theta,K0,alpha,beta,A_matrix,conc_grid,conc_d_grid,Y_grid,deltaY)




        fluxes.append(flux)


        #Optional: Print concentration profile during training
        """
        if index in [0,1,2,3,4]:
            df_conc = pd.DataFrame({'Y':Y_grid,'C':conc_grid})
            df_conc.to_csv(f'{CV_location} conc {index}.csv',index=False)
        """



    if save_voltammogram:
        df = pd.DataFrame({'Potential':E,'Flux':fluxes})
        df.to_csv(f'{saving_directory}/{kinetics} sigma={sigma:.2E}.csv',index=False)
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax.plot(df.iloc[:,0],df.iloc[:,1],label=f"sigma")
        ax.set_xlabel(r'Potential, $\theta$',fontsize='large',fontweight='bold')
        ax.set_ylabel(r'Flux, $J$',fontsize='large',fontweight='bold')
        fig.savefig(f'{saving_directory}/{kinetics} sigma={sigma:.2E}.png',dpi=200)
    return jnp.array(E),jnp.array(fluxes)   




if __name__ == "__main__":
    simulation(kinetics="BV",save_voltammogram=True)
    simulation(kinetics="Nernst",save_voltammogram=True)



    #Let's propose some common rotating diks simulation parameetrs

    nu = 1e-6 # m^2 s^-1 kinemaic viscosity of water
    freq = 50 # Hz
    v = 1.0 # Scan Rate, V/s
    L = 0.51023*(2*np.pi*freq)**1.5*nu**(-0.5)
    D = 1e-9 # m^2s^-1, a typical diffusion efficient
    sigma = 96485*v/(8.314*298)*(L**2*D)**(-1/3) # Dimensionless scan rate correlating rotational speed and scan rate

    

    simulation(kinetics="Nernst",sigma=sigma,save_voltammogram=True)
