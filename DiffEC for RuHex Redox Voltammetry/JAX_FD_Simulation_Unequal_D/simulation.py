import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as liang
import os 
import jax
jax.config.update('jax_enable_x64',True)
from JAX_FD_Simulation_Unequal_D.grid import genGrid,ini_conc,update_d,calc_grad
from JAX_FD_Simulation_Unequal_D.coeff import ini_coeff,CalcMatrix,Allcalc_abc_linear,Allcacl_abc_radial
from SimulationHyperParameters import cycles,deltaX,deltaTheta,expanding_grid_factor,SimulationSpaceMultiple

import tqdm
import pandas as pd



def simulation(sigma=1.0,K0=1.0,alpha=0.5,beta=None,theta_corr=0.0,kinetics='Nernst',mode='linear',C_A_bulk=1.0,C_B_bulk=0.0,dA=1.0,dB=1.0,theta_i=20.0,theta_v = -20.0,saving_directory='./FD_Data',save_voltammogram=False):
    
    if not os.path.exists(saving_directory):
         os.mkdir(saving_directory)

    if beta is None:
        beta = 1.0-alpha

    #Start and end potential of voltammetric scan 


    nTimeSteps = jnp.round(2*jnp.fabs(theta_v-theta_i)/deltaTheta*cycles)
    deltaT = deltaTheta/sigma
    maxT = 2.0*jnp.abs(theta_v-theta_i)/sigma * cycles


    Esteps = jnp.arange(nTimeSteps)
    E = jnp.where(Esteps<nTimeSteps/2.0,theta_i-deltaTheta*Esteps,theta_v+deltaTheta*(Esteps-nTimeSteps/2.0))
    E = jnp.tile(E,cycles)
    
    fluxes = jnp.zeros_like(E)

    Xi = 0.0 #The location of the electrode surface. For a planar electrode in Cartesian coorindate, it is 0; For a spherical electrode in Radial electrode, it is 1.0
    if mode == "linear":
        maxX = SimulationSpaceMultiple * jnp.sqrt(maxT)
        Xi = 0.0
        
    elif mode == "radial":
        maxX = 1.0 + SimulationSpaceMultiple * jnp.sqrt(maxT)
        Xi = 1.0 
    else:
        raise ValueError


    X_grid, n = genGrid(Xi=Xi,deltaX = deltaX, maxX=maxX,expanding_grid_factor = expanding_grid_factor)
    conc,conc_d,concA,concB = ini_conc(n,C_A_bulk=C_A_bulk,C_B_bulk=C_B_bulk)
    A_matrix, aA,bA,cA,aB,bB,cB = ini_coeff(n=n,X_grid=X_grid,deltaX=deltaX,maxX=maxX,dA=dA,dB=dB)


    if mode == "linear":
        aA,bA,cA,aB,bB,cB = Allcalc_abc_linear(n=n,X_grid=X_grid,deltaT=deltaT,aA=aA,bA=bA,cA=cA,dA=dA,aB=aB,bB=bB,cB=cB,dB=dB)

    elif mode == "radial":
        aA,bA,cA,aB,bB,cB = Allcacl_abc_radial(n=n,X_grid=X_grid,deltaT=deltaT,aA=aA,bA=bA,cA=cA,dA=dA,aB=aB,bB=bB,cB=cB,dB=dB)

    else:
        raise ValueError

    for index in tqdm.tqdm(range(0,len(E))):
        Theta = E[index]

        A_matrix = CalcMatrix(A_matrix=A_matrix,X_grid=X_grid,Theta=Theta,kinetics=kinetics,n=n,aA=aA,bA=bA,cA=cA,dA=dA,aB=aB,bB=bB,cB=cB,dB=dB,K0=K0,alpha=alpha,beta=beta)

        conc_d = update_d(Theta=Theta,conc=conc,conc_d=conc_d,n=n,C_A_bulk=C_A_bulk,C_B_bulk=C_B_bulk,kinetics=kinetics,theta_corr=theta_corr)

        conc = liang.solve(A_matrix,conc_d)

        flux = calc_grad(conc,n,dA,dB,X_grid=X_grid)

        fluxes = fluxes.at[index].set(flux)


    if save_voltammogram:
        df = pd.DataFrame({'Potential':E,'Flux':fluxes})
        df.to_csv(f'{saving_directory}/sigma={sigma:.2E} K0={K0:.2E} alpha={alpha:.2E} beta={beta:.2E} kinetics={kinetics} mode={mode},dA={dA:.2E},dB={dB:.2E} theta_i={theta_i:.2E} theta_v={theta_v:.2E}.csv',index=False)


    return E,fluxes

if __name__ == "__main__":

    simulation(kinetics='Nernst',save_voltammogram=True)

