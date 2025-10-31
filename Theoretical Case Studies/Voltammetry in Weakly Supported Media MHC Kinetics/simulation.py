import numpy as np 
import jax.numpy as jnp
import jax.scipy.linalg as linag
import os
import jax

jax.config.update('jax_enable_x64',True)

from grid import genGrid,ini_conc,calc_grad
from coeff import ini_coeff,update,calc_fx,calc_jacob,ini_fx,ini_jacob,ini_dx,xupdate
import tqdm
import time 
import pandas as pd
from matplotlib import pyplot as plt
from functools import partial







def simulation(K0=1.0,reorg_e=0.5,sigma=10,theta_i=10, theta_v=-10, deltaTheta = 1e-1, z_A=0.0,z_B=-1.0,z_M=1.0,z_N=-1.0,C_sup=100,d_A=1.0,d_B=1.0,d_M=1.0,d_N=1.0,Re=10000,cycles=1,number_of_interations = 5,saving_directory='./Data',save_voltammogram=False):

    """
    K0: Standard Elecrtochemical Rate Constants 
    reorg_e: Dimensionless reorganization energy 
    sigma: dimensionless scan rates 
    theta_i and theta_v: the start and reverse potential of voltammetry
    deltaTheta: potential step size 
    z_A, z_B, z_M, z_N, the charge of species A, B, M and N. M and N are ions of the supporting electrolyte 
    C_sup: The support ratio
    d_A, d_B, d_M, d_N: the dimensionless diffusion coefficient 
    Re: A dimensionless constant related to the Debye length. An important parameter in Possion equation. 
    cycles: Cycles of voltammetric scan 
    number_of_iterations: maximum number of Newton Raphson iterations   
    """


    C_A_bulk = 1.0
    C_B_bulk = 0.0

    if z_A >= 0: 
        C_M_bulk = C_sup
        C_N_bulk = C_sup + z_A
    elif z_A<0:
        C_M_bulk = C_sup - z_A
        C_N_bulk = C_sup  
    Phi_ini = 0.0
    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)

    print(C_A_bulk,C_B_bulk,C_M_bulk,C_N_bulk,Phi_ini)


    deltaX = 1e-7
    deltaTheta = 1e-1
    expanding_grid_factor = 1.1
    SimulationSpaceMultiple = 6.0




    #nTimeSteps_cycles = int(2*jnp.fabs(theta_v-theta_i)/deltaTheta)*cycles
    nTimeSteps = jnp.round(2*jnp.fabs(theta_v-theta_i)/deltaTheta)
    deltaT = deltaTheta/sigma
    maxT = 2.0*jnp.abs(theta_v-theta_i)/sigma * cycles
    maxX = SimulationSpaceMultiple*jnp.sqrt(maxT)
    Esteps = jnp.arange(nTimeSteps)
    E = jnp.where(Esteps<nTimeSteps/2.0,theta_i-deltaTheta*Esteps,theta_v+deltaTheta*(Esteps-nTimeSteps/2.0))
    E = jnp.tile(E,cycles)

    fluxes = jnp.zeros_like(E)

    XGrid, n = genGrid(deltaX=deltaX,maxX=maxX,expanding_grid_factor=expanding_grid_factor)

    conc,concA,concB,concM,concN,concPhi  = ini_conc(n,C_A_bulk=C_A_bulk,C_B_bulk=C_B_bulk,C_M_bulk=C_M_bulk,C_N_bulk=C_N_bulk,Phi_ini=Phi_ini)

    aA,bA,cA,dA, aB,bB,cB,dB, aM,bM,cM,dM, aN,bN,cN,dN, aPhi,bPhi,cPhi,dPhi = ini_coeff(n=n,XGrid=XGrid,deltaX=deltaX,deltaT=deltaT,maxX=maxX,K0=K0,reorg_e=reorg_e,Re=Re,d_A=d_A,d_B=d_B,d_M=d_M,d_N=d_N,z_A=z_A,z_B=z_B,z_M=z_M,z_N=z_N)


    fx = ini_fx(n)
    J = ini_jacob(n)
    dx = ini_dx(n)



    for index in tqdm.tqdm(range(0,len(E))):
        Theta = E[index]

        d = update(x=conc,C_A_bulk=C_A_bulk,C_B_bulk=C_B_bulk,C_M_bulk=C_M_bulk,C_N_bulk=C_N_bulk,Phi_ini=Phi_ini)

        for ii in range(number_of_interations):
            time1 = time.time()
            fx = calc_fx(K0=K0,reorg_e=reorg_e,x=conc,d=d,deltaX=deltaX,deltaT=deltaT,Theta=Theta,n=n,Re=Re,d_A=d_A,d_B=d_B,d_M=d_M,d_N=d_N,z_A=z_A,z_B=z_B,z_M=z_M,z_N=z_N,aA=aA,bA=bA,cA=cA,dA=dA,aB=aB,bB=bB,cB=cB,dB=dB,aM=aM,bM=bM,cM=cM,dM=dM,aN=aN,bN=bN,cN=cN,dN=dN,aPhi=aPhi,bPhi=bPhi,cPhi=cPhi,dPhi=dPhi,fx=fx)

            J = calc_jacob(K0=K0,reorg_e=reorg_e,x=conc,d=d,deltaX=deltaX,deltaT=deltaT,Theta=Theta,n=n,Re=Re,d_A=d_A,d_B=d_B,d_M=d_M,d_N=d_N,z_A=z_A,z_B=z_B,z_M=z_M,z_N=z_N,aA=aA,bA=bA,cA=cA,dA=dA,aB=aB,bB=bB,cB=cB,dB=dB,aM=aM,bM=bM,cM=cM,dM=dM,aN=aN,bN=bN,cN=cN,dN=dN,aPhi=aPhi,bPhi=bPhi,cPhi=cPhi,dPhi=dPhi,J=J)

            
            dx = linag.solve(J,fx)


            conc = xupdate(conc,dx)
            time2 = time.time()
            #print(f'{time2-time1:.2f} s')
            if np.mean(jnp.absolute(dx)) < 1e-12:
                print(f'Exit: Precision satisfied!\nExit at iteration {ii}')
                break



        flux = calc_grad(conc,deltaX)
        #jax.debug.print('{flux}',flux=flux)
        

        fluxes = fluxes.at[index].set(flux)


    if save_voltammogram:
    
        df = pd.DataFrame({'Potential':E,'Flux':fluxes})
        df.to_csv(f'{saving_directory}/logK0={np.log10(K0):.2E} reorg={reorg_e:.2E} sigma={sigma:.2E} C_sup={C_sup:.2E}.csv',index=False)

    return fluxes


if __name__ == "__main__":

    simulation(K0=10**0.5,reorg_e=1.21,sigma=10,C_sup=10,save_voltammogram=True)
    simulation(K0=10**0.5,reorg_e=0.95,sigma=10,C_sup=10,save_voltammogram=True)
