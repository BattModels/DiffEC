import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as liang
import os 
import jax

jax.config.update('jax_enable_x64',True)

import tqdm
import pandas as pd
from SimulationHyperParameters import cycles, deltaX, deltaTheta, expanding_grid_factor, SimulationSpaceMultiple, number_of_iteration
from grid import genGrid, ini_conc,update_d,calc_grad
from coeff import ini_coeff,Allcalc_abc_linear,Allcacl_abc_radial,calc_jacob,calc_fx,xupdate

def simulation(sigma=1.0,K0=1.0,alpha=0.5,beta=None,K0_ads=1.0,alpha_ads=0.5,beta_ads=None,K_A_ads=1.0,K_A_des=1.0,K_B_ads=1.0,K_B_des=1.0,C_A_bulk=1.0,C_B_bulk=0.0,saturation_number=1.0,kinetics='BV',mode='linear',dA=1.0,dB=1.0,reorg_e=1.0,theta_i=20,theta_v=-20,theta_f_ads=None,saving_directory='./FD_Data',save_voltammogram=False):
    if not os.path.exists(saving_directory):
         os.mkdir(saving_directory)

    if beta is None:
        beta = 1.0-alpha

    if beta_ads is None:
        beta_ads = 1.0 - alpha_ads


    #Now calculate the initial surface coverage of A and B
    K_A_eq = K_A_ads/K_A_des
    K_B_eq = K_B_ads/K_B_des
    zeta_A_ini = K_A_eq * C_A_bulk / (1.0 + K_A_eq*C_A_bulk + K_B_eq*C_B_bulk)
    zeta_B_ini = K_B_eq * C_B_bulk / (1.0 + K_A_eq*C_A_bulk + K_B_eq*C_B_bulk)

    if theta_f_ads is None:
        theta_f_ads = -jnp.log(K_A_eq/K_B_eq)

    deltaT = deltaTheta/sigma
    maxT = cycles*2.0*abs(theta_v-theta_i)/sigma

    #simulation steps
    nTimeSteps = int(2*jnp.fabs(theta_v-theta_i)/deltaTheta)+1
    Esteps = jnp.arange(nTimeSteps)
    E = jnp.where(Esteps<nTimeSteps/2.0,theta_i-deltaTheta*Esteps,theta_v+deltaTheta*(Esteps-nTimeSteps/2.0))
    E = jnp.tile(E,cycles)
    Fluxes = jnp.zeros_like(E)



    Xi = 0.0 
    if mode == "linear":
        maxX = SimulationSpaceMultiple * np.sqrt(maxT)
        Xi = 0.0
    elif mode =='radial':
        maxX = SimulationSpaceMultiple * np.sqrt(maxT) + 1.0
        Xi = 1.0
    else:
        raise ValueError
    
    X_grid,n = genGrid(Xi=Xi,deltaX=deltaX,maxX=maxX,expanding_grid_factor=expanding_grid_factor)

    conc,conc_d = ini_conc(n=n,C_A_bulk=C_A_bulk,C_B_bulk=C_B_bulk,zeta_A_ini=zeta_A_ini,zeta_B_ini=zeta_B_ini)
    J,fx,dx,aA,bA,cA,aB,bB,cB = ini_coeff(n=n)


    if mode == "linear":
        aA,bA,cA,aB,bB,cB = Allcalc_abc_linear(n=n,X_grid=X_grid,deltaT=deltaT,aA=aA,bA=bA,cA=cA,dA=dA,aB=aB,bB=bB,cB=cB,dB=dB)
    elif mode == "radial":
        aA,bA,cA,aB,bB,cB = Allcacl_abc_radial(n=n,X_grid=X_grid,deltaT=deltaT,aA=aA,bA=bA,cA=cA,dA=dA,aB=aB,bB=bB,cB=cB,dB=dB)

    else:
        raise ValueError



    for index in tqdm.tqdm(range(0,len(E))):
        
        Theta = E[index]

        conc_d = update_d(conc=conc,conc_d=conc_d,C_A_bulk=C_A_bulk,C_B_bulk=C_B_bulk)
        
        for ii in range(number_of_iteration):
            J = calc_jacob(J=J,Theta=Theta,n=n,X_grid=X_grid,conc=conc,deltaT=deltaT,K0=K0,alpha=alpha,beta=beta,K0_ads=K0_ads,alpha_ads=alpha_ads,beta_ads=beta_ads,K_A_ads=K_A_ads,K_A_des=K_A_des,K_B_ads=K_B_ads,K_B_des=K_B_des,C_A_bulk=C_A_bulk,C_B_bulk=C_B_bulk,saturation_number=saturation_number,kinetics=kinetics,mode=mode,dA=dA,dB=dB,reorg_e=reorg_e,theta_f_ads=theta_f_ads,aA=aA,bA=bA,cA=cA,aB=aB,bB=bB,cB=cB)
            fx = calc_fx(fx=fx,Theta=Theta,n=n,X_grid=X_grid,conc=conc,conc_d=conc_d,deltaT=deltaT,K0=K0,alpha=alpha,beta=beta,K0_ads=K0_ads,alpha_ads=alpha_ads,beta_ads=beta_ads,K_A_ads=K_A_ads,K_A_des=K_A_des,K_B_ads=K_B_ads,K_B_des=K_B_des,C_A_bulk=C_A_bulk,C_B_bulk=C_B_bulk,saturation_number=saturation_number,kinetics=kinetics,mode=mode,dA=dA,dB=dB,reorg_e=reorg_e,theta_f_ads=theta_f_ads,aA=aA,bA=bA,cA=cA,aB=aB,bB=bB,cB=cB)

            dx = liang.solve(J,fx)

            conc = xupdate(conc,dx)

            if index > 1 and jnp.mean(jnp.absolute(dx)) < 1e-12:
                #print(f'Exit: Precision satisfied!\nExit at iteration {ii}')
                break

        flux = calc_grad(conc=conc,conc_d=conc_d,X_grid=X_grid,deltaT=deltaT,saturation_number=saturation_number)

        Fluxes = Fluxes.at[index].set(flux)



    if save_voltammogram:
        if not os.path.exists(saving_directory):
            os.mkdir(saving_directory)
        df = pd.DataFrame({'Potential':E,'Flux':Fluxes})
        CV_location = f'{saving_directory}/sigma={sigma:.2E} K0={K0:.2E} alpha={alpha:.2E} beta={beta:.2E} K0_ads={K0_ads:.2E} alpha_ads={alpha_ads:.2E} beta_ads={beta_ads:.2E} K_A_ads={K_A_ads:.2E} K_A_des={K_A_des:.2E} K_B_ads={K_B_ads:.2E} K_B_des={K_B_des:.2E}'
        df.to_csv(f'{CV_location}.csv',index=False)

    return E,Fluxes




if __name__ == "__main__":
    simulation(save_voltammogram=True)