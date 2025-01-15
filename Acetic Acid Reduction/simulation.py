import numpy as np 
import pandas as pd
import os
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import pandas as pd
import math
from grid import genGrid,ini_conc,calc_grad
from coeff import ini_coeff,Allcalc_abc,ini_dx,ini_fx,ini_jacob,calc_fx,update,calc_jacob,xupdate
import jax.scipy.linalg as linag
import tqdm
from matplotlib import pyplot as plt

cRef=1.0 # reference concentration, 1M
P0 = 1.0 # reference pressure, 1 bar
dElectrode = 2.0*5e-6 / math.pi #radius of electrode corresponding to hemispherical electrode
KH_2 = 1292  # Henry law constant for H2
E0f = -0.2415-0.0994 # The formal potential of H+/H2 couple relative to SCE

Kappa = P0/(KH_2*cRef) 


DA = 1.29e-9 # diffusion coefficient of acetic acid, m^2/s , in aqueous solution
DB = 9.311e-9 # diffusion coefficient of H+ ion in aqueous
DY = 1.089e-9 # diffusion coefficient of acetate in aqueous
DZ = 5.11e-9 # diffusion coefficient of hydrogen in aqueous
F_const = 96485 #Faraday constant, 



def simulation(dimKf,dimKeq,cTstar,alpha=1.0,k0=5e-3,Theta=-25.667) :
    """
    dimKf: The forward reaction rate constant, s^-1
    dimKeq = dimKf/dimkb, the equilibrium constant, the unit is M
    cTstar: The initial concentration of acetic acid put into solution,M
    k0:standard electrochemical rate constant,m/s 
    alpha: cathodic transfer coefficient
    Theta: The applied dimensionless potential 
    """

    #Initial spatial step size 
    deltaX = 1e-5
    #Initial time step size 
    deltaT = 4.719462621177882e-05
    
    expanding_grid_factor = 1.10
    expanding_time_factor = 1.06
    
    dimKb = dimKf/ dimKeq

    Kf = dimKf*dElectrode*dElectrode/DB
    Kb = dimKb*cRef*dElectrode*dElectrode / DB

    K0 = k0*dElectrode / DB

    #Get the bulk concentration of Acetic acid after equilibrium
    cAstar = cTstar-(-1.0 + jnp.sqrt(1.0+4.0*cTstar/dimKeq))/(2.0/dimKeq)

    C_A_bulk = cAstar/cRef # Dimensionless bulk concentration of acetic acid after equilibrium 
    C_B_bulk = jnp.sqrt(cAstar*dimKeq)/cRef # Dimensionless bulk concentration of H+ after equilibrium 
    C_Z_bulk = jnp.sqrt(cAstar*dimKeq)/cRef #Dimensionless bulk concentration of acetate after equilibrium
    C_Y_bulk = 0.0 #Dimensionless bulk concentration of hydrogen 






    # dimensionless diffusion coefficients of every species
    d_A =DA/DB
    d_B =DB/DB   # diffusion coefficient of H+
    d_Y =DY/DB  # diffusion coeffficient of acetate+
    d_Z =DZ/DB    # diffusion coefficient of H_2 


    # the maximum number of iterations for Newton method
    number_of_iteration = 10

    maxT = 1e5#73516.70926283435
    maxX = 4.0*jnp.sqrt(maxT)#6.0*jnp.sqrt(maxT)


    n,XGrid,nt,TGrid = genGrid(deltaX=deltaX,maxX=maxX,expanding_grid_factor=expanding_grid_factor,deltaT=deltaT,maxT=maxT,expanding_time_factor=expanding_time_factor)

    dx = ini_dx(n)
    fx = ini_fx(n)
    J = ini_jacob(n)


    conc,concA,concB,concY,concZ = ini_conc(n,C_A_bulk=C_A_bulk,C_B_bulk=C_B_bulk,C_Y_bulk=C_Y_bulk,C_Z_bulk=C_Z_bulk)
    aA,bA,cA,aB,bB,cB,aY,bY,cY,aZ,bZ,cZ = ini_coeff(n,XGrid=XGrid,deltaX=deltaX,TGrid=TGrid,deltaT=deltaT,maxX=maxX,maxT=maxT,d_A=d_A,d_B=d_B,d_Y=d_Y,d_Z=d_Z)

    fluxes = jnp.zeros(nt-1)
    

    for i in range(nt-1):
        deltaT = TGrid[i+1]-TGrid[i]
        d = update(conc,C_A_bulk,C_B_bulk,C_Y_bulk,C_Z_bulk)
        aA,bA,cA,aB,bB,cB,aY,bY,cY,aZ,bZ,cZ = Allcalc_abc(n,XGrid,deltaT=deltaT,d_A=d_A,d_B=d_B,d_Y=d_Y,d_Z=d_Z,aA=aA,bA=bA,cA=cA,aB=aB,bB=bB,cB=cB,aY=aY,bY=bY,cY=cY,aZ=aZ,bZ=bZ,cZ=cZ)

        for ii in range(number_of_iteration):
            fx = calc_fx(x=conc,d=d,deltaX=deltaX,deltaT=deltaT,alpha=alpha,K0=K0,Theta=Theta,n=n,Kappa=Kappa,Kf=Kf,Kb=Kb,d_A=d_A,d_B=d_B,d_Y=d_Y,d_Z=d_Z,aA=aA,bA=bA,cA=cA,aB=aB,bB=bB,cB=cB,aY=aY,bY=bY,cY=cY,aZ=aZ,bZ=bZ,cZ=cZ,fx=fx)

            J = calc_jacob(x=conc,d=d,deltaX=deltaX,deltaT=deltaT,alpha=alpha,K0=K0,Theta=Theta,n=n,Kappa=Kappa,Kf=Kf,Kb=Kb,d_A=d_A,d_B=d_B,d_Y=d_Y,d_Z=d_Z,aA=aA,bA=bA,cA=cA,aB=aB,bB=bB,cB=cB,aY=aY,bY=bY,cY=cY,aZ=aZ,bZ=bZ,cZ=cZ,J=J)

            dx = linag.solve(J,fx)



            conc = xupdate(conc,dx)

            if np.mean(jnp.absolute(dx)) < 1e-12 and i>3:
                #print(f'Exit: Precision satisfied!\nExit at iteration {ii}')
                break

        flux = calc_grad(conc,deltaX)

        fluxes = fluxes.at[i].set(flux)


    
    currents = 2*np.pi*dElectrode*F_const*DB*1000*fluxes # The unit of current is A. The SI unit of concentration is mol/m^3. mol/L is multiplied by 1000 to get mol/m^3

    jax.debug.print('Bulk concentration is {cTstar}M',cTstar=cTstar)
    jax.debug.print('Steady State current is {current}A',current=currents[-1])
    jax.debug.print('Dimensionless Steady Stae Flux is {flux}\n',flux=fluxes[-1])

    

    return currents



if __name__ == "__main__":




    simulation(dimKeq=10**(-4.37),dimKf=10**5.74,cTstar=0.01)
    simulation(dimKeq=10**(-4.37),dimKf=10**5.74,cTstar=0.02)
    simulation(dimKeq=10**(-4.37),dimKf=10**5.74,cTstar=0.04)
    simulation(dimKeq=10**(-4.37),dimKf=10**5.74,cTstar=0.1)


    "The simulated steady state currents at the four bulk concentratiuons are 32.7 nA, 59.0 nA, 107 nA and 239 nA" 

    "The experimental currents are 30.5, 55.2, 102 and 238 nA."



