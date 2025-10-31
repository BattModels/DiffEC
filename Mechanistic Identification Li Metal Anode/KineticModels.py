import numpy as np 
import json
import jax
import jax.numpy as jnp 
import optax
import sys
import pandas as pd
from matplotlib import pyplot as plt
from functools import partial
F = 96485 # C/mol 
R = 8.314 # J/(mol * k)
T = 298 #K
df = pd.read_csv("Boyle Figure 4a.csv",skiprows=1)


def BV_current(j0,alpha,eta):
    j_BV = j0*(jnp.exp(-alpha*F/(R*T)*eta) - jnp.exp((1.0-alpha)*F/(R*T)*eta))
    return jnp.abs(j_BV)


def MH_current(j0,reorg_e,eta):
    """
    eta: Overpotential. Volt
    reorg_e: Reorganization energy in eV. 
    """
    alpha = 0.5 + eta/(4*reorg_e)
    j_MH = j0*(jnp.exp(-alpha*F/(R*T)*eta)) - j0*jnp.exp((1.0-alpha)*F/(R*T)*eta)

    return jnp.abs(j_MH)


def MHC_current(j0,reorg_e,eta):
    """
    eta: Overpotential. Volt
    reorg_e: Reorganization energy in eV. 
    """
    theta = eta*(F/(R*T)) #Dimensionless potential 
    Lambda = reorg_e * (F/(R*T)) # Dimensionless reorganization energy 

    def MHIntegrand_Red(theta,Lambda=Lambda,hermgauss_degree=50):
        sample_points,weights = np.polynomial.hermite.hermgauss(hermgauss_degree)
        def integrand(u):
            return 2*jnp.sqrt(Lambda)/(1.0+jnp.exp(-(Lambda*(u*2.0/jnp.sqrt(Lambda)-1.0)-theta)))
        y = integrand(sample_points)
        return jnp.sum(weights*y)
    
    def MHIntegrand_Ox(theta,Lambda=Lambda,hermgauss_degree=50):
        sample_points,weights = np.polynomial.hermite.hermgauss(hermgauss_degree)
        def integrand(u):
            return -2.0*jnp.sqrt(Lambda)/(1.0+jnp.exp(-Lambda*(u*2.0/jnp.sqrt(Lambda)-1.0)-theta))
        y = integrand(sample_points)
        return jnp.sum(weights*y)
    
    #j_MHC = j0*MHIntegrand_Red(theta=theta,Lambda=Lambda) - j0*MHIntegrand_Ox(theta=theta,Lambda=Lambda)
    kred = jax.vmap(MHIntegrand_Red)(theta)/MHIntegrand_Red(0.0)
    kox =  jax.vmap(MHIntegrand_Ox)(theta)/MHIntegrand_Ox(0.0)

    j_MHC = j0*(kred-kox)

    return jnp.abs(j_MHC)

def MHC_current_approx(j0,reorg_e,eta):
    """
    eta: Overpotential. Volt
    reorg_e: Reorganization energy in eV. 
    Journal of Electroanalytical Chemistry, Volume 735, 1 December 2014, Pages 77-83
    """
    theta = eta*(F/(R*T)) #Dimensionless potential 
    Lambda = reorg_e*(F/(R*T)) # Dimensionless reorganization energy
    #j_0_ref = jnp.sqrt(jnp.pi*Lambda)*jax.scipy.special.erfc((Lambda - jnp.sqrt(1.0+jnp.sqrt(Lambda)+0.0**2))/(2*jnp.sqrt(Lambda)))
    j_MHC = j0*jnp.sqrt(jnp.pi*Lambda)*jnp.tanh(theta/2.0)*jax.scipy.special.erfc((Lambda - jnp.sqrt(1.0+jnp.sqrt(Lambda)+theta**2))/(2*jnp.sqrt(Lambda)))

    j_MHC = j_MHC
    
    return jnp.abs(j_MHC)


    



if __name__ == "__main__":
    

    eta_1 = jnp.array(df.iloc[:,6].dropna())
    current_density_1 = jnp.array(df.iloc[:,7].dropna())
    eta_simulated  = eta_1#jnp.linspace(-0.5,0.5)


    fig, ax = plt.subplots(figsize=(8,4.5))


    #simulated_current_density_BV  = BV_current(1.9,0.5,eta_simulated,)
    simulated_current_density_MH = MH_current(1.9,0.33,eta_simulated)
    simulated_current_density_MHC =MHC_current(1.9,0.22,eta_simulated)
    simulated_current_density_MHC_approx = MHC_current_approx(1.9,0.22,eta_simulated)



    ax.plot(eta_1,current_density_1,label='Experiment',marker='o',ls="--",)
    #ax.plot(eta_simulated,simulated_current_density_BV,label='BV')
    ax.plot(eta_simulated,simulated_current_density_MH,label='MH',marker='o')
    ax.plot(eta_simulated,simulated_current_density_MHC,label='MHC',marker='o')
    ax.plot(eta_simulated,simulated_current_density_MHC_approx,label='MHC_approx',marker='o')

    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel(r'$E-E_{eq}, V$',fontsize='large',fontweight='bold')
    ax.set_ylabel(r'$j, mA/cm^2$',fontsize='large',fontweight='bold')

    plt.show()