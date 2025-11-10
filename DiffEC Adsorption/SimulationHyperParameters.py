import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
cycles = 1 
deltaX = 1e-6 # The initial space step
deltaTheta = 1e-1 # The expanding grid factor 
expanding_grid_factor = 1.10  
SimulationSpaceMultiple = 6.0 



epochs=600
logK0_initial  = -4.0 #The initial guess of dimensionless K0 in log10scale
alpha_initial  = 0.5  # The initial guess of cathodic transfer coefficient 
beta_initial   = 0.5  #The initial guess of anodic transfer coefficient 

logK0_ads_initial = 0.0
alpha_ads_initial = 0.5
beta_ads_initial  = 0.5

logK_A_ads_initial = -1.0
logK_B_ads_initial = 1.0 
logK_A_des_initial = 0.0 
logK_B_des_initial = 0.0



logK0_initial_range  = 1.5 
alpha_initial_range  = 0.2   
beta_initial_range   = 0.2  

logK0_ads_initial_range = 1.5
alpha_ads_initial_range = 0.2
beta_ads_initial_range  = 0.2

logK_A_ads_initial_range = 0.5
logK_B_ads_initial_range = 0.5 
logK_A_des_initial_range = 0.5 
logK_B_des_initial_range = 0.5


lr=5e-2 #Learning rate 
optimizer_name = "adam"

number_of_iteration = 6

Theta_i = 25.0
Theta_v = -25.0

sigmas = [100,40,4.0,0.4]
voltammograms_ground_truth= [
    "./Ground Truth/sigma=1.00E+02 K0=1.00E-03 alpha=4.00E-01 beta=6.00E-01 K0_ads=5.00E-01 alpha_ads=4.50E-01 beta_ads=5.50E-01 K_A_ads=4.50E+00 K_A_des=1.00E+00 K_B_ads=1.00E+00 K_B_des=1.00E+00.csv",
    "./Ground Truth/sigma=4.00E+01 K0=1.00E-03 alpha=4.00E-01 beta=6.00E-01 K0_ads=5.00E-01 alpha_ads=4.50E-01 beta_ads=5.50E-01 K_A_ads=4.50E+00 K_A_des=1.00E+00 K_B_ads=1.00E+00 K_B_des=1.00E+00.csv",
    "./Ground Truth/sigma=4.00E+00 K0=1.00E-03 alpha=4.00E-01 beta=6.00E-01 K0_ads=5.00E-01 alpha_ads=4.50E-01 beta_ads=5.50E-01 K_A_ads=4.50E+00 K_A_des=1.00E+00 K_B_ads=1.00E+00 K_B_des=1.00E+00.csv",
    "./Ground Truth/sigma=4.00E-01 K0=1.00E-03 alpha=4.00E-01 beta=6.00E-01 K0_ads=5.00E-01 alpha_ads=4.50E-01 beta_ads=5.50E-01 K_A_ads=4.50E+00 K_A_des=1.00E+00 K_B_ads=1.00E+00 K_B_des=1.00E+00.csv"
]

if __name__ == "__main__":
    fig,ax = plt.subplots(figsize=(8,4.5))
    for index,sigma in enumerate(sigmas):
        df = pd.read_csv(voltammograms_ground_truth[index])
        ax.plot(df.iloc[:,0],df.iloc[:,1]/np.sqrt(sigma),label=f'$\\sigma=${sigma:.2E}')

    ax.set_xlabel(r'Potential, $\theta$',fontsize='large',fontweight='bold')
    ax.set_ylabel(r'Normalized Flux, $J/\sqrt{\sigma}$',fontsize='large',fontweight='bold')
    ax.legend()

    fig.savefig('./Ground Truth/Voltammograms.png',dpi=250,bbox_inches='tight')

    