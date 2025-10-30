from matplotlib import pyplot as plt
from simulation import simulation
import pandas as pd
import numpy as np 
linewidth = 3
fontsize = 12
figsize = [8,4.5]

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwarg


fig, ax = plt.subplots(figsize=(8,4.5))

file_name_1 = "FD_Data\sigma=4.00E+01.csv"

df = pd.read_csv(file_name_1)

ax.plot(df.iloc[:,0],df.iloc[:,1],label='Simulation',lw=3,alpha=0.8)
thetas = np.linspace(-20,20,num=1000)

analytical_flux = 40*np.exp(-thetas)/((1.0+np.exp(-thetas))**2)

ax.plot(thetas,analytical_flux,label="Analytical equation",color='k',alpha=0.8,ls='--')
ax.plot(thetas,-analytical_flux,color='k',alpha=0.8,ls='--')

ax.axhline(y=-0.25*40,label='Analytical $J_p$',alpha=0.8,ls='--',color='b')

ax.set_xlabel(r'Potential,$\theta$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'Flux, $J$',fontsize='large',fontweight='bold')

ax.legend()
plt.tight_layout()
fig.savefig('Fully adsorptive.png',dpi=250,bbox_inches='tight')