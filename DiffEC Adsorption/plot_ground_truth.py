from matplotlib import pyplot as plt
from SimulationHyperParameters import voltammograms_ground_truth,sigmas
import pandas as pd
import numpy as np
from matplotlib.cm import viridis

fontsize = 12

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize, }
plt.rc('font', **font)  # pass in the font dict as kwargs



fig, ax = plt.subplots(figsize=(8,4.5))
fluxes_avg = []

colors = viridis(np.linspace(0,1,len(sigmas)))
for index,sigma in enumerate(sigmas):
    
    df = pd.read_csv(voltammograms_ground_truth[index])


    ax.plot(df.iloc[:,0],df.iloc[:,1]/np.sqrt(sigma),lw=2.5,label=f'$\sigma={sigma:.2E}$',color=tuple(colors[index]))

    fluxes_avg.append(np.average(np.abs(df.iloc[:,1])))


ax.set_xlabel(r'Potential, $\theta$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'Normalized Flux, $J/\sqrt{\sigma}$',fontsize='large',fontweight='bold')

ax.legend()
fig.savefig('Ground_Truth_Voltammograms.png',bbox_inches='tight')
