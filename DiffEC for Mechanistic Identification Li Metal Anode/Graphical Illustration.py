import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd
import jax.numpy as jnp 
from matplotlib.cm import tab10

linewidth = 3
fontsize = 15

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs




fig, ax = plt.subplots(figsize=(8,4.5))

df = pd.read_csv('Boyle Figure 4a.csv',skiprows=1)


system_names = [r"10% FEC",r'EC DEC',r'DEC',r'PC']

cut_offs = [8,13,6,6]
colors = tab10(np.linspace(0.05,0.95,len(system_names)))

for index, system_name in enumerate(system_names):
    expt_eta = jnp.array(df.iloc[:,index*2].dropna())
    expt_current_density = jnp.array(df.iloc[:,index*2+1].dropna())

    ax.plot(expt_eta[:cut_offs[index]],expt_current_density[:cut_offs[index]],color=tuple(colors[index]),label=f'{system_name}',marker='o',lw=3,markersize=8,alpha=0.9)
    ax.plot(expt_eta[cut_offs[index]:],expt_current_density[cut_offs[index]:],color=tuple(colors[index]),marker='o',lw=3,markersize=8,alpha=0.9)

ax.set_xlabel(r'Overpotential, V',fontsize='xx-large',fontweight='bold')
ax.set_ylabel(r'$\mathrm{mA/cm^2}$',fontsize='xx-large',fontweight='bold')

ax.legend()
ax.set_yscale('log')

plt.tight_layout()

fig.savefig('Electrolyte Overpotential.png',dpi=250,bbox_inches='tight')




