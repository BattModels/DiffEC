import numpy as np 
import json
import jax.numpy as jnp
import jax.scipy.linalg as linalg
import pandas as pd
from matplotlib.cm import tab10,viridis,Blues
import matplotlib.ticker as mtick
from main import calcSigma,simulation

from matplotlib import pyplot as plt

linewidth = 3
fontsize = 12
figsize = [8,4.5]

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwarg

def to_dimensionless_flux(j,freq=2500/60,D=9.311e-9,nu=1e-6,c_bulk=1e3,F=96485):
    """j:Flux Density, A/m^2
	c^*A: Bulk Concentration of Analyte 
    D:Diffusion Coefficient 
    L: Hydrodynamic Constant
    """
    L = 0.51023 *(2*np.pi*freq)**(1.5)*nu**(-0.5)
    J = j*(np.sqrt(1.65894)/(F*c_bulk*(D**2*L)**(1.0/3.0)))
    return J


def to_dimensionless_potential(E,E0f=0.0,R=8.314,T=298,F=96485):
    theta =  (E-E0f)*(F/(R*T))
    return theta

def to_dimensional_flux(J,freq=2500/60,D=9.311e-9,nu=1e-6,c_bulk=1e3,F=96485):
    L = 0.51023 *(2*np.pi*freq)**(1.5)*nu**(-0.5)
    j = J/(np.sqrt(1.65894)/(F*c_bulk*(D**2*L)**(1.0/3.0)))
    return j

def to_dimensional_potential(theta,E0f=0.0,R=8.314,T=298,F=96485):
    E = theta/(F/(R*T)) + E0f
    return E 
    


with open(r"histories/rel_noise_levl=0.00E+00.json") as f:
    histories = json.load(f)

fig, axs  = plt.subplots(figsize=(8,13.5),nrows=3)
ax = axs[0]
rel_noise_level = 0.0

ax.plot(histories['alpha_guess_history'],alpha=0.7,label=r'$\alpha$')
ax.plot(histories['beta_guess_history'],alpha=0.7,label=r'$\beta$')
ax.legend()
twin1 = ax.twinx()
twin1.plot(histories['loss_history'],color='k',alpha=0.5,ls='--',label='loss')
twin1.set_ylabel('Loss',fontsize='large',fontweight='bold')
twin1.set_yscale('log')
twin1.legend()
ax.set_ylabel(r'$\alpha$ or $\beta$',fontsize='large',fontweight='bold')
ax = axs[1]
ax.plot(histories['log10K0_guess_history'],alpha=0.7,label=r'$log_{10}K_0$')
ax.legend()
twin1 = ax.twinx()
twin1.plot(histories['loss_history'],color='k',alpha=0.5,ls='--',label='loss')
twin1.set_ylabel('Loss',fontsize='large',fontweight='bold')
twin1.set_yscale('log')
ax.set_xlabel('Epoch')
twin1.legend(loc=5)
ax.set_ylabel(r'$log_{10}K_0$',fontsize='large',fontweight='bold')

ax = axs[2]

df_koper_experiment = pd.read_csv("KoperExperimentDimensionless.csv")
theta_i = df_koper_experiment.iloc[0,0]
theta_v= df_koper_experiment.iloc[-1,0]


nu = 1e-6 #m^2/s 
D = 9.311e-9 #m^2/s 
c_bulk = 1e3 #mol/m^3 
rot_freq  = 2500/60 # RMP to Hz
omega = rot_freq*np.pi*2 #Rad/second
scan_rate = 2e-3 #V/s 

L = 0.51023 *(2*np.pi*rot_freq)**(1.5)*nu**(-0.5)
sigma = calcSigma(omega=omega,nu=nu,D=D,scan_rate=scan_rate)
theta_i = 0.755379149453578
theta_v = -2.188620850546423
K0 = np.power(10,histories['log10K0_guess_history'][-1])
alpha = histories['alpha_guess_history'][-1]
beta = histories['beta_guess_history'][-1]
saving_directory = f"ExperimentalCV"
CV_location  = f'{saving_directory}/K0={K0:.2E} alpha={alpha:.2E} beta={beta:.2E}'
k0 = K0/(np.sqrt(1.65894)*(D**2*L)**(-1.0/3.0))
print(f'The extracted alpha is {alpha:.2E} beta is {beta:.2E} K0 is {K0:.2E}, k0 is {k0:.3E}m/s')
simulation(sigma=sigma,K0=K0,alpha=alpha,beta=beta,theta_i=theta_i,theta_v=theta_v,CV_location = CV_location,save_voltammogram=True)

df = pd.read_csv(f'{CV_location}.csv')
ax.plot(df.iloc[:1398,0],df.iloc[:1398,1],label='Simulation')
ax.plot(df_koper_experiment.iloc[:1398,0],df_koper_experiment.iloc[:1398,1],label='Experiment')
ax.legend()


ax.set_xlabel(r'Dimensionless Potential, $\theta$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'Dimensionless Flux, $J$',fontsize='large',fontweight='bold')

sec_xax = ax.secondary_xaxis(-0.2, functions=(to_dimensional_potential,to_dimensionless_potential))
sec_xax.set_xlabel(r'Overpotential $\eta$, V',fontsize='large',fontweight='bold')

sec_yax  =ax.secondary_yaxis(-0.19,functions=(to_dimensional_flux,to_dimensionless_flux))
sec_yax.set_ylabel(r'Flux j, $A/m^2$',fontsize='large',fontweight='bold')
ax.legend()


fig.text(0.05,0.88,'a)',fontsize=20,fontweight='bold')
fig.text(0.05,0.62,'b)',fontsize=20,fontweight='bold')
fig.text(0.05,0.35,'c)',fontsize=20,fontweight='bold')

fig.savefig('History.png',dpi=250,bbox_inches='tight')


