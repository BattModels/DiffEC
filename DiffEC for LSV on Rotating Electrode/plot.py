from matplotlib import pyplot as plt 
import pandas as pd
import numpy as np 
from helper import interp1DExperiment


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
    



scan_rate = 2e-3 # V/s 
freq = 2500 #rpm

df = pd.read_csv('KoperData.csv')

df.iloc[:,1] = df.iloc[:,1]* 10 # Convert mA/cm^2 to A/m^2
df = df.iloc[::-1]

fig,ax = plt.subplots()

ax.plot(df.iloc[:,0],df.iloc[:,1])
ax.set_xlabel('E vs. RHE (V)',fontsize='large',fontweight='bold')
ax.set_ylabel('$j$, $A/m^2$',fontsize='large',fontweight='bold')
fig.savefig('Koper Experiment.png',dpi=250,bbox_inches='tight')



Potential = to_dimensionless_potential(df.iloc[:,0],)
Flux = to_dimensionless_flux(df.iloc[:,1])


theta_i = 0.755379149453578
theta_v= -2.188620850546422
Koper_Theta, Koper_Flux = interp1DExperiment(Potential,Flux,deltaTheta=2e-3,theta_i=theta_i,theta_v=theta_v)

df = pd.DataFrame({'Potential':Koper_Theta,"Flux":Koper_Flux})
df.to_csv('KoperExperimentDimensionless.csv',index=False,)


fig,ax = plt.subplots(figsize=(8,4.5))
#ax.plot(Koper_Theta,Koper_Flux,color='k',lw=2.5,alpha=0.4,label='Koper Data')
ax.plot(Koper_Theta,Koper_Flux,lw=2.5,color='k',label='LSV for Diff. EC.',ls='--',alpha=0.8)


ax.set_xlabel(r'Dimensionless Potential, $\theta$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'Dimensionless Flux, $J$',fontsize='large',fontweight='bold')

sec_xax = ax.secondary_xaxis(-0.2, functions=(to_dimensional_potential,to_dimensionless_potential))
sec_xax.set_xlabel(r'Overpotential $\eta$, V',fontsize='large',fontweight='bold')

sec_yax  =ax.secondary_yaxis(-0.19,functions=(to_dimensional_flux,to_dimensionless_flux))
sec_yax.set_ylabel(r'Flux j, $A/m^2$',fontsize='large',fontweight='bold')
ax.legend()
fig.savefig('Dimensionless Koper Experiment.png',dpi=250,bbox_inches='tight')

