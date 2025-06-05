from main import calcSigma,simulation
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from helper import interp1DExperiment

#experimental_parameters
nu = 1e-6 #m^2/s 
D = 9.311e-9 #m^2/s 
c_bulk = 1e3 #mol/m^3 
rot_freq  = 2500/60 # rpm 
omega = rot_freq*np.pi*2 #Rad/second
scan_rate = 2e-3 #V/s 
sigma = calcSigma(omega=omega,nu=nu,D=D,scan_rate=scan_rate)


df_koper_experiment = pd.read_csv("KoperExperimentDimensionless.csv")
theta_i = df_koper_experiment.iloc[0,0]
theta_v= df_koper_experiment.iloc[-1,0]










for K0 in [1e-2,8e-3,6e-3,4e-3,2e-3,1e-3]:
    for alpha in [0.5]:
        for beta in [0.5]:
            saving_directory = f"ExperimentalCV"
            if not os.path.exists(saving_directory):
                os.mkdir(saving_directory)
            CV_location  = f'{saving_directory}/K0={K0:.2E} alpha={alpha:.2E} beta={beta:.2E}'

            simulation(sigma=sigma,K0=K0,alpha=alpha,beta=beta,theta_i=theta_i,theta_v=theta_v,CV_location = CV_location,save_voltammogram=True)

            fig, ax = plt.subplots()
            df = pd.read_csv(f'{CV_location}.csv')

            ax.plot(df.iloc[:,0],df.iloc[:,1],label='Simulation')
            ax.plot(df_koper_experiment.iloc[:,0],df_koper_experiment.iloc[:,1],label='Exp')
            ax.legend()

            fig.savefig(f'{CV_location}.png')






