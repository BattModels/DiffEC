from simulation import simulation
from matplotlib import pyplot as plt
import pandas as pd
from SimulationHyperParameters import Theta_i,Theta_v
import numpy as np 
sigmas = [100,40,4.0,0.4]
file_names = []
for sigma in sigmas:
    #Reactant adsorb more strongly, postwave
    file_name = simulation(sigma=sigma,K0=1e-3,K0_ads=5e-1,alpha=0.4,beta=0.6,alpha_ads=0.45,beta_ads=0.55,K_A_ads=4.5,K_B_ads=1.0,theta_i=Theta_i,theta_v=Theta_v,saving_directory="Ground Truth",save_voltammogram=True)
    file_names.append(file_name)







