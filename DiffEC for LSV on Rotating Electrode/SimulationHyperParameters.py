import numpy as np 

nu = 1e-6 #m^2/s, kinematic viscosity of water
D = 9.311e-9 #m^2/s 
c_bulk = 1e3 #mol/m^3 
rot_freq  = 2500/60 # RMP to Hz
omega = rot_freq*np.pi*2 #Rad/second
scan_rate = 2e-3 #V/s 

theta_i = 0.755379149453578
theta_v = -2.188620850546423

lr = 5e-2
optimizer_name  = 'adam'

epochs = 1000

alpha_initial = 1.0
log10K0_initial = -4.0
alpha_initial_range = 0.5
log10K0_initial_range  = 1.0



exp_dimensionless_file = r'KoperExperimentDimensionless.csv'