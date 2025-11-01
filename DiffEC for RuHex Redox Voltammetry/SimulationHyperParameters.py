cycles = 1 
deltaX = 1e-6 # The initial space step
deltaTheta = 5e-2 # The expanding grid factor 
expanding_grid_factor = 1.1  
SimulationSpaceMultiple = 6.0 


scan_rates = [0.025,0.05,0.1,0.2]
sigmas = [243.39605468579722, 486.79210937159445, 973.5842187431889,1947.1684374863778 ]
exp_dimensionless_files = [
    r"ExpData/Exp Dimensionless sigma=2.4340E+02 theta_i=1.0515E+01 theta_v=-8.9570E+00 dA=8.43E-01 dB=1.19E+00 ERef=-1.700E-01.csv",
    r"ExpData/Exp Dimensionless sigma=4.8679E+02 theta_i=1.0515E+01 theta_v=-8.9570E+00 dA=8.43E-01 dB=1.19E+00 ERef=-1.700E-01.csv",
    r"ExpData/Exp Dimensionless sigma=9.7358E+02 theta_i=1.0515E+01 theta_v=-8.9570E+00 dA=8.43E-01 dB=1.19E+00 ERef=-1.700E-01.csv",
    r"ExpData/Exp Dimensionless sigma=1.9472E+03 theta_i=1.0515E+01 theta_v=-8.9570E+00 dA=8.43E-01 dB=1.19E+00 ERef=-1.700E-01.csv",
]




epochs=400
theta_corr_initial = 2.0
theta_corr_initial_range = 4.0

dA_initial = 1.0  #The initial guess of diffusion coefficient
dA_initial_range = 0.2 

lr=1e-3 #Learning rate 
optimizer_name = "sgd"


Theta_i = 10.51470956242644
Theta_v = -8.956974812437338