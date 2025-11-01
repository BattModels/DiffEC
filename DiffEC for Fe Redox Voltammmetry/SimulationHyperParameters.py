cycles = 1 
deltaX = 2e-6 # The initial space step
deltaTheta = 2e-1 # The expanding grid factor 
expanding_grid_factor = 1.1  
SimulationSpaceMultiple = 6.0 


scan_rates = [0.01,0.02,0.05,0.1,0.2]
exp_dimensionless_files = [
    r"ExpData/Exp Dimensionless sigma=2.8137E+02 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv",
    r"ExpData/Exp Dimensionless sigma=5.6273E+02 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv",
    r"ExpData/Exp Dimensionless sigma=1.4068E+03 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv",
    r"ExpData/Exp Dimensionless sigma=2.8137E+03 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv",
    r"ExpData/Exp Dimensionless sigma=5.6273E+03 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv"
]


epochs=400
logK0_initial  = 1.0 #The initial guess of dimensionless K0 in logscale
alpha_initial = 0.4  # The initial guess of cathodic transfer coefficient 
beta_initial = 0.4  #The initial guess of anodic transfer coefficient 
dA_initial = 0.4  #The initial guess of diffusion coefficient

logK0_initial_range = 0.5 
alpha_initial_range = 0.2
beta_initial_range = 0.2
dA_initial_range = 0.2 

lr=1e-3 #Learning rate 
optimizer_name = "sgd"

Theta_i = 14.268850309900179
Theta_v = -12.991507814909113