import math
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from jax.scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd

def calcW(Y):
    # This function converts Y (which is after Hale transformation) to W. 
    def dWdY(W,Y):
        dWdY = np.sqrt(1.65894)*np.exp(1.0/3.0*W**3)
        return dWdY

    Y_ = np.linspace(0,0.99999,num=100000,endpoint=False)
    W0 = 0.0

    W = odeint(dWdY,W0,Y_).reshape(-1)

    func_1D = interp1d(Y_,W,fill_value='extrapolate')

    return func_1D(Y)



def GetInterpW():
    def dWdY(W,Y):
        dWdY = np.sqrt(1.65894)*np.exp(1.0/3.0*W**3)
        return dWdY

    Y_ = np.linspace(0,0.9999,num=100000,endpoint=False)
    W0 = 0.0

    W = odeint(dWdY,W0,Y_).reshape(-1,1)

    interp_1D = RegularGridInterpolator(Y_.reshape(1,-1),W)
    return interp_1D









if __name__ == '__main__': 

	
	Y = np.linspace(0,0.9999,num=1000).reshape(-1,1)

	InterpW = GetInterpW()
	W = InterpW(Y)

	#Y = 0.5

	plt.scatter(Y,W)
	plt.xlabel('Y')
	plt.ylabel('W')
	plt.show()
	

