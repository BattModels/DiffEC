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



# Define Adam optimizer
def adam_update(param, grad,epoch,m_prev,v_prev, learning_rate=1e-3, beta1=0.9, beta2=0.999,epsilon=1e-7):
	"""
	Adam update algorithm is an optimization method used for training machine
	learning models, particularly neural networks.

	Intuition:
	Adam combines the benefits of two other popular optimization algorithms:
	AdaGrad and RMSProp.

	1. AdaGrad adapts the learning rate to parameters, performing larger updates
		for infrequent parameters and smaller updates for frequent ones. However,
		its continuously accumulating squared gradients can lead to an overly
		aggressive and monotonically decreasing learning rate.

	2. RMSProp modifies AdaGrad by using a moving average of squared gradients to
		adapt the learning rate, which resolves the radical diminishing learning
		rates of AdaGrad.

	Adam takes this a step further by:
	- Calculating an exponentially moving average of the gradients (m) to smooth
	out the gradient descent path, addressing the issue of noisy gradients.
	- Computing an exponentially moving average of the squared gradients (v),
	which scales the learning rate inversely proportional to the square root of
	the second moments of the gradients. This helps in adaptive learning rate
	adjustments.
	- Implementing bias corrections to the first (m_hat) and second (v_hat) moment
	estimates to account for their initialization at the origin, leading to more
	accurate updates at the beginning of the training.

	This results in an optimization algorithm that can handle sparse gradients on
	noisy problems, which is efficient for large datasets and high-dimensional
	parameter spaces.

	Note: The function requires initialization or previous values of m_prev,
	v_prev, and epsilon (a small number to prevent division by zero).
	"""

	# Update biased first moment estimate.
	# m is the exponentially moving average of the gradients.
	# beta1 is the decay rate for the first moment.
	m = beta1 * m_prev + (1 - beta1) * grad

	# Update biased second raw moment estimate.
	# v is the exponentially moving average of the squared gradients.
	# beta2 is the decay rate for the second moment.
	v = beta2 * v_prev + (1 - beta2) * grad**2

	# Compute bias-corrected first moment estimate.
	# This corrects the bias in the first moment caused by initialization at origin.
	m_hat = m / (1 - beta1**(epoch + 1))

	# Compute bias-corrected second raw moment estimate.
	# This corrects the bias in the second moment caused by initialization at origin.
	v_hat = v / (1 - beta2**(epoch + 1))

	# Update parameters.
	# Parameters are adjusted based on the learning rate, corrected first moment,
	# and the square root of the corrected second moment.
	# epsilon is a small number to avoid division by zero.
	param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

	# Return the updated parameters, as well as the first and second moment estimates.
	return param, m, v




def load_experimental_flux(file_name,rel_noise_level=0.0,abs_noise_level=0.0 ):
	key = jax.random.PRNGKey(42)
	df = pd.read_csv(file_name)

	flux = jnp.array(df.iloc[:,1].to_numpy())
	flux = flux*(1.0+jax.random.normal(key,shape=flux.shape)*rel_noise_level) + jax.random.normal(key,shape=flux.shape)*abs_noise_level

	return flux



def to_dimensionless_flux(j,freq,D,nu,c_bulk,F=96485):
    """j:Flux Density, A/m^2
	c^*A: Bulk Concentration of Analyte 
    D:Diffusion Coefficient 
    L: Hydrodynamic Constant
    """
    L = 0.51023 *(2*np.pi*freq)**(1.5)*nu**(-0.5)
    J = j*(np.sqrt(1.65894)/(F*c_bulk*(D**2*L)**(1.0/3.0)))
    return J


def to_dimensionless_potential(E,E0f,R=8.314,T=298,F=96485):
    theta =  (E-E0f)*(F/(R*T))
    return theta

def to_dimensional_flux(J,freq,D,nu,c_bulk,F=96485):
    L = 0.51023 *(2*np.pi*freq)**(1.5)*nu**(-0.5)
    j = J/(np.sqrt(1.65894)/(F*c_bulk*(D**2*L)**(1.0/3.0)))
    return J

def to_dimensional_potential(theta,E0f,R=8.314,T=298,F=96485):
    E = theta/(F/(R*T)) + E0f
    return E 
    



def interp1DExperiment(Potential,Flux,deltaTheta,theta_i,theta_v):
    nTimeSteps = int(np.fabs(theta_v-theta_i)/deltaTheta)
    Esteps = np.arange(nTimeSteps)
    E = theta_i-deltaTheta*Esteps
    f = interp1d(Potential,Flux)
    Flux_New = f(E)
    return E,Flux_New
    



if __name__ == '__main__': 
	flux = load_experimental_flux(r"GroundTruth\rot_freq=2.00E+01 nu=1.35E-06 sigma=2.15E+00.csv",noise_level=0.02)
	plt.plot(flux)
	plt.show()
	


	"""
	Y = np.linspace(0,0.9999,num=1000).reshape(-1,1)

	InterpW = GetInterpW()
	W = InterpW(Y)

	#Y = 0.5

	plt.scatter(Y,W)
	plt.xlabel('Y')
	plt.ylabel('W')
	plt.show()
	"""

