import jax.numpy as jnp
import jax
import time
from functools import partial
from jax.scipy.integrate import trapezoid
import numpy as np



def ini_coeff(n):


    J = jnp.zeros((2*n,2*n))
    fx = jnp.zeros(2*n)
    dx = jnp.zeros(2*n)

    aA = jnp.zeros(n)
    bA = jnp.zeros(n)
    cA = jnp.zeros(n)


    aB = jnp.zeros(n)
    bB = jnp.zeros(n)
    cB = jnp.zeros(n)



    return J,fx,dx,aA,bA,cA,aB,bB,cB

def xupdate(conc,dx):
    conc = conc + dx
    return conc 

def Acal_abc_linear(n,X_grid,deltaT,aA,bA,cA,dA):
    arow = []
    aval = []
    arow.append(0)
    aval.append(0.0)
    arow.append(1)
    aval.append(0.0)

    brow = []
    bval = []
    brow.append(0)
    bval.append(0.0)
    brow.append(1)
    bval.append(0.0)

    crow = []
    cval = []
    crow.append(0)
    cval.append(0.0)
    crow.append(1)
    cval.append(0.0)

    for i in range(2,n-1):
        deltaX_m = X_grid[i] - X_grid[i-1]
        deltaX_p = X_grid[i+1] - X_grid[i]

        arow.append(i)
        aval.append(dA*((-2.0*deltaT)/(deltaX_m*(deltaX_m+deltaX_p))))

        crow.append(i)
        cval.append(dA*((-2.0*deltaT)/(deltaX_p*(deltaX_m+deltaX_p))))

        brow.append(i)
        bval.append(1.0 - (dA*((-2.0*deltaT)/(deltaX_m*(deltaX_m+deltaX_p)))) - (dA*((-2.0*deltaT)/(deltaX_p*(deltaX_m+deltaX_p)))))


    arow.append(-1)
    aval.append(0.0)

    brow.append(-1)
    bval.append(0.0)

    crow.append(-1)
    cval.append(0.0)

    aA = aA.at[jnp.array(arow)].set(jnp.array(aval))
    bA = bA.at[jnp.array(brow)].set(jnp.array(bval))
    cA = cA.at[jnp.array(crow)].set(jnp.array(cval))

    return aA,bA,cA

def Acal_abc_radial(n,X_grid,deltaT,aA,bA,cA,dA):
    arow = []
    aval = []
    arow.append(0)
    aval.append(0.0)
    arow.append(1)
    aval.append(0.0)

    brow = []
    bval = []
    brow.append(0)
    bval.append(0.0)
    brow.append(1)
    bval.append(0.0)

    crow = []
    cval = []
    crow.append(0)
    cval.append(0.0)
    crow.append(1)
    cval.append(0.0)

    for i in range(2,n-1):
        deltaX_m = X_grid[i] - X_grid[i-1]
        deltaX_p = X_grid[i+1] - X_grid[i]


        arow.append(i)
        aval.append(dA*((-(2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)) + 2.0 / X_grid[i] * (deltaT / (deltaX_m + deltaX_p)))))

        brow.append(i)
        bval.append(dA*(((2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) + (2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)))) + 1.0)


        crow.append(i)
        cval.append(dA*((-(2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) - 2.0 / X_grid[i] * (deltaT / (deltaX_m + deltaX_p)))))

    arow.append(-1)
    aval.append(0.0)

    brow.append(-1)
    bval.append(0.0)

    crow.append(-1)
    cval.append(0.0)

    aA = aA.at[jnp.array(arow)].set(jnp.array(aval))
    bA = bA.at[jnp.array(brow)].set(jnp.array(bval))
    cA = cA.at[jnp.array(crow)].set(jnp.array(cval))

    return aA,bA,cA


def Bcal_abc_radial(n,X_grid,deltaT,aB,bB,cB,dB):
    arow = []
    aval = []
    arow.append(0)
    aval.append(0.0)
    arow.append(1)
    aval.append(0.0)

    brow = []
    bval = []
    brow.append(0)
    bval.append(0.0)
    brow.append(1)
    bval.append(0.0)

    crow = []
    cval = []
    crow.append(0)
    cval.append(0.0)
    crow.append(1)
    cval.append(0.0)

    for i in range(2,n-1):
        deltaX_m = X_grid[i] - X_grid[i-1]
        deltaX_p = X_grid[i+1] - X_grid[i]

        arow.append(i)
        aval.append(dB*((-(2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)) + 2.0 / X_grid[i] * (deltaT / (deltaX_m + deltaX_p)))))

        brow.append(i)
        bval.append(dB*(((2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) + (2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)))) + 1.0)

        crow.append(i)
        cval.append(dB*((-(2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) - 2.0 / X_grid[i] * (deltaT / (deltaX_m + deltaX_p)))))

    arow.append(-1)
    aval.append(0.0)

    brow.append(-1)
    bval.append(0.0)

    crow.append(-1)
    cval.append(0.0)

    aB = aB.at[jnp.array(arow)].set(jnp.array(aval))
    bB = bB.at[jnp.array(brow)].set(jnp.array(bval))
    cB = cB.at[jnp.array(crow)].set(jnp.array(cval))

    return aB,bB,cB

def Bcal_abc_linear(n,X_grid,deltaT,aB,bB,cB,dB):

    arow = []
    aval = []
    arow.append(0)
    aval.append(0.0)
    arow.append(1)
    aval.append(0.0)

    brow = []
    bval = []
    brow.append(0)
    bval.append(0.0)
    brow.append(1)
    bval.append(0.0)

    crow = []
    cval = []
    crow.append(0)
    cval.append(0.0)
    crow.append(1)
    cval.append(0.0)

    for i in range(2,n-1):
        deltaX_m = X_grid[i] - X_grid[i-1]
        deltaX_p = X_grid[i+1] - X_grid[i]

        arow.append(i)
        aval.append(dB*((-2.0*deltaT)/(deltaX_m*(deltaX_m+deltaX_p))))

        crow.append(i)
        cval.append(dB*((-2.0*deltaT)/(deltaX_p*(deltaX_m+deltaX_p))))

        brow.append(i)
        bval.append(1.0 - dB*((-2.0*deltaT)/(deltaX_m*(deltaX_m+deltaX_p))) - dB*((-2.0*deltaT)/(deltaX_p*(deltaX_m+deltaX_p))))


    arow.append(-1)
    aval.append(0.0)

    brow.append(-1)
    bval.append(0.0)

    crow.append(-1)
    cval.append(0.0)

    aB = aB.at[jnp.array(arow)].set(jnp.array(aval))
    bB = bB.at[jnp.array(brow)].set(jnp.array(bval))
    cB = cB.at[jnp.array(crow)].set(jnp.array(cval))

    return aB,bB,cB


def Allcalc_abc_linear(n,X_grid,deltaT,aA,bA,cA,dA,aB,bB,cB,dB):
    aA,bA,cA = Acal_abc_linear(n=n,X_grid=X_grid,deltaT=deltaT,aA=aA,bA=bA,cA=cA,dA=dA)
    aB,bB,cB = Bcal_abc_linear(n=n,X_grid=X_grid,deltaT=deltaT,aB=aB,bB=bB,cB=cB,dB=dB)

    return aA,bA,cA,aB,bB,cB

def Allcacl_abc_radial(n,X_grid,deltaT,aA,bA,cA,dA,aB,bB,cB,dB):
    aA,bA,cA = Acal_abc_radial(n=n,X_grid=X_grid,deltaT=deltaT,aA=aA,bA=bA,cA=cA,dA=dA)
    aB,bB,cB = Bcal_abc_radial(n=n,X_grid=X_grid,deltaT=deltaT,aB=aB,bB=bB,cB=cB,dB=dB)

    return aA,bA,cA,aB,bB,cB

@partial(jax.jit, static_argnames=["n","kinetics",'mode'])
def calc_jacob(J,Theta,n,X_grid,conc,deltaT,K0,alpha,beta,K0_ads,alpha_ads,beta_ads,K_A_ads,K_A_des,K_B_ads,K_B_des,C_A_bulk,C_B_bulk,saturation_number,kinetics,mode,dA,dB,reorg_e,theta_f_ads,aA,bA,cA,aB,bB,cB):
    
    arow = []
    acol = []
    aval = []

    if kinetics == "BV":
        X0 = X_grid[2] - X_grid[1]

        

        Kred = K0 * jnp.exp(-alpha*Theta)
        Kox = K0 * jnp.exp(beta*Theta)

        Kred_ads = K0_ads * jnp.exp(-alpha_ads*(Theta-theta_f_ads))
        Kox_ads = K0_ads * jnp.exp(beta_ads*(Theta-theta_f_ads))

        arow.append(0)
        acol.append(0)
        aval.append(-deltaT*Kred_ads - deltaT*K_A_ads*saturation_number*conc[2] - deltaT*K_A_des*saturation_number - 1.0)

        arow.append(0)
        acol.append(1)
        aval.append(deltaT*Kox_ads - deltaT*K_A_ads*saturation_number*conc[0])

        arow.append(0)
        acol.append(2)
        aval.append(deltaT*K_A_ads*saturation_number*(1.0-conc[0]-conc[1]))

        arow.append(0)
        acol.append(3)
        aval.append(0.0)



        arow.append(1)
        acol.append(0)
        aval.append(deltaT*Kred_ads - deltaT*K_B_ads*saturation_number*conc[3])

        arow.append(1)
        acol.append(1)
        aval.append(-deltaT*Kox_ads - deltaT*K_B_ads*saturation_number*conc[3] - deltaT*K_B_des*saturation_number - 1.0)

        arow.append(1)
        acol.append(2)
        aval.append(0.0)

        arow.append(1)
        acol.append(3)
        aval.append(deltaT*K_B_ads*saturation_number*(1.0-conc[0]-conc[1]))



        arow.append(2)
        acol.append(0)
        aval.append(-X0/dA*K_A_ads*conc[2] - X0/dA*K_A_des)

        arow.append(2)
        acol.append(1)
        aval.append(-X0/dA*K_A_ads*conc[2])

        arow.append(2)
        acol.append(2)
        aval.append(X0/dA*Kred + X0/dA*K_A_ads*(1.0-conc[0]-conc[1]) + 1.0)

        arow.append(2)
        acol.append(3)
        aval.append(-X0/dA*Kox)

        arow.append(2)
        acol.append(4)
        aval.append(-1.0)

        arow.append(2)
        acol.append(5)
        aval.append(0.0)

        arow.append(3)
        acol.append(0)
        aval.append(-X0/dB*K_B_ads*conc[3])

        arow.append(3)
        acol.append(1)
        aval.append(X0/dB*K_B_ads*conc[3] - X0/dB*K_B_des)

        arow.append(3)
        acol.append(2)
        aval.append(-X0/dB*Kred)

        arow.append(3)
        acol.append(3)
        aval.append(X0/dB*Kox + X0/dB*K_B_ads*(1.0-conc[0]-conc[1]) + 1.0)

        arow.append(3)
        acol.append(4)
        aval.append(0.0)

        arow.append(3)
        acol.append(5)
        aval.append(-1.0)


    else:
        raise ValueError

    J = J.at[jnp.arange(4,2*n-2,2),jnp.arange(2,2*n-4,2)].set(aA[2:n-1])
    J = J.at[jnp.arange(4,2*n-2,2),jnp.arange(4,2*n-2,2)].set(bA[2:n-1])
    J = J.at[jnp.arange(4,2*n-2,2),jnp.arange(6,2*n,2)].set(cA[2:n-1])

    J = J.at[jnp.arange(5,2*n-1,2),jnp.arange(3,2*n-3,2)].set(aB[2:n-1])
    J = J.at[jnp.arange(5,2*n-1,2),jnp.arange(5,2*n-1,2)].set(bB[2:n-1])
    J = J.at[jnp.arange(5,2*n-1,2),jnp.arange(7,2*n+1,2)].set(cB[2:n-1])


    arow.append(2*n-2)
    acol.append(2*n-2)
    aval.append(1.0)

    arow.append(2*n-1)
    acol.append(2*n-1)
    aval.append(1.0)

    J = J.at[jnp.array(arow),jnp.array(acol)].set(jnp.array(aval))

    return J


@partial(jax.jit, static_argnames=["n","kinetics",'mode'])
def calc_fx(fx,Theta,n,X_grid,conc,conc_d,deltaT,K0,alpha,beta,K0_ads,alpha_ads,beta_ads,K_A_ads,K_A_des,K_B_ads,K_B_des,C_A_bulk,C_B_bulk,saturation_number,kinetics,mode,dA,dB,reorg_e,theta_f_ads,aA,bA,cA,aB,bB,cB):
    arow = []
    aval = []

    if kinetics == "BV":
        X0 = X_grid[2] - X_grid[1]

        Kred = K0 * jnp.exp(-alpha*Theta)
        Kox = K0 * jnp.exp(beta*Theta)

        Kred_ads = K0_ads * jnp.exp(-alpha_ads*(Theta-theta_f_ads))
        Kox_ads = K0_ads * jnp.exp(beta_ads*(Theta-theta_f_ads))

        arow.append(0)
        aval.append( -deltaT*Kred_ads*conc[0] + deltaT*Kox_ads*conc[1] + deltaT*K_A_ads*saturation_number*conc[2]*(1.0-conc[0]-conc[1]) - deltaT*K_A_des*saturation_number*conc[0] - conc[0] + conc_d[0])

        arow.append(1)
        aval.append(deltaT*Kred_ads*conc[0] - deltaT*Kox_ads*conc[1] + deltaT*K_B_ads*saturation_number*conc[3]*(1.0-conc[0]-conc[1]) - deltaT*K_B_des*saturation_number*conc[1] - conc[1] + conc_d[1])

        arow.append(2)
        aval.append(X0/dA*Kred*conc[2] - X0/dA*Kox*conc[3] + X0/dA*K_A_ads*conc[2]*(1.0-conc[0]-conc[1]) - X0/dA*K_A_des*conc[0] - conc[4] + conc[2])

        arow.append(3)
        aval.append( -X0/dB*Kred*conc[2] + X0/dB*Kox*conc[3] + X0/dB*K_B_ads*conc[3]*(1.0-conc[0]-conc[1]) - X0/dB*K_B_des*conc[1] - conc[5] + conc[3])

    else: 
        raise ValueError
    

    fx = fx.at[jnp.arange(4,2*n-2,2)].set(aA[2:n-1]*conc[2:2*n-4:2] + bA[2:n-1]*conc[4:2*n-2:2] + cA[2:n-1]*conc[6:2*n:2] - conc_d[4:2*n-2:2])
    fx = fx.at[jnp.arange(5,2*n-1,2)].set(aB[2:n-1]*conc[3:2*n-3:2] + bB[2:n-1]*conc[5:2*n-1:2] + cB[2:n-1]*conc[7:2*n+1:2] - conc_d[5:2*n-1:2])


    arow.append(2*n-2) 
    aval.append(conc[2*n-2] - conc_d[2*n-2])

    arow.append(2*n-1)
    aval.append(conc[2*n-1] - conc_d[2*n-1])

    fx = fx.at[jnp.array(arow)].set(jnp.array(aval))

    fx = -fx


    return fx 