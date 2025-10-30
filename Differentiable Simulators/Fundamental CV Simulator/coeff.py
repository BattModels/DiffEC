import jax.numpy as jnp
import jax
import time
from functools import partial
from jax.scipy.integrate import trapezoid
import numpy as np



def ini_coeff(n,X_grid,deltaX,maxX,dA,dB):



    A_matrix = jnp.zeros((2*n,2*n))

    aA = jnp.zeros(n)
    bA = jnp.zeros(n)
    cA = jnp.zeros(n)


    aB = jnp.zeros(n)
    bB = jnp.zeros(n)
    cB = jnp.zeros(n)


    return A_matrix, aA,bA,cA,aB,bB,cB,



def Acal_abc_linear(n,X_grid,deltaT,aA,bA,cA,dA):
    arow = []
    aval = []
    arow.append(0)
    aval.append(0.0)

    brow = []
    bval = []
    brow.append(0)
    bval.append(0.0)

    crow = []
    cval = []
    crow.append(0)
    cval.append(0.0)

    for i in range(1,n-1):
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

    brow = []
    bval = []
    brow.append(0)
    bval.append(0.0)

    crow = []
    cval = []
    crow.append(0)
    cval.append(0.0)

    for i in range(1,n-1):
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

    brow = []
    bval = []
    brow.append(0)
    bval.append(0.0)

    crow = []
    cval = []
    crow.append(0)
    cval.append(0.0)

    for i in range(1,n-1):
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

    brow = []
    bval = []
    brow.append(0)
    bval.append(0.0)

    crow = []
    cval = []
    crow.append(0)
    cval.append(0.0)

    for i in range(1,n-1):
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



def CalcMatrix(A_matrix,X_grid,Theta,kinetics,n,aA,bA,cA,dA,aB,bB,cB,dB,K0,alpha,beta):

    arow = []
    acol = []
    aval = []
    if kinetics == 'Nernst':
        arow.append(n-1)
        acol.append(n-2)
        aval.append(0.0)

        arow.append(n-1)
        acol.append(n-1)
        aval.append(1.0)

        arow.append(n-1)
        acol.append(n)
        aval.append(0.0)


        arow.append(n)
        acol.append(n-2)
        aval.append(-dA)

        arow.append(n)
        acol.append(n-1)
        aval.append(dA)

        arow.append(n)
        acol.append(n)
        aval.append(dB)

        arow.append(n)
        acol.append(n+1)
        aval.append(-dB)



    elif kinetics == "BV":
        X0 = X_grid[1] - X_grid[0]
        K_red = K0*jnp.exp(-alpha*Theta)
        K_ox = K0*jnp.exp(beta*Theta)

        arow.append(n-1)
        acol.append(n-2)
        aval.append(-1.0)

        arow.append(n-1)
        acol.append(n-1)
        aval.append(1.0 + X0/dA*K_red)

        arow.append(n-1)
        acol.append(n)
        aval.append(-X0/dA * K_ox)

        arow.append(n)
        acol.append(n-1)
        aval.append(- X0/dB*K_red)

        arow.append(n)
        acol.append(n)
        aval.append((1.0 + X0/dB*K_ox))

        arow.append(n)
        acol.append(n+1)
        aval.append(-1.0)

    else:
        raise ValueError
    

    A_matrix = A_matrix.at[jnp.arange(n-2,0,-1),jnp.arange(n-3,-1,-1)].set(cA[1:n-1])
    A_matrix = A_matrix.at[jnp.arange(n-2,0,-1),jnp.arange(n-2,0,-1)].set(bA[1:n-1])
    A_matrix = A_matrix.at[jnp.arange(n-2,0,-1),jnp.arange(n-1,1,-1)].set(aA[1:n-1])




    A_matrix = A_matrix.at[jnp.arange(n+1,2*n-1),jnp.arange(n,2*n-2)].set(aB[1:n-1])
    A_matrix = A_matrix.at[jnp.arange(n+1,2*n-1),jnp.arange(n+1,2*n-1)].set(bB[1:n-1])
    A_matrix = A_matrix.at[jnp.arange(n+1,2*n-1),jnp.arange(n+2,2*n)].set(cB[1:n-1])




    arow.append(0)
    acol.append(0)
    aval.append(1.0)

    arow.append(0)
    acol.append(1)
    aval.append(0.0)

    arow.append(2*n-1)
    acol.append(2*n-1)
    aval.append(1.0)

    arow.append(2*n-1)
    acol.append(2*n-2)
    aval.append(0.0)

    A_matrix = A_matrix.at[jnp.array(arow),jnp.array(acol)].set(jnp.array(aval))


    return A_matrix










def Allcalc_abc_linear(n,X_grid,deltaT,aA,bA,cA,dA,aB,bB,cB,dB):
    aA,bA,cA = Acal_abc_linear(n=n,X_grid=X_grid,deltaT=deltaT,aA=aA,bA=bA,cA=cA,dA=dA)
    aB,bB,cB = Bcal_abc_linear(n=n,X_grid=X_grid,deltaT=deltaT,aB=aB,bB=bB,cB=cB,dB=dB)

    return aA,bA,cA,aB,bB,cB


def Allcacl_abc_radial(n,X_grid,deltaT,aA,bA,cA,dA,aB,bB,cB,dB):
    aA,bA,cA = Acal_abc_radial(n=n,X_grid=X_grid,deltaT=deltaT,aA=aA,bA=bA,cA=cA,dA=dA)
    aB,bB,cB = Bcal_abc_radial(n=n,X_grid=X_grid,deltaT=deltaT,aB=aB,bB=bB,cB=cB,dB=dB)

    return aA,bA,cA,aB,bB,cB