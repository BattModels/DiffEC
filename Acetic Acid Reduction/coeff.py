import jax.numpy as jnp
import jax
import time
from helper import timer
from functools import partial


def Acalc_abc_radial(n,XGrid,deltaT,d_A,aA,bA,cA):



    arow = []
    aval = []

    brow = []
    bval = []
    
    crow = []
    cval = []

    arow.append(0)
    aval.append(0.0)

    brow.append(0)
    bval.append(0.0)

    crow.append(0)
    cval.append(0.0)
    



    deltaX_m = XGrid[1:n-1] - XGrid[0:n-2]
    deltaX_p = XGrid[2:n] - XGrid[1:n-1]


    aA = aA.at[1:n-1].set(d_A*((-(2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)) + 2.0 / XGrid[1:n-1] * (deltaT / (deltaX_m + deltaX_p)))))
    bA = bA.at[1:n-1].set(d_A*(((2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) + (2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)))) + 1.0)
    cA = cA.at[1:n-1].set(d_A*((-(2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) - 2.0 / XGrid[1:n-1] * (deltaT / (deltaX_m + deltaX_p)))))


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


def Bcalc_abc_radial(n,XGrid,deltaT,d_B,aB,bB,cB):
    arow = []
    aval = []

    brow = []
    bval = []
    
    crow = []
    cval = []

    arow.append(0)
    aval.append(0.0)

    brow.append(0)
    bval.append(0.0)

    crow.append(0)
    cval.append(0.0)
    



    deltaX_m = XGrid[1:n-1] - XGrid[0:n-2]
    deltaX_p = XGrid[2:n] - XGrid[1:n-1]


    aB = aB.at[1:n-1].set(d_B*((-(2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)) + 2.0 / XGrid[1:n-1] * (deltaT / (deltaX_m + deltaX_p)))))
    bB = bB.at[1:n-1].set(d_B*(((2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) + (2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)))) + 1.0)
    cB = cB.at[1:n-1].set(d_B*((-(2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) - 2.0 / XGrid[1:n-1] * (deltaT / (deltaX_m + deltaX_p)))))


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




def Ycalc_abc_radial(n,XGrid,deltaT,d_Y,aY,bY,cY):
    arow = []
    aval = []

    brow = []
    bval = []
    
    crow = []
    cval = []

    arow.append(0)
    aval.append(0.0)

    brow.append(0)
    bval.append(0.0)

    crow.append(0)
    cval.append(0.0)
    



    deltaX_m = XGrid[1:n-1] - XGrid[0:n-2]
    deltaX_p = XGrid[2:n] - XGrid[1:n-1]


    aY = aY.at[1:n-1].set(d_Y*((-(2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)) + 2.0 / XGrid[1:n-1] * (deltaT / (deltaX_m + deltaX_p)))))
    bY = bY.at[1:n-1].set(d_Y*(((2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) + (2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)))) + 1.0)
    cY = cY.at[1:n-1].set(d_Y*((-(2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) - 2.0 / XGrid[1:n-1] * (deltaT / (deltaX_m + deltaX_p)))))


    arow.append(-1)
    aval.append(0.0)

    brow.append(-1)
    bval.append(0.0)

    crow.append(-1)
    cval.append(0.0)


    aY = aY.at[jnp.array(arow)].set(jnp.array(aval))
    bY = bY.at[jnp.array(brow)].set(jnp.array(bval))
    cY = cY.at[jnp.array(crow)].set(jnp.array(cval))


    




    return aY,bY,cY

def Zcalc_abc_radial(n,XGrid,deltaT,d_Z,aZ,bZ,cZ):
    arow = []
    aval = []

    brow = []
    bval = []
    
    crow = []
    cval = []

    arow.append(0)
    aval.append(0.0)

    brow.append(0)
    bval.append(0.0)

    crow.append(0)
    cval.append(0.0)
    

    deltaX_m = XGrid[1:n-1] - XGrid[0:n-2]
    deltaX_p = XGrid[2:n] - XGrid[1:n-1]


    aZ = aZ.at[1:n-1].set(d_Z*((-(2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)) + 2.0 / XGrid[1:n-1] * (deltaT / (deltaX_m + deltaX_p)))))
    bZ = bZ.at[1:n-1].set(d_Z*(((2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) + (2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)))) + 1.0)
    cZ = cZ.at[1:n-1].set(d_Z*((-(2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) - 2.0 / XGrid[1:n-1] * (deltaT / (deltaX_m + deltaX_p)))))


    arow.append(-1)
    aval.append(0.0)

    brow.append(-1)
    bval.append(0.0)

    crow.append(-1)
    cval.append(0.0)


    aZ = aZ.at[jnp.array(arow)].set(jnp.array(aval))
    bZ = bZ.at[jnp.array(brow)].set(jnp.array(bval))
    cZ = cZ.at[jnp.array(crow)].set(jnp.array(cval))




    return aZ,bZ,cZ



def ini_coeff(n,XGrid,deltaX,TGrid,deltaT,maxX,maxT,d_A,d_B,d_Y,d_Z):
    aA = jnp.zeros(n)
    bA = jnp.zeros(n)
    cA = jnp.zeros(n)


    aB = jnp.zeros(n)
    bB = jnp.zeros(n)
    cB = jnp.zeros(n)


    aY = jnp.zeros(n)
    bY = jnp.zeros(n)
    cY = jnp.zeros(n)


    aZ = jnp.zeros(n)
    bZ = jnp.zeros(n)
    cZ = jnp.zeros(n)




    return aA,bA,cA,aB,bB,cB,aY,bY,cY,aZ,bZ,cZ


def ini_fx(n):
    return jnp.zeros(4*n)

def ini_jacob(n):
    return jnp.zeros((4*n,4*n))

def ini_dx(n):
    return jnp.zeros(4*n)


def update(x,C_A_bulk,C_B_bulk,C_Y_bulk,C_Z_bulk):
    
    d = x.copy()

    d = d.at[-4].set(C_A_bulk)
    d = d.at[-3].set(C_B_bulk)
    d = d.at[-2].set(C_Y_bulk)
    d = d.at[-1].set(C_Z_bulk)

    return d

def xupdate(x, dx):
    x = x + dx
    return x

@partial(jax.jit,static_argnums=0)
def Allcalc_abc(n,XGrid,deltaT,d_A,d_B,d_Y,d_Z,aA,bA,cA,aB,bB,cB,aY,bY,cY,aZ,bZ,cZ):
    aA,bA,cA = Acalc_abc_radial(n=n,XGrid=XGrid,deltaT=deltaT,d_A=d_A,aA=aA,bA=bA,cA=cA)
    aB,bB,cB = Bcalc_abc_radial(n=n,XGrid=XGrid,deltaT=deltaT,d_B=d_B,aB=aB,bB=bB,cB=cB)
    aY,bY,cY = Ycalc_abc_radial(n=n,XGrid=XGrid,deltaT=deltaT,d_Y=d_Y,aY=aY,bY=bY,cY=cY)
    aZ,bZ,cZ = Zcalc_abc_radial(n=n,XGrid=XGrid,deltaT=deltaT,d_Z=d_Z,aZ=aZ,bZ=bZ,cZ=cZ)


    return aA,bA,cA,aB,bB,cB,aY,bY,cY,aZ,bZ,cZ




@partial(jax.jit,static_argnums=7)
def calc_fx(x,d,deltaX,deltaT,alpha,K0,Theta,n,Kappa,Kf,Kb,d_A,d_B,d_Y,d_Z,aA,bA,cA,aB,bB,cB,aY,bY,cY,aZ,bZ,cZ,fx):
    h = deltaX



    Kred = K0*Kappa*jnp.exp(-alpha*Theta)
    Kox = K0*jnp.exp((2.0-alpha)*Theta)

    arow = []
    aval = []


    arow.append(0)
    aval.append(x[4]-x[0])

    arow.append(1)
    aval.append(x[1] + Kred*h* (1.0/d_B)*x[1]*x[1] - Kox*h*(1.0/d_B)*x[2]- x[5])

    arow.append(2)
    aval.append(x[2] - 0.5*Kred*h*(1.0/d_Y)*x[1]*x[1] + 0.5*Kox*h*(1.0/d_Y)*x[2] - x[6])

    arow.append(3)
    aval.append(x[7] - x[3])




    fx = fx.at[4:4*n-4:4].set(aA[1:n-1]*x[0:4*n-8:4] + bA[1:n-1]*x[4:4*n-4:4] + cA[1:n-1]*x[8:4*n:4] + Kf*deltaT*x[4:4*n-4:4] - Kb*deltaT*x[5:4*n-3:4]*x[7:4*n-1:4] - d[4:4*n-4:4])
    fx = fx.at[5:4*n-3:4].set(aB[1:n-1]*x[1:4*n-7:4] + bB[1:n-1]*x[5:4*n-3:4] + cB[1:n-1]*x[9:4*n+1:4]  - Kf*deltaT*x[4:4*n-4:4] + Kb*deltaT*x[5:4*n-3:4]*x[7:4*n-1:4] - d[5:4*n-3:4])
    fx = fx.at[6:4*n-2:4].set(aY[1:n-1]*x[2:4*n-6:4] + bY[1:n-1]*x[6:4*n-2:4] + cY[1:n-1]*x[10:4*n+2:4] - d[6:4*n-2:4])
    fx = fx.at[7:4*n-1:4].set(aZ[1:n-1]*x[3:4*n-5:4] + bZ[1:n-1]*x[7:4*n-1:4] + cZ[1:n-1]*x[11:4*n+3:4] - Kf*deltaT*x[4:4*n-4:4] + Kb*deltaT*x[5:4*n-3:4]*x[7:4*n-1:4] - d[7:4*n-1:4])


    arow.append(-4)
    aval.append(x[4*n-4] - d[4*n-4])
    arow.append(-3)
    aval.append(x[4*n-3] - d[4*n-3])
    arow.append(-2)
    aval.append(x[4*n-2] - d[4*n-2])
    arow.append(-1)
    aval.append(x[4*n-1] - d[4*n-1])


    fx = fx.at[jnp.array(arow)].set(jnp.array(aval))

    fx = -fx

    return fx



    
@partial(jax.jit,static_argnums=7)
def calc_jacob(x,d,deltaX,deltaT,alpha,K0,Theta,n,Kappa,Kf,Kb,d_A,d_B,d_Y,d_Z,aA,bA,cA,aB,bB,cB,aY,bY,cY,aZ,bZ,cZ,J):
    h = deltaX

    Kred = K0*Kappa*jnp.exp(-alpha*Theta)
    Kox = K0*jnp.exp((2.0-alpha)*Theta)

    arow = []
    acol = []
    aval = []

    arow.append(0)
    acol.append(0)
    aval.append(-1.0)
    arow.append(0)
    acol.append(4)
    aval.append(1.0)



    arow.append(1)
    acol.append(1)
    aval.append((1.0+(1.0/d_B)*Kred*h*2*x[1]))

    arow.append(1)
    acol.append(2)
    aval.append(-(1.0/d_B)*Kox*h)

    arow.append(1)
    acol.append(5)
    aval.append(-1.0)


    arow.append(2)
    acol.append(1)
    aval.append( (-1.0/d_Y)*Kred*h * x[1])

    arow.append(2)
    acol.append(2)
    aval.append((0.5*(1.0/d_Y)*Kox*h+1.0))

    arow.append(2)
    acol.append(6)
    aval.append(-1.0)

    arow.append(3)
    acol.append(3)
    aval.append(-1.0)

    arow.append(3)
    acol.append(7)
    aval.append(1.0)


    J = J.at[jnp.array(arow),jnp.array(acol)].set(jnp.array(aval))

    ##To Do: Implement the Jacobian Matrix


    J = J.at[jnp.arange(4,4*n-4,4),jnp.arange(0,4*n-8,4)].set(aA[1:n-1])
    J = J.at[jnp.arange(4,4*n-4,4),jnp.arange(4,4*n-4,4)].set(bA[1:n-1] + Kf*deltaT)
    J = J.at[jnp.arange(4,4*n-4,4),jnp.arange(5,4*n-3,4)].set(-Kb*deltaT * x[7:4*n-1:4])
    J = J.at[jnp.arange(4,4*n-4,4),jnp.arange(7,4*n-1,4)].set(-Kb*deltaT * x[5:4*n-3:4])
    J = J.at[jnp.arange(4,4*n-4,4),jnp.arange(8,4*n,4)].set(cA[1:n-1])


    J = J.at[jnp.arange(5,4*n-3,4),jnp.arange(1,4*n-7,4)].set(aB[1:n-1])
    J = J.at[jnp.arange(5,4*n-3,4),jnp.arange(4,4*n-4,4)].set(-Kf*deltaT)
    J = J.at[jnp.arange(5,4*n-3,4),jnp.arange(5,4*n-3,4)].set(bB[1:n-1]+Kb*deltaT*x[7:4*n-1:4])
    J = J.at[jnp.arange(5,4*n-3,4),jnp.arange(7,4*n-3,4)].set(Kb*deltaT*x[5:4*n-3:4])
    J = J.at[jnp.arange(5,4*n-3,4),jnp.arange(9,4*n+1,4)].set(cB[1:n-1])


    J = J.at[jnp.arange(6,4*n-2,4),jnp.arange(2,4*n-6,4)].set(aY[1:n-1])
    J = J.at[jnp.arange(6,4*n-2,4),jnp.arange(6,4*n-2,4)].set(bY[1:n-1])
    J = J.at[jnp.arange(6,4*n-2,4),jnp.arange(10,4*n+2,4)].set(cY[1:n-1])

    J = J.at[jnp.arange(7,4*n-1,4),jnp.arange(3,4*n-5,4)].set(aZ[1:n-1])
    J = J.at[jnp.arange(7,4*n-1,4),jnp.arange(4,4*n-4,4)].set(-Kf*deltaT)
    J = J.at[jnp.arange(7,4*n-1,4),jnp.arange(5,4*n-3,4)].set(Kb*deltaT*x[7:4*n-1:4])
    J = J.at[jnp.arange(7,4*n-1,4),jnp.arange(7,4*n-1,4)].set(bZ[1:n-1] + Kb*deltaT*x[5:4*n-3:4])
    J = J.at[jnp.arange(7,4*n-1,4),jnp.arange(11,4*n+3,4)].set(cZ[1:n-1])



    J = J.at[-4,-4].set(1.0)
    J = J.at[-3,-3].set(1.0)
    J = J.at[-2,-2].set(1.0)
    J = J.at[-1,-1].set(1.0)




    return J 

    
     




    


