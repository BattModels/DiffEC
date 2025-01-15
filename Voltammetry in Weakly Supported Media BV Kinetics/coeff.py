import jax.numpy as jnp
import jax
import time
from helper import timer
from functools import partial

def ini_fx(n):
    fx = jnp.zeros(5*n)
    return fx 

def ini_jacob(n):
    J = jnp.zeros((5*n,5*n))
    return J

def ini_dx(n):
    dx = jnp.zeros(5*n)
    return dx 

def xupdate(x, dx):
    x = x + dx
    return x

def update(x,C_A_bulk,C_B_bulk,C_M_bulk,C_N_bulk,Phi_ini):
    d = x.copy()

    d.at[-5].set(C_A_bulk)
    d.at[-4].set(C_B_bulk)
    d.at[-3].set(C_M_bulk)
    d.at[-2].set(C_N_bulk)
    d.at[-1].set(Phi_ini)

    return d 


def Acalc_abcd_linear(n,XGrid,deltaT,d_A,aA,bA,cA,dA):

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
    
    drow = []
    dval = []

    drow.append(0)
    dval.append(0.0)


    for i in range(1,n-1):
        deltaX_m = XGrid[i] - XGrid[i - 1]
        deltaX_p = XGrid[i + 1] - XGrid[i]
        arow.append(i)
        aval.append(d_A*(-2.0*deltaT)/(deltaX_m*(deltaX_m+deltaX_p)))

        brow.append(i)
        bval.append(d_A*(2.0*deltaT)/(deltaX_m*(deltaX_m+deltaX_p)) + d_A*(2.0*deltaT)/(deltaX_p*(deltaX_m+deltaX_p)))

        crow.append(i)
        cval.append(d_A*(-2.0*deltaT)/(deltaX_p*(deltaX_m+deltaX_p)))

        drow.append(i)
        dval.append(d_A*(-deltaT/((deltaX_m+deltaX_p)**2)))



    arow.append(-1)
    aval.append(0.0)

    brow.append(-1)
    bval.append(0.0)

    crow.append(-1)
    cval.append(0.0)
    
    drow.append(-1)
    dval.append(0.0)

    aA = aA.at[jnp.array(arow)].set(jnp.array(aval))
    bA = bA.at[jnp.array(brow)].set(jnp.array(bval))
    cA = cA.at[jnp.array(crow)].set(jnp.array(cval))
    dA = dA.at[jnp.array(drow)].set(jnp.array(dval))

    return aA,bA,cA,dA


def Bcalc_abcd_linear(n,XGrid,deltaT,d_B,aB,bB,cB,dB):
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
    
    drow = []
    dval = []

    drow.append(0)
    dval.append(0.0)


    for i in range(1,n-1):
        deltaX_m = XGrid[i] - XGrid[i - 1]
        deltaX_p = XGrid[i + 1] - XGrid[i]
        arow.append(i)
        aval.append(d_B*(-2.0*deltaT)/(deltaX_m*(deltaX_m+deltaX_p)))

        brow.append(i)
        bval.append(d_B*(2.0*deltaT)/(deltaX_m*(deltaX_m+deltaX_p)) + d_B*(2.0*deltaT)/(deltaX_p*(deltaX_m+deltaX_p)))

        crow.append(i)
        cval.append(d_B*(-2.0*deltaT)/(deltaX_p*(deltaX_m+deltaX_p)))

        drow.append(i)
        dval.append(d_B*(-deltaT/((deltaX_m+deltaX_p)**2)))




    arow.append(-1)
    aval.append(0.0)



    brow.append(-1)
    bval.append(0.0)




    crow.append(-1)
    cval.append(0.0)


    drow.append(-1)
    dval.append(0.0)

    aB = aB.at[jnp.array(arow)].set(jnp.array(aval))
    bB = bB.at[jnp.array(brow)].set(jnp.array(bval))
    cB = cB.at[jnp.array(crow)].set(jnp.array(cval))
    dB = dB.at[jnp.array(drow)].set(jnp.array(dval))

    return aB,bB,cB,dB


def Mcalc_abcd_linear(n,XGrid,deltaT,d_M,aM,bM,cM,dM):
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
    
    drow = []
    dval = []

    drow.append(0)
    dval.append(0.0)


    for i in range(1,n-1):
        deltaX_m = XGrid[i] - XGrid[i - 1]
        deltaX_p = XGrid[i + 1] - XGrid[i]
        arow.append(i)
        aval.append(d_M*(-2.0*deltaT)/(deltaX_m*(deltaX_m+deltaX_p)))

        brow.append(i)
        bval.append(d_M*(2.0*deltaT)/(deltaX_m*(deltaX_m+deltaX_p)) + d_M*(2.0*deltaT)/(deltaX_p*(deltaX_m+deltaX_p)))

        crow.append(i)
        cval.append(d_M*(-2.0*deltaT)/(deltaX_p*(deltaX_m+deltaX_p)))

        drow.append(i)
        dval.append(d_M*(-deltaT/((deltaX_m+deltaX_p)**2)))



    arow.append(-1)
    aval.append(0.0)



    brow.append(-1)
    bval.append(0.0)



    crow.append(-1)
    cval.append(0.0)


    drow.append(-1)
    dval.append(0.0)

    aM = aM.at[jnp.array(arow)].set(jnp.array(aval))
    bM = bM.at[jnp.array(brow)].set(jnp.array(bval))
    cM = cM.at[jnp.array(crow)].set(jnp.array(cval))
    dM = dM.at[jnp.array(drow)].set(jnp.array(dval))

    return aM,bM,cM,dM


def Ncalc_abcd_linear(n,XGrid,deltaT,d_N,aN,bN,cN,dN):
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
    
    drow = []
    dval = []

    drow.append(0)
    dval.append(0.0)


    for i in range(1,n-1):
        deltaX_m = XGrid[i] - XGrid[i - 1]
        deltaX_p = XGrid[i + 1] - XGrid[i]
        arow.append(i)
        aval.append(d_N*(-2.0*deltaT)/(deltaX_m*(deltaX_m+deltaX_p)))

        brow.append(i)
        bval.append(d_N*(2.0*deltaT)/(deltaX_m*(deltaX_m+deltaX_p)) + d_N*(2.0*deltaT)/(deltaX_p*(deltaX_m+deltaX_p)))

        crow.append(i)
        cval.append(d_N*(-2.0*deltaT)/(deltaX_p*(deltaX_m+deltaX_p)))

        drow.append(i)
        dval.append(d_N*(-deltaT/((deltaX_m+deltaX_p)**2)))



    arow.append(-1)
    aval.append(0.0)


    brow.append(-1)
    bval.append(0.0)




    crow.append(-1)
    cval.append(0.0)
    


    drow.append(-1)
    dval.append(0.0)

    aN = aN.at[jnp.array(arow)].set(jnp.array(aval))
    bN = bN.at[jnp.array(brow)].set(jnp.array(bval))
    cN = cN.at[jnp.array(crow)].set(jnp.array(cval))
    dN = dN.at[jnp.array(drow)].set(jnp.array(dval))

    return aN,bN,cN,dN


def Phicalc_abcd_linear(n,XGrid,deltaT,aPhi,bPhi,cPhi,dPhi):
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
    
    drow = []
    dval = []

    drow.append(0)
    dval.append(0.0)


    for i in range(1,n-1):
        deltaX_m = XGrid[i] - XGrid[i - 1]
        deltaX_p = XGrid[i + 1] - XGrid[i]       
        
        arow.append(i)
        aval.append(2.0/(deltaX_m*(deltaX_m+deltaX_p)))
        brow.append(i)
        bval.append((-2.0)/(deltaX_m*(deltaX_m+deltaX_p)) + (-2.0)/(deltaX_p*(deltaX_m+deltaX_p)))
        crow.append(i) 
        cval.append(2.0/(deltaX_p*(deltaX_m+deltaX_p)))
        drow.append(i)
        dval.append(0.0)




    arow.append(-1)
    aval.append(0.0)

    brow.append(-1)
    bval.append(0.0)


    crow.append(-1)
    cval.append(0.0)
    


    drow.append(-1)
    dval.append(0.0)

    aPhi = aPhi.at[jnp.array(arow)].set(jnp.array(aval))
    bPhi = bPhi.at[jnp.array(brow)].set(jnp.array(bval))
    cPhi = cPhi.at[jnp.array(crow)].set(jnp.array(cval))
    dPhi = dPhi.at[jnp.array(drow)].set(jnp.array(dval))

    return aPhi,bPhi,cPhi,dPhi

@jax.jit
def update_fx_in_loop(i,Re,x,d,d_A,d_B,d_M,d_N,z_A,z_B,z_M,z_N,aA,bA,cA,dA, aB,bB,cB,dB, aM,bM,cM,dM, aN,bN,cN,dN, aPhi,bPhi,cPhi,dPhi):
    aval0 = aA[i]*x[5*i-5] + bA[i]*x[5*i] + cA[i]*x[5*i+5] + z_A*dA[i]*x[5*i+5]*x[5*i+9] - z_A*dA[i]*x[5*i+5]*x[5*i-1] - z_A*dA[i]*x[5*i-5]*x[5*i+9] + z_A*dA[i]*x[5*i-5]*x[5*i-1] + z_A*x[5*i]*aA[i]*x[5*i-1]   + z_A*x[5*i]*bA[i]*x[5*i+4]   + z_A*x[5*i]*cA[i]*x[5*i+9]   + x[5*i]   - d[5*i]

    aval1 = aB[i]*x[5*i-4] + bB[i]*x[5*i+1] + cB[i]*x[5*i+6] + z_B*dB[i]*x[5*i+6]*x[5*i+9] - z_B*dB[i]*x[5*i+6]*x[5*i-1] - z_B*dB[i]*x[5*i-4]*x[5*i+9] + z_B*dB[i]*x[5*i-4]*x[5*i-1] + z_B*x[5*i+1]*aB[i]*x[5*i-1] + z_B*x[5*i+1]*bB[i]*x[5*i+4] + z_B*x[5*i+1]*cB[i]*x[5*i+9] + x[5*i+1] - d[5*i+1]
    
    aval2 = aM[i]*x[5*i-3] + bM[i]*x[5*i+2] + cM[i]*x[5*i+7] + z_M*dM[i]*x[5*i+7]*x[5*i+9] - z_M*dM[i]*x[5*i+7]*x[5*i-1] - z_M*dM[i]*x[5*i-3]*x[5*i+9] + z_M*dM[i]*x[5*i-3]*x[5*i-1] + z_M*x[5*i+2]*aM[i]*x[5*i-1] + z_M*x[5*i+2]*bM[i]*x[5*i+4] + z_M*x[5*i+2]*cM[i]*x[5*i+9] + x[5*i+2] - d[5*i+2]

    aval3 = aN[i]*x[5*i-2] + bN[i]*x[5*i+3] + cN[i]*x[5*i+8] + z_N*dN[i]*x[5*i+8]*x[5*i+9] - z_N*dN[i]*x[5*i+8]*x[5*i-1] - z_N*dN[i]*x[5*i-2]*x[5*i+9] + z_N*dN[i]*x[5*i-2]*x[5*i-1] + z_N*x[5*i+3]*aN[i]*x[5*i-1] + z_N*x[5*i+3]*bN[i]*x[5*i+4] + z_N*x[5*i+3]*cN[i]*x[5*i+9] + x[5*i+3] - d[5*i+3]

    aval4 = aPhi[i]*x[5*i-1] + bPhi[i]*x[5*i+4] + cPhi[i]*x[5*i+9] + Re*Re*z_A*x[5*i] + Re*Re*z_B*x[5*i+1] + Re*Re*z_M*x[5*i+2] + Re*Re*z_N*x[5*i+3]


    return [aval0,aval1,aval2,aval3,aval4]

#@timer
@partial(jax.jit,static_argnums=8)
def calc_fx(K0,alpha,beta,x,d,deltaX,deltaT,Theta,n,Re,d_A,d_B,d_M,d_N,z_A,z_B,z_M,z_N,aA,bA,cA,dA, aB,bB,cB,dB, aM,bM,cM,dM, aN,bN,cN,dN, aPhi,bPhi,cPhi,dPhi,fx,):
    h = deltaX
    Kred_prime = K0*jnp.exp(-alpha*Theta)
    Kox_prime = K0*jnp.exp(beta*Theta)
    arow = []
    aval = []
    
    arow.append(0)
    aval.append((1+Kred_prime*jnp.exp(alpha*x[4])*h/d_A)*x[0] - Kox_prime*jnp.exp(-beta*x[4])*h/d_A * x[1] - x[5])
    arow.append(1)
    aval.append((1+Kox_prime*jnp.exp(-beta*x[4])*h/d_B)*x[1] - Kred_prime*jnp.exp(alpha*x[4])*h/d_B * x[0] - x[6])
    arow.append(2)
    aval.append(x[7] - x[2])
    arow.append(3)
    aval.append(x[8]-x[3])
    arow.append(4)
    aval.append(x[9] - x[4])


    """
    for j in range(5,5*n-5,5):
        i = int(j/5)
        
   
        #aval1 = (aA[i]*x[5*i-5] + bA[i]*x[5*i] + cA[i]*x[5*i+5] + z_A*dA[i]*x[5*i+5]*x[5*i+9] - z_A*dA[i]*x[5*i+5]*x[5*i-1] - z_A*dA[i]*x[5*i-5]*x[5*i+9] + z_A*dA[i]*x[5*i-5]*x[5*i-1] + z_A*x[5*i]*aA[i]*x[5*i-1]   + z_A*x[5*i]*bA[i]*x[5*i+4]   + z_A*x[5*i]*cA[i]*x[5*i+9]   + x[5*i]   - d[5*i])

        #aval2 = (aB[i]*x[5*i-4] + bB[i]*x[5*i+1] + cB[i]*x[5*i+6] + z_B*dB[i]*x[5*i+6]*x[5*i+9] - z_B*dB[i]*x[5*i+6]*x[5*i-1] - z_B*dB[i]*x[5*i-4]*x[5*i+9] + z_B*dB[i]*x[5*i-4]*x[5*i-1] + z_B*x[5*i+1]*aB[i]*x[5*i-1] + z_B*x[5*i+1]*bB[i]*x[5*i+4] + z_B*x[5*i+1]*cB[i]*x[5*i+9] + x[5*i+1] - d[5*i+1])
        
        #aval3 = (aM[i]*x[5*i-3] + bM[i]*x[5*i+2] + cM[i]*x[5*i+7] + z_M*dM[i]*x[5*i+7]*x[5*i+9] - z_M*dM[i]*x[5*i+7]*x[5*i-1] - z_M*dM[i]*x[5*i-3]*x[5*i+9] + z_M*dM[i]*x[5*i-3]*x[5*i-1] + z_M*x[5*i+2]*aM[i]*x[5*i-1] + z_M*x[5*i+2]*bM[i]*x[5*i+4] + z_M*x[5*i+2]*cM[i]*x[5*i+9] + x[5*i+2] - d[5*i+2])

        #aval4 = (aN[i]*x[5*i-2] + bN[i]*x[5*i+3] + cN[i]*x[5*i+8] + z_N*dN[i]*x[5*i+8]*x[5*i+9] - z_N*dN[i]*x[5*i+8]*x[5*i-1] - z_N*dN[i]*x[5*i-2]*x[5*i+9] + z_N*dN[i]*x[5*i-2]*x[5*i-1] + z_N*x[5*i+3]*aN[i]*x[5*i-1] + z_N*x[5*i+3]*bN[i]*x[5*i+4] + z_N*x[5*i+3]*cN[i]*x[5*i+9] + x[5*i+3] - d[5*i+3])

        #aval5 = (aPhi[i]*x[5*i-1] + bPhi[i]*x[5*i+4] + cPhi[i]*x[5*i+9] + Re*Re*z_A*x[5*i] + Re*Re*z_B*x[5*i+1] + Re*Re*z_M*x[5*i+2] + Re*Re*z_N*x[5*i+3])
 
        avals = update_fx_in_loop(i,Re,x,d,d_A,d_B,d_M,d_N,z_A,z_B,z_M,z_N,aA,bA,cA,dA, aB,bB,cB,dB, aM,bM,cM,dM, aN,bN,cN,dN, aPhi,bPhi,cPhi,dPhi)

        arow.extend([j,j+1,j+2,j+3,j+4])
        aval.extend(avals)
    """

    fx = fx.at[5:5*n-5:5].set(aA[1:n-1]*x[0:5*n-10:5] + bA[1:n-1]*x[5:5*n-5:5] + cA[1:n-1]*x[10:5*n:5]      + z_A*dA[1:n-1]*x[10:5*n:5]*x[14:5*n+4:5]  - z_A*dA[1:n-1]*x[10:5*n:5]*x[4:5*n-6:5]   - z_A*dA[1:n-1]*x[0:5*n-10:5]*x[14:5*n+4:5] + z_A*dA[1:n-1]*x[0:5*n-10:5]*x[4:5*n-6:5]         +z_A*x[5:5*n-5:5]*aA[1:n-1]*x[4:5*n-6:5]       + z_A*x[5:5*n-5:5]*bA[1:n-1]*x[9:5*n-1:5]    + z_A*x[5:5*n-5:5]*cA[1:n-1]*x[14:5*n+4:5]  +  x[5:5*n-5:5] - d[5:5*n-5:5])
    fx = fx.at[6:5*n-4:5].set(aB[1:n-1]*x[1:5*n-9:5]  + bB[1:n-1]*x[6:5*n-4:5] + cB[1:n-1]*x[11:5*n+1:5]    + z_B*dB[1:n-1]*x[11:5*n+1:5]*x[14:5*n+4:5]- z_B*dB[1:n-1]*x[11:5*n+1:5]*x[4:5*n-6:5] - z_B*dB[1:n-1]*x[1:5*n-9:5]*x[14:5*n+4:5]  + z_B*dB[1:n-1]*x[1:5*n-9:5]*x[4:5*n-6:5]         +z_B*x[6:5*n-4:5]*aB[1:n-1]*x[4:5*n-6:5]       + z_B*x[6:5*n-4:5]*bB[1:n-1]*x[9:5*n-1:5]    + z_B*x[6:5*n-4:5]*cB[1:n-1]*x[14:5*n+4:5]  +  x[6:5*n-4:5] - d[6:5*n-4:5])
    fx = fx.at[7:5*n-3:5].set(aM[1:n-1]*x[2:5*n-8:5]  + bM[1:n-1]*x[7:5*n-3:5] + cM[1:n-1]*x[12:5*n+2:5]    + z_M*dM[1:n-1]*x[12:5*n+2:5]*x[14:5*n+4:5]- z_M*dM[1:n-1]*x[12:5*n+2:5]*x[4:5*n-6:5] - z_M*dM[1:n-1]*x[2:5*n-8:5]*x[14:5*n+4:5]  + z_M*dM[1:n-1]*x[2:5*n-8:5]*x[4:5*n-6:5]         +z_M*x[7:5*n-3:5]*aM[1:n-1]*x[4:5*n-6:5]       + z_M*x[7:5*n-3:5]*bM[1:n-1]*x[9:5*n-1:5]    + z_M*x[7:5*n-3:5]*cM[1:n-1]*x[14:5*n+4:5]  +  x[7:5*n-3:5] - d[7:5*n-3:5])
    fx = fx.at[8:5*n-2:5].set(aN[1:n-1]*x[3:5*n-7:5]  + bN[1:n-1]*x[8:5*n-2:5] + cN[1:n-1]*x[13:5*n+3:5]    + z_N*dN[1:n-1]*x[13:5*n+3:5]*x[14:5*n+4:5]- z_N*dN[1:n-1]*x[13:5*n+3:5]*x[4:5*n-6:5] - z_N*dN[1:n-1]*x[3:5*n-7:5]*x[14:5*n+4:5]  + z_N*dN[1:n-1]*x[3:5*n-7:5]*x[4:5*n-6:5]         +z_N*x[8:5*n-2:5]*aN[1:n-1]*x[4:5*n-6:5]       + z_N*x[8:5*n-2:5]*bN[1:n-1]*x[9:5*n-1:5]    + z_N*x[8:5*n-2:5]*cN[1:n-1]*x[14:5*n+4:5]  +  x[8:5*n-2:5] - d[8:5*n-2:5])
    fx = fx.at[9:5*n-1:5].set(aPhi[1:n-1]*x[4:5*n-6:5]  + bPhi[1:n-1]*x[9:5*n-1:5] + cPhi[1:n-1]*x[14:5*n+4:5]  + Re**2*z_A*x[5:5*n-5:5] + Re**2*z_B*x[6:5*n-4:5] + Re**2*z_M*x[7:5*n-3:5] + Re**2*z_N*x[8:5*n-2:5])



    
    arow.append(5*n-5)
    aval.append(x[n*5-5]-d[5*n-5])

    arow.append(5*n-4)
    aval.append(x[n*5-4]-d[5*n-4])

    arow.append(5*n-3)
    aval.append(x[n*5-3]-d[5*n-3])


    arow.append(5*n-2)
    aval.append(x[n*5-2]-d[5*n-2])

    arow.append(5*n-1)
    aval.append(x[n*5-1]-d[5*n-1])


    fx = fx.at[jnp.array(arow)].set(jnp.array(aval))


    fx = -fx



    return fx

@jax.jit
def calc_J_A(row,i,x,aA,bA,cA,dA,z_A):
    val0 = aA[i] - z_A*dA[i]*x[5*i+9] + z_A*dA[i]*x[5*i-1]
    val1 = -z_A*dA[i]*x[5*i+5] + z_A*dA[i]*x[5*i-5] + z_A*x[5*i]*aA[i]
    val2 = bA[i] + z_A*aA[i]*x[5*i-1] + z_A*bA[i]*x[5*i+4] + z_A*cA[i]*x[5*i+9] + 1.0 
    val3 = z_A*x[5*i]*bA[i] 
    val4 = cA[i] + z_A*dA[i]*x[5*i+9] - z_A*dA[i]*x[5*i-1]
    val5 =  z_A*dA[i]*x[5*i+5] - z_A*dA[i]*x[5*i-5] + z_A*x[5*i]*cA[i]

    return [val0,val1,val2,val3,val4,val5]


@jax.jit
def calc_J_B(row,i,x,aB,bB,cB,dB,z_B):
    val0 = aB[i] - z_B*dB[i]*x[5*i+9] + z_B*dB[i]*x[5*i-1]
    val1 = -z_B*dB[i]*x[5*i+6] + z_B*dB[i]*x[5*i-4] + z_B*x[5*i+1]*aB[i] 
    val2 = bB[i] +z_B*aB[i]*x[5*i-1] + z_B*bB[i]*x[5*i+4] + z_B*cB[i]*x[5*i+9] + 1.0
    val3 = z_B*x[5*i+1]*bB[i]
    val4 = cB[i] + z_B*dB[i]*x[5*i+9] - z_B*dB[i]*x[5*i-1]
    val5 = z_B*dB[i]*x[5*i+6] - z_B*dB[i]*x[5*i-4] + z_B*x[5*i+1]*cB[i]

    return [val0,val1,val2,val3,val4,val5]


@jax.jit
def calc_J_M(row,i,x,aM,bM,cM,dM,z_M):
    val0 = aM[i] - z_M*dM[i]*x[5*i+9] + z_M*dM[i]*x[5*i-1]
    val1 = -z_M*dM[i]*x[5*i+7] + z_M*dM[i]*x[5*i-3] + z_M*x[5*i+2]*aM[i]
    val2 = bM[i] + z_M*aM[i]*x[5*i-1] + z_M*bM[i]*x[5*i+4] + z_M*cM[i]*x[5*i+9] + 1.0
    val3 = z_M*x[5*i+2]*bM[i]
    val4 = cM[i] + z_M*dM[i]*x[5*i+9] - z_M*dM[i]*x[5*i-1]
    val5 = z_M*dM[i]*x[5*i+7] - z_M*dM[i]*x[5*i-3] + z_M*x[5*i+2]*cM[i]

    return [val0,val1,val2,val3,val4,val5]


@jax.jit
def calc_J_N(row,i,x,aN,bN,cN,dN,z_N):
    val0 = aN[i] - z_N*dN[i]*x[5*i+9] + z_N*dN[i]*x[5*i-1]
    val1 = -z_N*dN[i]*x[5*i+8] + z_N*dN[i]*x[5*i-2] + z_N*x[5*i+3]*aN[i]
    val2 = bN[i] + z_N*aN[i]*x[5*i-1] + z_N*bN[i]*x[5*i+4] + z_N*cN[i]*x[5*i+9] + 1.0
    val3 = z_N*x[5*i+3]*bN[i] 
    val4 = cN[i] + z_N*dN[i]*x[5*i+9] - z_N*dN[i]*x[5*i-1]
    val5 = z_N*dN[i]*x[5*i+8] - z_N*dN[i]*x[5*i-2] + z_N*x[5*i+3]*cN[i]

    return [val0,val1,val2,val3,val4,val5]
@jax.jit
def calc_J_Phi(row,i,x,aPhi,bPhi,cPhi,dPhi,Re,z_A,z_B,z_M,z_N):

    #Initialize Phi
    val0 = aPhi[i]
    val1   = Re*Re*z_A 
    val2 = Re*Re*z_B
    val3 = Re*Re*z_M
    val4 = Re*Re*z_N
    val5 = bPhi[i] 
    val6 = cPhi[i]

    return [val0,val1,val2,val3,val4,val5,val6]
#@timer
@partial(jax.jit,static_argnums=8)
def calc_jacob(K0,alpha,beta,x,d,deltaX,deltaT,Theta,n,Re,d_A,d_B,d_M,d_N,z_A,z_B,z_M,z_N,aA,bA,cA,dA, aB,bB,cB,dB, aM,bM,cM,dM, aN,bN,cN,dN, aPhi,bPhi,cPhi,dPhi,J):
    h = deltaX
    Kred_prime = K0*jnp.exp(-alpha*Theta)
    Kox_prime = K0*jnp.exp(beta*Theta)
    arow = []
    acol = []
    aval = []

    arow.append(0)
    acol.append(0)
    aval.append(Kred_prime*jnp.exp(alpha*x[4])*h/d_A + 1.0)

    arow.append(0)
    acol.append(1)
    aval.append(-Kox_prime*jnp.exp(-beta*x[4])*h/d_A)

    arow.append(0)
    acol.append(4)
    aval.append(alpha*h/d_A*Kred_prime*jnp.exp(alpha*x[4])*x[0] + beta*h/d_A*Kox_prime*jnp.exp(-beta*x[4])*x[1])

    arow.append(0)
    acol.append(5)
    aval.append(-1.0)

    arow.append(1)
    acol.append(0)
    aval.append(-Kred_prime*jnp.exp(alpha*x[4])*h/d_B)

    arow.append(1)
    acol.append(1)
    aval.append(Kox_prime*jnp.exp(-beta*x[4])*h/d_B + 1.0)

    arow.append(1)
    acol.append(4)
    aval.append(-alpha*h/d_B*Kred_prime*jnp.exp(alpha*x[4])*x[0] - beta*h/d_B*Kox_prime*jnp.exp(-beta*x[4])*x[1])

    arow.append(1)
    acol.append(6)
    aval.append(-1.0)

    arow.append(2)
    acol.append(2)
    aval.append(-1.0)

    arow.append(2)
    acol.append(7)
    aval.append(1.0)

    arow.append(3)
    acol.append(3)
    aval.append(-1.0)
    
    arow.append(3)
    acol.append(8)
    aval.append(1.0)

    arow.append(4)
    acol.append(4)
    aval.append(-1.0)


    arow.append(4)
    acol.append(9)
    aval.append(1.0)

    
    J = J.at[jnp.arange(5,5*n-5,5),jnp.arange(0,5*n-10,5)].set(aA[1:n-1] - z_A*dA[1:n-1]*x[14:5*n+4:5] + z_A*dA[1:n-1]*x[4:5*n-6:5])
    J = J.at[jnp.arange(5,5*n-5,5),jnp.arange(4,5*n-6,5)].set(-z_A*dA[1:n-1]*x[10:5*n:5] + z_A*dA[1:n-1]*x[0:5*n-10:5] + z_A*x[5:5*n-5:5]*aA[1:n-1])
    J = J.at[jnp.arange(5,5*n-5,5),jnp.arange(5,5*n-5,5)].set(bA[1:n-1] + z_A*aA[1:n-1]*x[4:5*n-6:5] + z_A*bA[1:n-1]*x[9:5*n-1:5] + z_A*cA[1:n-1]*x[14:5*n+4:5] + 1.0)
    J = J.at[jnp.arange(5,5*n-5,5),jnp.arange(9,5*n-1,5)].set(z_A*x[5:5*n-5:5]*bA[1:n-1])
    J = J.at[jnp.arange(5,5*n-5,5),jnp.arange(10,5*n,5)].set(cA[1:n-1] + z_A*dA[1:n-1]*x[14:5*n+4:5] - z_A*dA[1:n-1]*x[4:5*n-6:5])
    J = J.at[jnp.arange(5,5*n-5,5),jnp.arange(14,5*n+4,5)].set(z_A*dA[1:n-1]*x[10:5*n:5] - z_A*dA[1:n-1]*x[0:5*n-10:5] + z_A*x[5:5*n-5:5]*cA[1:n-1])


    J = J.at[jnp.arange(6,5*n-4,5),jnp.arange(1,5*n-9,5)].set(aB[1:n-1] - z_B*dB[1:n-1]*x[14:5*n+4:5] + z_B*dB[1:n-1]*x[4:5*n-6:5])
    J = J.at[jnp.arange(6,5*n-4,5),jnp.arange(4,5*n-6,5)].set(-z_B*dB[1:n-1]*x[11:5*n+1:5] + z_B*dB[1:n-1]*x[1:5*n-9:5] + z_B*x[6:5*n-4:5]*aB[1:n-1])
    J = J.at[jnp.arange(6,5*n-4,5),jnp.arange(6,5*n-4,5)].set(bB[1:n-1] + z_B*aB[1:n-1]*x[4:5*n-6:5] + z_B*bB[1:n-1]*x[9:5*n-1:5] + z_B*cB[1:n-1]*x[14:5*n+4:5] + 1.0)
    J = J.at[jnp.arange(6,5*n-4,5),jnp.arange(9,5*n-1,5)].set(z_B*x[6:5*n-4:5]*bB[1:n-1])
    J = J.at[jnp.arange(6,5*n-4,5),jnp.arange(11,5*n+1,5)].set(cB[1:n-1] + z_B*dB[1:n-1]*x[14:5*n+4:5] - z_B*dB[1:n-1]*x[4:5*n-6:5])
    J = J.at[jnp.arange(6,5*n-4,5),jnp.arange(14,5*n+4,5)].set(z_B*dB[1:n-1]*x[11:5*n+1:5] - z_B*dB[1:n-1]*x[1:5*n-9:5] + z_B*x[6:5*n-4:5]*cB[1:n-1])


    J = J.at[jnp.arange(7,5*n-3,5),jnp.arange(2,5*n-8,5)].set(aM[1:n-1] - z_M*dM[1:n-1]*x[14:5*n+4:5] + z_M*dM[1:n-1]*x[4:5*n-6:5])
    J = J.at[jnp.arange(7,5*n-3,5),jnp.arange(4,5*n-6,5)].set(-z_M*dM[1:n-1]*x[12:5*n+2:5] + z_M*dM[1:n-1]*x[2:5*n-8:5] + z_M*x[7:5*n-3:5]*aM[1:n-1])
    J = J.at[jnp.arange(7,5*n-3,5),jnp.arange(7,5*n-3,5)].set(bM[1:n-1] + z_M*aM[1:n-1]*x[4:5*n-6:5] + z_M*bM[1:n-1]*x[9:5*n-1:5] + z_M*cM[1:n-1]*x[14:5*n+4:5] + 1.0)
    J = J.at[jnp.arange(7,5*n-3,5),jnp.arange(9,5*n-1,5)].set(z_M*x[7:5*n-3:5]*bM[1:n-1])
    J = J.at[jnp.arange(7,5*n-3,5),jnp.arange(12,5*n+2,5)].set(cM[1:n-1] + z_M*dM[1:n-1]*x[14:5*n+4:5] - z_M*dM[1:n-1]*x[4:5*n-6:5])
    J = J.at[jnp.arange(7,5*n-3,5),jnp.arange(14,5*n+4,5)].set(z_M*dM[1:n-1]*x[12:5*n+2:5] - z_M*dM[1:n-1]*x[2:5*n-8:5] + z_M*x[7:5*n-3:5]*cM[1:n-1])


    J = J.at[jnp.arange(8,5*n-2,5),jnp.arange(3,5*n-7,5)].set(aN[1:n-1] - z_N*dN[1:n-1]*x[14:5*n+4:5] + z_N*dN[1:n-1]*x[4:5*n-6:5])
    J = J.at[jnp.arange(8,5*n-2,5),jnp.arange(4,5*n-6,5)].set(-z_N*dN[1:n-1]*x[13:5*n+3:5] + z_N*dN[1:n-1]*x[3:5*n-7:5] + z_N*x[8:5*n-2:5]*aN[1:n-1])
    J = J.at[jnp.arange(8,5*n-2,5),jnp.arange(8,5*n-2,5)].set(bN[1:n-1] + z_N*aN[1:n-1]*x[4:5*n-6:5] + z_N*bN[1:n-1]*x[9:5*n-1:5] + z_N*cN[1:n-1]*x[14:5*n+4:5] + 1.0)
    J = J.at[jnp.arange(8,5*n-2,5),jnp.arange(9,5*n-1,5)].set(z_N*x[8:5*n-2:5]*bN[1:n-1])
    J = J.at[jnp.arange(8,5*n-2,5),jnp.arange(13,5*n+3,5)].set(cN[1:n-1] + z_N*dN[1:n-1]*x[14:5*n+4:5] - z_N*dN[1:n-1]*x[4:5*n-6:5])
    J = J.at[jnp.arange(8,5*n-2,5),jnp.arange(14,5*n+4,5)].set(z_N*dN[1:n-1]*x[13:5*n+3:5] - z_N*dN[1:n-1]*x[3:5*n-7:5] + z_N*x[8:5*n-2:5]*cN[1:n-1])

    
        

    J = J.at[jnp.arange(9,5*n-1,5),jnp.arange(4,5*n-6,5)].set(aPhi[1:n-1]) 
    J = J.at[jnp.arange(9,5*n-1,5),jnp.arange(5,5*n-5,5)].set(Re**2*z_A)
    J = J.at[jnp.arange(9,5*n-1,5),jnp.arange(6,5*n-4,5)].set(Re**2*z_B)
    J = J.at[jnp.arange(9,5*n-1,5),jnp.arange(7,5*n-3,5)].set(Re**2*z_M)
    J = J.at[jnp.arange(9,5*n-1,5),jnp.arange(8,5*n-2,5)].set(Re**2*z_N)
    J = J.at[jnp.arange(9,5*n-1,5),jnp.arange(9,5*n-1,5)].set(bPhi[1:n-1])
    J = J.at[jnp.arange(9,5*n-1,5),jnp.arange(14,5*n+4,5)].set(cPhi[1:n-1])

    arow.append(5*n-5)
    acol.append(5*n-5)
    aval.append(1.0)

    arow.append(5*n-4)
    acol.append(5*n-4)
    aval.append(1.0)

    arow.append(5*n-3)
    acol.append(5*n-3)
    aval.append(1.0)

    arow.append(5*n-2)
    acol.append(5*n-2)
    aval.append(1.0)

    arow.append(5*n-1)
    acol.append(5*n-1)
    aval.append(1.0)



    J = J.at[jnp.array(arow),jnp.array(acol)].set(jnp.array(aval))


    return J






def Allcalc_abc(n,XGrid,deltaT,d_A,d_B,d_M,d_N,aA,bA,cA,dA, aB,bB,cB,dB, aM,bM,cM,dM, aN,bN,cN,dN, aPhi,bPhi,cPhi,dPhi):
    
    aA,bA,cA,dA = Acalc_abcd_linear(n=n,XGrid=XGrid,deltaT=deltaT,d_A=d_A,aA=aA,bA=bA,cA=cA,dA=dA)
    aB,bB,cB,dB = Bcalc_abcd_linear(n=n,XGrid=XGrid,deltaT=deltaT,d_B=d_B,aB=aB,bB=bB,cB=cB,dB=dB)
    aM,bM,cM,dM = Mcalc_abcd_linear(n=n,XGrid=XGrid,deltaT=deltaT,d_M=d_M,aM=aM,bM=bM,cM=cM,dM=dM)
    aN,bN,cN,dN = Ncalc_abcd_linear(n=n,XGrid=XGrid,deltaT=deltaT,d_N=d_N,aN=aN,bN=bN,cN=cN,dN=dN)
    aPhi,bPhi,cPhi,dPhi = Phicalc_abcd_linear(n=n, XGrid=XGrid,deltaT=deltaT,aPhi=aPhi,bPhi=bPhi,cPhi=cPhi,dPhi=dPhi)

    return  aA,bA,cA,dA, aB,bB,cB,dB, aM,bM,cM,dM, aN,bN,cN,dN, aPhi,bPhi,cPhi,dPhi




def ini_coeff(n,XGrid,deltaX,deltaT,maxX,K0,alpha,beta,Re,d_A,d_B,d_M,d_N,z_A,z_B,z_M,z_N):
    aA = jnp.zeros(n)
    bA = jnp.zeros(n)
    cA = jnp.zeros(n)
    dA = jnp.zeros(n)

    aB = jnp.zeros(n)
    bB = jnp.zeros(n)
    cB = jnp.zeros(n)
    dB = jnp.zeros(n)

    aM = jnp.zeros(n)
    bM = jnp.zeros(n)
    cM = jnp.zeros(n)
    dM = jnp.zeros(n)


    aN = jnp.zeros(n)
    bN = jnp.zeros(n)
    cN = jnp.zeros(n)
    dN = jnp.zeros(n)


    aPhi = jnp.zeros(n)
    bPhi = jnp.zeros(n)
    cPhi = jnp.zeros(n)
    dPhi = jnp.zeros(n)


    aA,bA,cA,dA, aB,bB,cB,dB, aM,bM,cM,dM, aN,bN,cN,dN, aPhi,bPhi,cPhi,dPhi = Allcalc_abc(n,XGrid=XGrid,deltaT=deltaT,d_A=d_A,d_B=d_B,d_M=d_M,d_N=d_N,aA=aA,bA=bA,cA=cA,dA=dA,aB=aB,bB=bB,cB=cB,dB=dB,aM=aM,bM=bM,cM=cM,dM=dM,aN=aN,bN=bN,cN=cN,dN=dN,aPhi=aPhi,bPhi=bPhi,cPhi=cPhi,dPhi=dPhi)

    return aA,bA,cA,dA, aB,bB,cB,dB, aM,bM,cM,dM, aN,bN,cN,dN, aPhi,bPhi,cPhi,dPhi 




