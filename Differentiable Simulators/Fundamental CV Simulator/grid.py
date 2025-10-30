import numpy as np 
import jax.numpy as jnp
import jax.scipy.linalg as linag
import os
import jax


def calc_n(Xi,deltaX,maxX,expanding_grid_factor):
    current_X = Xi

    n = 0
    while current_X < maxX:
        current_X += deltaX 
        deltaX = deltaX*expanding_grid_factor
        
        n += 1 

    return n+1 

def ini_grid(n,Xi,deltaX,expanding_grid_factor):
    X_grid = jnp.zeros(n) + Xi
    dX = deltaX
    for i in jnp.arange(1,n):
        X_grid = X_grid.at[i].set(X_grid[i-1]+dX)
        dX = dX * expanding_grid_factor

    return X_grid


def genGrid(Xi,deltaX,maxX,expanding_grid_factor):
    
    n = calc_n(Xi=Xi,deltaX=deltaX,maxX=maxX,expanding_grid_factor=expanding_grid_factor)

    X_grid = ini_grid(n=n,Xi=Xi,deltaX=deltaX,expanding_grid_factor=expanding_grid_factor)

    return X_grid, n


def ini_conc(n,C_A_bulk,C_B_bulk):
    
    conc = jnp.zeros(2*n)

    concA = jnp.zeros(n)
    concB = jnp.zeros(n)


    concA = concA.at[:].set(C_A_bulk)
    concB = concB.at[:].set(C_B_bulk)

    conc = conc.at[:n].set(C_A_bulk)
    conc = conc.at[n:].set(C_B_bulk)

    conc_d = conc.at[:].set(conc[:])

    return conc,conc_d, concA,concB



def update_d(Theta,conc,conc_d,n,C_A_bulk,C_B_bulk,kinetics):
    conc_d = conc_d.at[:].set(conc[:])
    if kinetics == 'Nernst':
        conc_d = conc_d.at[n-1].set(1.0/(1.0+jnp.exp(-Theta)))
        conc_d = conc_d.at[n].set(0.0)

    elif kinetics =='BV':
        conc_d = conc_d.at[n-1].set(0.0)
        conc_d = conc_d.at[n].set(0.0)

    else:
        raise ValueError
    
    conc_d = conc_d.at[0].set(C_A_bulk)
    conc_d = conc_d.at[2*n-1].set(C_B_bulk)

    return conc_d




def calc_grad(conc,n,dA,dB,X_grid):

    g = - dA * (conc[n-2] - conc[n-1])/ (X_grid[1]-X_grid[0])
    return g


    




