import numpy as np 
import jax.numpy as jnp
import jax.scipy.linalg as linag
import os
import jax

def calc_n(Xi,deltaX,maxX,expanding_grid_factor):
    current_X = Xi
    
    n = 1 #Reserve the adsorption layer

    while current_X < maxX:
        current_X += deltaX
        deltaX = deltaX*expanding_grid_factor

        n+=1 

    return n+1

def ini_grid(n,Xi,deltaX,expanding_grid_factor):
    X_grid = jnp.zeros(n) + Xi
    dX = deltaX

    for i in jnp.arange(2,n):
        X_grid = X_grid.at[i].set(X_grid[i-1]+dX)
        dX = dX*expanding_grid_factor

    return X_grid

def genGrid(Xi,deltaX,maxX,expanding_grid_factor):
    n = calc_n(Xi=Xi,deltaX=deltaX,maxX=maxX,expanding_grid_factor=expanding_grid_factor)
    X_grid = ini_grid(n=n,Xi=Xi,deltaX=deltaX,expanding_grid_factor=expanding_grid_factor)

    return X_grid,n


def ini_conc(n,C_A_bulk,C_B_bulk,zeta_A_ini,zeta_B_ini):

    conc  = jnp.zeros(2*n)

    conc = conc.at[0].set(zeta_A_ini)
    conc = conc.at[1].set(zeta_B_ini)

    conc = conc.at[2::2].set(C_A_bulk)
    conc = conc.at[3::2].set(C_B_bulk)

    conc_d = conc.at[:].set(conc[:])

    return conc, conc_d


def update_d(conc,conc_d,C_A_bulk,C_B_bulk):
    conc_d = conc_d.at[:].set(conc[:])

    conc_d = conc_d.at[-1].set(C_A_bulk)
    conc_d = conc_d.at[-2].set(C_B_bulk)
    return conc_d


def calc_grad(conc, conc_d,X_grid,deltaT,saturation_number):
    flux = - (conc[4] - conc[2])/(X_grid[2] - X_grid[1]) + 1.0/saturation_number*(conc[0]-conc_d[0])/deltaT

    return flux
