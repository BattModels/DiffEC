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
        deltaX = deltaX * expanding_grid_factor
        n +=  1

    return n + 1 

def ini_grid(n,deltaX,expanding_grid_factor):
    XGrid = jnp.zeros(n)
    dX = deltaX 
    for i in jnp.arange(1,n):
        XGrid = XGrid.at[i].set(XGrid[i-1] + dX)
        dX = dX * expanding_grid_factor
    return XGrid

def genGrid(deltaX,maxX,expanding_grid_factor):
    Xi = 1.0

    n = calc_n(Xi=Xi,deltaX=deltaX,maxX=maxX,expanding_grid_factor=expanding_grid_factor)

    XGrid = ini_grid(n,deltaX=deltaX,expanding_grid_factor=expanding_grid_factor)

    return XGrid,n


def ini_conc(n,C_A_bulk,C_B_bulk,C_M_bulk,C_N_bulk,Phi_ini):
    conc = jnp.zeros(5*n)
    concA = jnp.zeros(n)
    concB = jnp.zeros(n)
    concM = jnp.zeros(n)
    concN = jnp.zeros(n)
    concPhi = jnp.zeros(n)

    concA = concA.at[:].set(C_A_bulk)
    concB = concB.at[:].set(C_B_bulk)
    concM = concM.at[:].set(C_M_bulk)
    concN = concN.at[:].set(C_N_bulk)
    concPhi = concPhi.at[:].set(Phi_ini)


    conc = conc.at[::5].set(C_A_bulk)
    conc = conc.at[1::5].set(C_B_bulk)
    conc = conc.at[2::5].set(C_M_bulk)
    conc = conc.at[3::5].set(C_N_bulk)
    conc = conc.at[4::5].set(Phi_ini)


    return conc,concA,concB,concM,concN,concPhi 
    

def calc_grad(x,deltaX):

    g = -(x[5]-x[0])/deltaX
    return g
