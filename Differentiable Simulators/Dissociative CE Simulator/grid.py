import numpy as np 
import jax.numpy as jnp
import jax.scipy.linalg as linag
import os
import jax


def calc_n(Xi,deltaX,maxX,expanding_grid_factor):
    current_X = Xi
    n = 0
    dX = deltaX
    while current_X < maxX:
        current_X += dX
        dX = dX * expanding_grid_factor
        n +=  1

    return n + 1 

def calc_nt(Ti,deltaT,maxT,expanding_time_factor):
    current_T = Ti
    nt = 0
    dT = deltaT
    while current_T < maxT:
        current_T += dT
        dT = dT * expanding_time_factor
        nt+=1



    return nt+1

def ini_XGrid(n,deltaX,expanding_grid_factor):
    Xi = 1.0
    XGrid = jnp.zeros(n)
    XGrid = XGrid.at[0].set(Xi)
    dX = deltaX 
    for i in jnp.arange(1,n):
        XGrid = XGrid.at[i].set(XGrid[i-1] + dX)
        dX = dX * expanding_grid_factor
    return XGrid

def ini_TGrid(nt,deltaT,expanding_time_factor):
    TGrid = jnp.zeros(nt)
    TGrid = TGrid.at[0].set(0.0)
    dT = deltaT
    for i in jnp.arange(1,nt):
        TGrid = TGrid.at[i].set(TGrid[i-1] + dT)

        dT = dT * expanding_time_factor
        
    return TGrid


def genGrid(deltaX,maxX,expanding_grid_factor,deltaT,maxT,expanding_time_factor):
    Xi = 1.0
    Ti = 0.0

    n = calc_n(Xi=Xi,deltaX=deltaX,maxX=maxX,expanding_grid_factor=expanding_grid_factor)
    nt = calc_nt(Ti=Ti,deltaT=deltaT,maxT=maxT,expanding_time_factor=expanding_time_factor)

    XGrid = ini_XGrid(n,deltaX=deltaX,expanding_grid_factor=expanding_grid_factor)
    TGrid = ini_TGrid(nt,deltaT=deltaT,expanding_time_factor=expanding_time_factor)

    return n,XGrid,nt,TGrid



def ini_conc(n,C_A_bulk,C_B_bulk,C_Y_bulk,C_Z_bulk):
    conc = jnp.zeros(4*n)
    concA = jnp.zeros(n)
    concB = jnp.zeros(n)
    concY = jnp.zeros(n)
    concZ = jnp.zeros(n)


    concA = concA.at[:].set(C_A_bulk)
    concB = concB.at[:].set(C_B_bulk)
    concY = concY.at[:].set(C_Y_bulk)
    concZ = concZ.at[:].set(C_Z_bulk)

    conc = conc.at[::4].set(C_A_bulk)
    conc = conc.at[1::4].set(C_B_bulk)
    conc = conc.at[2::4].set(C_Y_bulk)
    conc = conc.at[3::4].set(C_Z_bulk)

    return conc,concA,concB,concY,concZ


def calc_grad(x,deltaX):

    flux = -(x[5]-x[1])/deltaX
    return flux