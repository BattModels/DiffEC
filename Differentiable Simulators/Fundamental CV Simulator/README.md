# Fundamental Cyclic Voltammetry Simulator 

This folder provides the differentiable implementation of a fundamental differentiable electrochemistry simulator of the following reaction:

A + e<sup>-</sup> = B


The simulator supports Butler-Volmer or Nernst boundary condition, linear or convergent diffusion mass transport. 

Note that the entire simulator process is differentiable. You can easily differentiate the error of simulated current compared with experiment with any simulation parameters. 

To run the simulator,just run **simulation.py**