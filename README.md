# DiffEC
This is a code repository in company with "Differentiable Electrochemistry: A paradigm for uncovering hidden physical phenomena in electrochemical systems" currently under consideration. 

This repository features differentiable simulation of voltammetry covering diffusion, migration and convection, along with both Butler-Volmer and Marcus-Hush-Chidsey kinetics. In short, electrochemical simulations are made end-to-end differentiable for parameter estimation and optimization. 

The electrochemical simulations used Finite Difference (FD) or Finite Element (FE) methods, and made differentiable using JAX. 

![TOC](TOC.png)

# Requirements 
The programs are run with Python 3.11 and JAX 0.4.34.  The memory requirements for parameter estimations of nonlinear problems are very high. For nonlinear problems, it was run with 480 GB of memory on 6 CPU cores. For linear problem, a normal laptop with 16 GB of memory will suffice. 

# Differentiable Simulators
Since Differentiable Simulation is a new regime in scientific modeling, it is thus very important for beginners to learn the art of differentiable simulation and differentiable simulation in the context of electrochemistry. In here, five simulators that are fully differentiable and transferable are provided to enlighten readers the art of differentiable simulation. They five differentiable simulators are:

1. Fundamental Cyclic Voltammetry Simulator. Models the simplest one-electron redox reaction. 
2. Voltammetry in Weakly Supported Media. Extends fundamental CV with electrolyte migration effects by solving Nernst-Planck-Poisson equation. 
3. Dissociative EC Simulator. Models coupled chemical-electrochemical mechanisms, where a species dissociates chemically before electron transfer. A <-> B+C , B+e<sup>-</sup> <-> D. 
4. Hydrodynamic Voltammetry Simulator. Models rotating-disk electrode (RDE) experiments under convection-diffusion mass transport. 
5. Electrochemical Adsorption/Desorption Simulator. Simulates surface-confined redox reactions and adsorption/desorption processes. The mechanism is shown below:
![Electrochemical Adsorption/Desorption Mechanism](AdsorptionMechanism.png)

Overall, these simulators cover a broad range of electrochemical phenomena (diffusion, migration, convection, and coupled reactions), and are differentiable with respect to key physicochemical parameters, enabling **gradient-based fitting, sensitivity analysis, and machine learning integration**.
 

# Contents 
* Voltammetry in weakly supported media BV kinetics: Estimating electrochemical kinetics from Butler-Volmer or Marcus-Hush-Chidsey formalism with migration-diffusion mass transport described with Nernst-Planck-Poisson equations
* Voltammetry in weakly supported media MHC kinetics: Estimating electrochemical kinetics from Marcus-Hush-Chidsey formalism with migration-diffusion mass transport described with Nernst-Planck-Poisson equations
* Chronoamperometry of acetic acid reduction: Estimating nonlinear chemical kinetics with convergent diffusion mass transport 
* Hydrodynamic voltammetry: Estimating kinematic viscosity with convection-diffusion mass transport. This example is available at https://colab.research.google.com/drive/1Pq3szUPe8uvd9pw-ZVAZSCmX8nQH2CSM?usp=sharing
* Transfer coefficient from LSV on Rotating Electrode: Using differentiable electrochemistry to build a direct correlation between voltammogram and its underlying physical/chemical properties. In this example, transfer coefficient and electrochemical rate constants are identified. 
* Mass transport in concentrated electrolytes: Estimating salt diffusivity and transference number from operando fields based on concentrated solution theory


# Issue Reports
We recommend issue reports in the Discussions channel. 
