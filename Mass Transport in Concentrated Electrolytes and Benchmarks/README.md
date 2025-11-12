# Mass transport in concentrated electrolytes

This directory contains a differentiable finite volume solver for modeling mass transport in electrolytes, implemented in Python using JAX and JAXopt. The solver is designed to optimize parameters related to electrolyte transport properties (diffusion coefficients and transference number) by fitting simulation results to experimental data.


The figure below provides a graphical overview of this project. (a) A schematic of the Li symmertic cell and the moving electrolyte frame because of stripping of anode/electrodeposition of cathode. (b) The predicted and experimental concentration profiles at different time. (c) The predicted and experimental transference number and diffusion coefficients at different concentration of electrolyte. Note that the transference number is negative at at a highly concentrated electrolyte (>2M). (d) The predicted velocity profile and in comparison with experiment.

![LiSymmetric](LiSymmetric.png)

## Requirements

- Python 3.11.5  
- JAX 0.5.3  
- JAXopt 0.8.4  


## Directory Structure

- **benchmarks/** — Scripts for benchmarking various gradient-free methods (BO, CMA-ES, PSO, NM).
- **data/** — Contains all experimental datasets.  
- **results/** — Generated output from simulations using optimized parameters.  
- **solver.py** — Script for differentiable finite volume simulation using JAX.
- **bfgs.py** — Implementation of the BFGS optimization using JAXopt based on the gradient information obtained using automatic differentiation (AD) provided by JAX. 
- **plot.py** — Script for visualizing simulation results.  
- **utils.py** — Auxiliary helper functions.

## Usage

Run the gradient-based BFGS optimization with initial guesses for parameters `p0` and `p1`, and a regularization weight `lbd`:

```bash
python bfgs.py --p0 -3.0 --p1 -3.0 --lbd 0.005 --save (optional)
```

This will optimize the parameters based on the experimental data and store the results in the `results/` directory.

Run the gradient-free optimization methods in `benchmarks/` as a module. For example, to run CMA-ES:

```bash
python -m benchmarks.cma_es
```

## Visualization

To visualize the simulation results stored in the `results/` directory, run:

```bash
python plot.py
```

## Results
A sample training trajectory is shown in the figure below. Because of the efficiency of Differentiable Electrochemistry simulation, the optimization converged within 12 iterations. 

![parameter_and_loss_history](parameter_and_loss_history.png)]




## Benchmarks
We also provide benchmark scripts for various gradient-free optimization methods including Bayesian Optimization (BO), Covariance Matrix Adaptation Evolution Strategy (CMA-ES), Particle Swarm Optimization (PSO), and Nelder-Mead (NM) algorithm. Among all methods tested, the gradient-based BFGS optimization using JAXopt demonstrated superior performance in terms of wall-clock time and number of function evaluations to reach a target loss.

![Benchmark Results](./benchmark_results.png)