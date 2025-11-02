# Mass transport in concentrated electrolytes

This directory contains a differentiable finite volume solver for modeling mass transport in electrolytes, implemented in Python using JAX and JAXopt.

## Requirements

- Python 3.11.5  
- JAX 0.5.3  
- JAXopt 0.8.4  


## Directory Structure

- **benchmarks/** — Scripts for benchmarking various gradient-free methods including Bayesian Optimization, Covariance Matrix Adaptation Evolution Strategy (CMA-ES), Particle Swarm Optimization (PSO), and Nelder-Mead (NM) algorithm.
- **data/** — Contains all experimental datasets.  
- **results/** — Generated output from simulations using optimized parameters.  
- **solver.py** — Script for finite volume simulation using JAX.
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

To visualize the results:

```bash
python plot.py
```
