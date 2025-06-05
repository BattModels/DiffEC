# Differentiable mass transport in electrolytes

This directory contains a differentiable finite volume solver for modeling mass transport in electrolytes, implemented in Python using JAX and JAXopt.

## Requirements

- Python 3.11.5  
- JAX 0.5.3  
- JAXopt 0.8.4  


## Directory Structure

- **data/** — Contains all experimental datasets.  
- **results/** — Generated output from simulations using optimized parameters.  
- **solver.py** — Main script for parameter optimization and simulation.  
- **plot.py** — Script for visualizing simulation results.  
- **utils.py** — Auxiliary helper functions.  

## Usage

Run the solver with initial guesses for parameters `p0` and `p1`, and a regularization weight `lbd`:

```bash
python solver.py --p0 -3.0 --p1 -3.0 --lbd 0.005
```


This will optimize the parameters based on the experimental data and store the results in the `results/` directory.

## Visualization

To visualize the results:

```bash
python plot.py
```