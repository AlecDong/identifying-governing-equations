# Fluid Queue Simulation with Governing Equation Discovery

This repository implements a comprehensive Python project for simulating fluid queue systems using two different approaches and discovering their governing equations using machine learning.

## Overview

The project implements two models for fluid queue systems:

1. **Model 1 (ODE)**: Deterministic ordinary differential equation model using scipy
2. **Model 2 (SimPy)**: Stochastic discrete-event simulation using SimPy

Both models simulate a fluid queue with:
- Server queue level x(t)
- Return path level y(t)  
- Parameters: λ (arrival rate), μ (service rate), p (return probability), γ (return rate), N (max capacity)

The system is governed by the ODEs:
- x'(t) = λ - μ·min(x,N) + γ·y
- y'(t) = p·μ·min(x,N) - γ·y

## Key Features

- **ODE Simulation**: Solve the governing differential equations numerically
- **Stochastic Simulation**: Generate realistic sample paths with random events
- **Equation Discovery**: Use PySINDy to learn governing equations from data
- **Comprehensive Visualization**: Compare models and analyze discovered equations
- **Performance Analysis**: Quantitative metrics for equation discovery quality

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AlecDong/identifying-governing-equations.git
cd identifying-governing-equations
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the complete analysis pipeline:

```bash
python main.py
```

This will:
1. Simulate the ODE model
2. Run the SimPy stochastic simulation
3. Apply PySINDy to discover equations
4. Generate comparison plots and save results

## Output Files

The simulation generates several output files in the `results/` directory:

- `ode_data.csv`: ODE simulation time-series data
- `simpy_data.csv`: SimPy simulation time-series data
- `equation_discovery_results.csv`: Comparison of true vs discovered systems
- `discovered_equations.txt`: Text file with discovered equations and metrics
- `model_comparison.png`: Visualization comparing ODE and SimPy models
- `sindy_comparison.png`: Visualization comparing true and discovered equations
- `summary_report.png`: Comprehensive analysis summary

## Mathematical Background

The fluid queue system models a service system where:

- **x(t)**: Level of fluid in the main server queue
- **y(t)**: Level of fluid in the return path
- **λ**: Rate at which fluid arrives to the system
- **μ**: Maximum service rate 
- **N**: Service capacity constraint
- **p**: Probability that served fluid goes to return path
- **γ**: Rate at which fluid returns from return path to main queue

The dynamics are governed by:
```
dx/dt = λ - μ·min(x(t),N) + γ·y(t)
dy/dt = p·μ·min(x(t),N) - γ·y(t)
```

## Dependencies

- `numpy>=1.21.0`: Numerical computations
- `scipy>=1.7.0`: ODE solving
- `matplotlib>=3.5.0`: Plotting  
- `simpy>=4.0.0`: Discrete-event simulation
- `pysindy>=1.7.0`: Equation discovery
- `pandas>=1.3.0`: Data handling
- `seaborn>=0.11.0`: Enhanced plotting
