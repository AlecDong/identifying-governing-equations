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

3. Install the package (optional):
```bash
pip install -e .
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

## Usage Examples

### Basic ODE Simulation

```python
from fluid_queue import FluidQueueODE

# Initialize model
model = FluidQueueODE(lambda_arrival=2.0, mu_service=3.0, 
                      p_return=0.3, gamma_return=1.5, N_capacity=10.0)

# Simulate
time_points, solution = model.simulate(t_span=(0, 20), 
                                     initial_state=(5.0, 2.0), 
                                     num_points=1000)

# Plot results
fig = model.plot_trajectories()
```

### Stochastic SimPy Simulation

```python
from fluid_queue import FluidQueueSimPy

# Initialize stochastic model
model = FluidQueueSimPy(lambda_arrival=2.0, mu_service=3.0,
                        p_return=0.3, gamma_return=1.5, N_capacity=10.0)

# Simulate
t_simpy, x_simpy, y_simpy = model.simulate(simulation_time=20.0,
                                          initial_x=5.0, initial_y=2.0)

# Get statistics
stats = model.get_statistics()
```

### Equation Discovery with PySINDy

```python
from fluid_queue import EquationDiscovery

# Initialize equation discovery
discovery = EquationDiscovery(threshold=0.01, alpha=0.05)

# Prepare data from ODE simulation
X, X_dot, t = discovery.prepare_data(ode_data_frame)

# Fit SINDy model
sindy_model = discovery.fit_sindy_model(X, X_dot, t)

# Print discovered equations
discovery.print_discovered_equations()

# Compare with true system
comparison = discovery.compare_with_true_system(parameters, initial_state, time_points)
```

## Command Line Options

```bash
# Display plots interactively
python main.py --show-plots

# Skip saving plots to files  
python main.py --no-save

# Run parameter sensitivity study
python main.py --parameter-study
```

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

## Project Structure

```
identifying-governing-equations/
├── fluid_queue/                    # Main package
│   ├── __init__.py                # Package initialization
│   ├── fluid_queue_ode.py         # ODE model implementation
│   ├── fluid_queue_simpy.py       # SimPy stochastic model
│   ├── equation_discovery.py      # PySINDy integration
│   └── visualizer.py              # Visualization tools
├── main.py                        # Main script
├── requirements.txt               # Dependencies
├── setup.py                       # Package setup
└── README.md                      # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.