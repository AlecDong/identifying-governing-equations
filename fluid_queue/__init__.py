"""
Fluid Queue Simulation with Governing Equation Discovery

This package implements two models for simulating fluid queue systems:
1. ODE-based deterministic model using scipy
2. Stochastic discrete-event simulation using SimPy

The package also includes PySINDy integration to discover governing equations
from the generated time-series data and compare them with the true equations.
"""

__version__ = "1.0.0"
__author__ = "Fluid Queue Simulation Project"

# Import main modules - with error handling for dependencies
try:
    from .fluid_queue_ode import FluidQueueODE
    from .fluid_queue_simpy import FluidQueueSimPy
    from .equation_discovery import EquationDiscovery
    from .visualizer import Visualizer
    
    __all__ = [
        'FluidQueueODE',
        'FluidQueueSimPy', 
        'EquationDiscovery',
        'Visualizer'
    ]
except ImportError as e:
    print(f"Warning: Could not import all modules. Please install required dependencies: {e}")
    __all__ = []