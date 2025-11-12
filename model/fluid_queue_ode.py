"""
Model 1: ODE-based Fluid Queue Simulation

This module implements the deterministic fluid queue model using ordinary differential equations:
- x'(t) = λ - μ*min(x,N) + γ*y  (server queue level)
- y'(t) = p*μ*min(x,N) - γ*y    (return path level)

Where:
- λ: arrival rate
- μ: service rate  
- p: return probability
- γ: return rate
- N: maximum service capacity
- x(t): server queue level
- y(t): return path level
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd


class FluidQueueODE:
    """
    ODE-based fluid queue simulator.
    
    This class solves the system of ODEs that govern the fluid queue dynamics
    and generates time-series data for analysis.
    """
    
    def __init__(self, lambda_arrival=2.0, mu_service=3.0, p_return=0.3, 
                 gamma_return=1.5, N_capacity=10.0):
        """
        Initialize the fluid queue parameters.
        
        Parameters:
        -----------
        lambda_arrival : float
            Arrival rate (λ)
        mu_service : float
            Service rate (μ)
        p_return : float
            Return probability (p), should be between 0 and 1
        gamma_return : float
            Return rate (γ)
        N_capacity : float
            Maximum service capacity (N)
        """
        self.lambda_arrival = lambda_arrival
        self.mu_service = mu_service
        self.p_return = p_return
        self.gamma_return = gamma_return
        self.N_capacity = N_capacity
        
        # Store simulation results
        self.time_points = None
        self.solution = None
        self.x_trajectory = None
        self.y_trajectory = None
        
    def system_dynamics(self, state, t):
        """
        Define the system of ODEs.
        
        Parameters:
        -----------
        state : array_like
            Current state [x, y] where x is server level, y is return level
        t : float
            Current time
            
        Returns:
        --------
        derivatives : array_like
            [x'(t), y'(t)] - derivatives of the state variables
        """
        x, y = state
        
        # Ensure non-negative queue levels
        x = max(0, x)
        y = max(0, y)
        
        # Calculate service rate limited by capacity and current server level
        effective_service = self.mu_service * min(x, self.N_capacity)
        
        # System of ODEs
        dx_dt = self.lambda_arrival - effective_service + self.gamma_return * y
        dy_dt = self.p_return * effective_service - self.gamma_return * y
        
        return [dx_dt, dy_dt]
    
    def simulate(self, t_span=(0, 20), initial_state=(5.0, 2.0), num_points=1000):
        """
        Simulate the fluid queue system.
        
        Parameters:
        -----------
        t_span : tuple
            Time span for simulation (start_time, end_time)
        initial_state : tuple
            Initial conditions [x(0), y(0)]
        num_points : int
            Number of time points to generate
            
        Returns:
        --------
        time_points : ndarray
            Array of time points
        solution : ndarray
            Solution array with shape (num_points, 2) containing [x(t), y(t)]
        """
        # Generate time points
        self.time_points = np.linspace(t_span[0], t_span[1], num_points)
        
        # Solve the ODE system
        self.solution = odeint(self.system_dynamics, initial_state, self.time_points)
        
        # Extract individual trajectories
        self.x_trajectory = self.solution[:, 0]
        self.y_trajectory = self.solution[:, 1]
        
        return self.time_points, self.solution
    
    def get_data_frame(self):
        """
        Get simulation results as a pandas DataFrame.
        
        Returns:
        --------
        df : pandas.DataFrame
            DataFrame with columns: time, x, y, dx_dt, dy_dt
        """
        if self.solution is None:
            raise ValueError("No simulation data available. Run simulate() first.")
        
        # Calculate derivatives at each point
        derivatives = np.array([self.system_dynamics(state, t) 
                              for state, t in zip(self.solution, self.time_points)])
        
        df = pd.DataFrame({
            'time': self.time_points,
            'x': self.x_trajectory,
            'y': self.y_trajectory,
            'dx_dt': derivatives[:, 0],
            'dy_dt': derivatives[:, 1]
        })
        
        return df
    
    def plot_trajectories(self, figsize=(12, 8)):
        """
        Plot the simulation results.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if self.solution is None:
            raise ValueError("No simulation data available. Run simulate() first.")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot x(t) - server queue level
        ax1.plot(self.time_points, self.x_trajectory, 'b-', linewidth=2, label='x(t) - Server Level')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Server Queue Level')
        ax1.set_title('Server Queue Level x(t)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot y(t) - return path level
        ax2.plot(self.time_points, self.y_trajectory, 'r-', linewidth=2, label='y(t) - Return Level')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Return Path Level')
        ax2.set_title('Return Path Level y(t)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Phase portrait
        ax3.plot(self.x_trajectory, self.y_trajectory, 'g-', linewidth=2)
        ax3.plot(self.x_trajectory[0], self.y_trajectory[0], 'go', markersize=8, label='Initial')
        ax3.plot(self.x_trajectory[-1], self.y_trajectory[-1], 'ro', markersize=8, label='Final')
        ax3.set_xlabel('x(t) - Server Level')
        ax3.set_ylabel('y(t) - Return Level')
        ax3.set_title('Phase Portrait')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Both trajectories on same plot
        ax4.plot(self.time_points, self.x_trajectory, 'b-', linewidth=2, label='x(t) - Server')
        ax4.plot(self.time_points, self.y_trajectory, 'r-', linewidth=2, label='y(t) - Return')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Queue Levels')
        ax4.set_title('Both Trajectories')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        return fig
