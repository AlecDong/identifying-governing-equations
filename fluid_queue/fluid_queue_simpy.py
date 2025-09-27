"""
Model 2: Stochastic Fluid Queue Simulation using SimPy

This module implements a discrete-event simulation of the fluid queue system
using SimPy to generate stochastic sample paths that can be compared with
the deterministic ODE model.
"""

import simpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import random


class FluidQueueSimPy:
    """
    SimPy-based stochastic fluid queue simulator.
    
    This class implements a discrete-event simulation of the fluid queue
    system with arrivals, service, and returns modeled as stochastic processes.
    """
    
    def __init__(self, lambda_arrival=2.0, mu_service=3.0, p_return=0.3,
                 gamma_return=1.5, N_capacity=10.0, random_seed=42):
        """
        Initialize the stochastic fluid queue parameters.
        
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
        random_seed : int
            Random seed for reproducibility
        """
        self.lambda_arrival = lambda_arrival
        self.mu_service = mu_service
        self.p_return = p_return
        self.gamma_return = gamma_return
        self.N_capacity = N_capacity
        self.random_seed = random_seed
        
        # Initialize random number generator
        self.rng = np.random.RandomState(random_seed)
        random.seed(random_seed)
        
        # Simulation state tracking
        self.time_series = []
        self.x_levels = []
        self.y_levels = []
        
        # Current levels
        self.current_x = 0.0
        self.current_y = 0.0
        
        # Statistics tracking
        self.total_arrivals = 0
        self.total_services = 0
        self.total_returns = 0
        
    def arrival_process(self, env):
        """
        Generate arrival events following Poisson process.
        
        Parameters:
        -----------
        env : simpy.Environment
            SimPy environment
        """
        while True:
            # Exponential inter-arrival times
            inter_arrival_time = self.rng.exponential(1.0 / self.lambda_arrival)
            yield env.timeout(inter_arrival_time)
            
            # Add arrival to server queue
            arrival_amount = 1.0  # Unit arrivals
            self.current_x += arrival_amount
            self.total_arrivals += 1
            
            # Record state
            self.record_state(env.now)
    
    def service_process(self, env):
        """
        Generate service events.
        
        Parameters:
        -----------
        env : simpy.Environment
            SimPy environment
        """
        while True:
            if self.current_x > 0:
                # Service rate depends on min(x, N)
                effective_capacity = min(self.current_x, self.N_capacity)
                service_rate = self.mu_service * (effective_capacity / self.N_capacity)
                
                if service_rate > 0:
                    # Exponential service times
                    service_time = self.rng.exponential(1.0 / service_rate)
                    yield env.timeout(service_time)
                    
                    # Service amount proportional to effective capacity
                    service_amount = min(1.0, self.current_x)
                    self.current_x = max(0, self.current_x - service_amount)
                    self.total_services += 1
                    
                    # Determine if service goes to return path
                    if self.rng.random() < self.p_return:
                        self.current_y += service_amount
                        self.total_returns += 1
                    
                    # Record state
                    self.record_state(env.now)
                else:
                    # Wait a bit if no service possible
                    yield env.timeout(0.1)
            else:
                # Wait for arrivals if server empty
                yield env.timeout(0.1)
    
    def return_process(self, env):
        """
        Generate return events from return path.
        
        Parameters:
        -----------
        env : simpy.Environment
            SimPy environment
        """
        while True:
            if self.current_y > 0:
                # Return rate
                return_rate = self.gamma_return * self.current_y
                
                if return_rate > 0:
                    # Exponential return times
                    return_time = self.rng.exponential(1.0 / return_rate)
                    yield env.timeout(return_time)
                    
                    # Return amount
                    return_amount = min(0.5, self.current_y)  # Smaller return amounts
                    self.current_y = max(0, self.current_y - return_amount)
                    
                    # Record state
                    self.record_state(env.now)
                else:
                    yield env.timeout(0.1)
            else:
                # Wait if return path empty
                yield env.timeout(0.1)
    
    def record_state(self, time):
        """
        Record current system state.
        
        Parameters:
        -----------
        time : float
            Current simulation time
        """
        self.time_series.append(time)
        self.x_levels.append(self.current_x)
        self.y_levels.append(self.current_y)
    
    def simulate(self, simulation_time=20.0, initial_x=5.0, initial_y=2.0):
        """
        Run the stochastic simulation.
        
        Parameters:
        -----------
        simulation_time : float
            Total simulation time
        initial_x : float
            Initial server queue level
        initial_y : float
            Initial return path level
            
        Returns:
        --------
        time_points : array
            Time points
        x_trajectory : array
            Server queue levels
        y_trajectory : array
            Return path levels
        """
        # Reset simulation state
        self.time_series = []
        self.x_levels = []
        self.y_levels = []
        self.current_x = initial_x
        self.current_y = initial_y
        self.total_arrivals = 0
        self.total_services = 0
        self.total_returns = 0
        
        # Create SimPy environment
        env = simpy.Environment()
        
        # Record initial state
        self.record_state(0.0)
        
        # Start processes
        env.process(self.arrival_process(env))
        env.process(self.service_process(env))
        env.process(self.return_process(env))
        
        # Run simulation
        env.run(until=simulation_time)
        
        return np.array(self.time_series), np.array(self.x_levels), np.array(self.y_levels)
    
    def get_interpolated_data(self, time_grid):
        """
        Interpolate simulation data to regular time grid.
        
        Parameters:
        -----------
        time_grid : array
            Regular time grid for interpolation
            
        Returns:
        --------
        x_interp : array
            Interpolated server levels
        y_interp : array
            Interpolated return levels
        """
        if len(self.time_series) == 0:
            raise ValueError("No simulation data available. Run simulate() first.")
        
        x_interp = np.interp(time_grid, self.time_series, self.x_levels)
        y_interp = np.interp(time_grid, self.time_series, self.y_levels)
        
        return x_interp, y_interp
    
    def get_data_frame(self, time_grid=None):
        """
        Get simulation results as a pandas DataFrame.
        
        Parameters:
        -----------
        time_grid : array, optional
            Regular time grid. If None, uses simulation time points.
            
        Returns:
        --------
        df : pandas.DataFrame
            DataFrame with columns: time, x, y
        """
        if len(self.time_series) == 0:
            raise ValueError("No simulation data available. Run simulate() first.")
        
        if time_grid is not None:
            x_data, y_data = self.get_interpolated_data(time_grid)
            time_data = time_grid
        else:
            time_data = np.array(self.time_series)
            x_data = np.array(self.x_levels)
            y_data = np.array(self.y_levels)
        
        df = pd.DataFrame({
            'time': time_data,
            'x': x_data,
            'y': y_data
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
        if len(self.time_series) == 0:
            raise ValueError("No simulation data available. Run simulate() first.")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        time_array = np.array(self.time_series)
        x_array = np.array(self.x_levels)
        y_array = np.array(self.y_levels)
        
        # Plot x(t) - server queue level
        ax1.plot(time_array, x_array, 'b-', linewidth=1.5, alpha=0.8, label='x(t) - Server Level')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Server Queue Level')
        ax1.set_title('Server Queue Level x(t) - Stochastic')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot y(t) - return path level
        ax2.plot(time_array, y_array, 'r-', linewidth=1.5, alpha=0.8, label='y(t) - Return Level')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Return Path Level')
        ax2.set_title('Return Path Level y(t) - Stochastic')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Phase portrait
        ax3.plot(x_array, y_array, 'g-', linewidth=1.5, alpha=0.8)
        ax3.plot(x_array[0], y_array[0], 'go', markersize=8, label='Initial')
        ax3.plot(x_array[-1], y_array[-1], 'ro', markersize=8, label='Final')
        ax3.set_xlabel('x(t) - Server Level')
        ax3.set_ylabel('y(t) - Return Level')
        ax3.set_title('Phase Portrait - Stochastic')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Both trajectories on same plot
        ax4.plot(time_array, x_array, 'b-', linewidth=1.5, alpha=0.8, label='x(t) - Server')
        ax4.plot(time_array, y_array, 'r-', linewidth=1.5, alpha=0.8, label='y(t) - Return')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Queue Levels')
        ax4.set_title('Both Trajectories - Stochastic')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def get_statistics(self):
        """
        Get simulation statistics.
        
        Returns:
        --------
        stats : dict
            Dictionary containing simulation statistics
        """
        if len(self.time_series) == 0:
            return {}
        
        final_time = self.time_series[-1] if self.time_series else 0
        x_array = np.array(self.x_levels)
        y_array = np.array(self.y_levels)
        
        stats = {
            'simulation_time': final_time,
            'total_arrivals': self.total_arrivals,
            'total_services': self.total_services,
            'total_returns': self.total_returns,
            'final_x': self.x_levels[-1] if self.x_levels else 0,
            'final_y': self.y_levels[-1] if self.y_levels else 0,
            'mean_x': np.mean(x_array) if len(x_array) > 0 else 0,
            'mean_y': np.mean(y_array) if len(y_array) > 0 else 0,
            'std_x': np.std(x_array) if len(x_array) > 0 else 0,
            'std_y': np.std(y_array) if len(y_array) > 0 else 0,
            'effective_arrival_rate': self.total_arrivals / final_time if final_time > 0 else 0,
            'effective_service_rate': self.total_services / final_time if final_time > 0 else 0,
            'effective_return_rate': self.total_returns / final_time if final_time > 0 else 0
        }
        
        return stats