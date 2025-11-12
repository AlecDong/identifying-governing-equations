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
                 gamma_return=1.5, return_is_exponential=False,
                 N_capacity=1, random_seed=42):
        self.lambda_arrival = lambda_arrival  # arrival rate
        self.mu_service = mu_service          # service rate
        self.p_return = p_return              # probability of returning
        self.gamma_return = gamma_return      # rate or inverse of delay
        self.return_is_exponential = return_is_exponential
        self.num_servers = N_capacity
        self.random_seed = random_seed

        # Random seeds for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Tracking
        self.time_series = []
        self.x_levels = []   # main queue (server) level
        self.y_levels = []   # return queue level

        # Counters
        self.total_arrivals = 0
        self.total_services = 0
        self.total_returns = 0

    # ------------------------------------------------------------------
    # Processes
    # ------------------------------------------------------------------

    def arrival_process(self, env):
        """Generate arrivals following a Poisson process."""
        while True:
            inter_arrival = np.random.exponential(1.0 / self.lambda_arrival)
            yield env.timeout(inter_arrival)
            self.total_arrivals += 1
            env.process(self.customer(env))
            self.record_state(env)

    def customer(self, env):
        """Customer lifecycle: service + optional return."""
        with self.server.request() as req:
            yield req
            service_time = np.random.exponential(1.0 / self.mu_service)
            yield env.timeout(service_time)
            self.total_services += 1
            self.record_state(env)

            # Return with probability p_return
            if np.random.random() < self.p_return:
                self.total_returns += 1
                env.process(self.return_process(env))
                self.record_state(env)

    def return_process(self, env):
        """Fixed or exponential delay before returning to main queue."""
        # Customer enters return queue
        yield self.return_queue.put(1)
        self.record_state(env)

        # Delay before returning
        if self.return_is_exponential:
            return_time = np.random.exponential(1.0 / self.gamma_return)
        else:
            return_time = 1.0 / self.gamma_return
        yield env.timeout(return_time)

        # Customer leaves return queue
        yield self.return_queue.get()
        self.record_state(env)

        # Re-enter as a new customer
        self.total_arrivals += 1
        env.process(self.customer(env))

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------

    def simulate(self, simulation_time=20.0, initial_x=0, initial_y=0):
        """Run the full simulation."""
        # Reset all
        self.time_series.clear()
        self.x_levels.clear()
        self.y_levels.clear()

        # Initialize internal counters
        self.total_arrivals = int(initial_x)
        self.total_services = 0
        self.total_returns = int(initial_y)

        # Record initial state manually
        self.time_series.append(0.0)
        self.x_levels.append(int(initial_x))
        self.y_levels.append(int(initial_y))


        # Environment and resources
        env = simpy.Environment()
        self.env = env
        self.server = simpy.Resource(env, capacity=self.num_servers)
        self.return_queue = simpy.Store(env, capacity=float('inf'))

        # Initialize with pre-existing customers
        for _ in range(int(initial_x)):
            env.process(self.customer(env))
        for _ in range(int(initial_y)):
            env.process(self.return_process(env))

        # Start arrivals
        env.process(self.arrival_process(env))

        # Record initial state
        self.record_state(env)

        # Run
        env.run(until=simulation_time)

        return np.array(self.time_series), np.array(self.x_levels), np.array(self.y_levels)

    # ------------------------------------------------------------------
    # Data recording and postprocessing
    # ------------------------------------------------------------------

    def record_state(self, env):
        """Record current system state."""
        # Avoid duplicate timestamps
        if self.time_series and env.now == self.time_series[-1]:
            return
        self.time_series.append(env.now)
        x_level = len(self.server.queue) + len(self.server.users)
        y_level = len(self.return_queue.items)
        self.x_levels.append(x_level)
        self.y_levels.append(y_level)

    def get_data_frame(self, time_grid=None):
        """Return results as a pandas DataFrame."""
        if len(self.time_series) == 0:
            raise ValueError("Run simulate() first.")
        if time_grid is not None:
            x_interp = np.interp(time_grid, self.time_series, self.x_levels)
            y_interp = np.interp(time_grid, self.time_series, self.y_levels)
            time_data = time_grid
        else:
            x_interp = np.array(self.x_levels)
            y_interp = np.array(self.y_levels)
            time_data = np.array(self.time_series)
        return pd.DataFrame({"time": time_data, "x": x_interp, "y": y_interp})

    def plot_trajectories(self, figsize=(12, 8)):
        """Plot x(t), y(t), and phase trajectory."""
        if len(self.time_series) == 0:
            raise ValueError("No data to plot. Run simulate() first.")

        t = np.array(self.time_series)
        x = np.array(self.x_levels)
        y = np.array(self.y_levels)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        (ax1, ax2), (ax3, ax4) = axes

        # x(t)
        ax1.plot(t, x, "b-", label="x(t): Server Queue")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("x(t)")
        ax1.set_title("Server Queue Level")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # y(t)
        ax2.plot(t, y, "r-", label="y(t): Return Queue")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("y(t)")
        ax2.set_title("Return Queue Level")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Phase portrait
        ax3.plot(x, y, "g-", alpha=0.8)
        ax3.set_xlabel("x(t)")
        ax3.set_ylabel("y(t)")
        ax3.set_title("Phase Portrait")
        ax3.grid(True, alpha=0.3)

        # Combined
        ax4.plot(t, x, "b-", label="x(t)")
        ax4.plot(t, y, "r-", label="y(t)")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Queue Levels")
        ax4.set_title("Both Trajectories")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def get_statistics(self):
        """Compute simple descriptive statistics."""
        if len(self.time_series) == 0:
            return {}
        final_time = self.time_series[-1]
        x = np.array(self.x_levels)
        y = np.array(self.y_levels)
        stats = {
            "simulation_time": final_time,
            "total_arrivals": self.total_arrivals,
            "total_services": self.total_services,
            "total_returns": self.total_returns,
            "mean_x": np.mean(x),
            "mean_y": np.mean(y),
            "std_x": np.std(x),
            "std_y": np.std(y),
            "effective_arrival_rate": self.total_arrivals / final_time,
            "effective_service_rate": self.total_services / final_time,
            "effective_return_rate": self.total_returns / final_time,
        }
        return stats

    def get_interpolated_data(self, time_grid):
        """ Interpolate simulation data to a regular time grid. """
        if len(self.time_series) == 0:
            raise ValueError("No simulation data available. Run simulate() first.")
        x_interp = np.interp(time_grid, self.time_series, self.x_levels)
        y_interp = np.interp(time_grid, self.time_series, self.y_levels)
        return x_interp, y_interp
