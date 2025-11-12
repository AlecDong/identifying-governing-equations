"""
Visualization Module for Fluid Queue Analysis

This module provides comprehensive visualization tools for comparing
ODE solutions, SimPy simulations, and PySINDy discoveries.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec


class Visualizer:
    """
    Comprehensive visualization tools for fluid queue analysis.
    """
    
    def __init__(self, style='seaborn-v0_8', figsize_default=(12, 8)):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        style : str
            Matplotlib style
        figsize_default : tuple
            Default figure size
        """
        self.style = style
        self.figsize_default = figsize_default
        
        # Set style
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('default')
            
        # Set color palette
        self.colors = {
            'ode': '#1f77b4',      # Blue
            'simpy': '#ff7f0e',    # Orange  
            'sindy': '#2ca02c',    # Green
            'true': '#d62728',     # Red
            'error': '#9467bd'     # Purple
        }
    
    def plot_model_comparison(self, ode_data, simpy_data, time_grid=None, figsize=None):
        """
        Compare ODE and SimPy model outputs.
        
        Parameters:
        -----------
        ode_data : tuple
            (time_points, x_trajectory, y_trajectory) from ODE model
        simpy_data : tuple
            (time_points, x_trajectory, y_trajectory) from SimPy model
        time_grid : array, optional
            Common time grid for comparison
        figsize : tuple, optional
            Figure size
        """
        if figsize is None:
            figsize = (15, 10)
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Extract data
        t_ode, x_ode, y_ode = ode_data
        t_simpy, x_simpy, y_simpy = simpy_data
        
        # Interpolate SimPy data to common grid if provided
        if time_grid is not None:
            x_simpy_interp = np.interp(time_grid, t_simpy, x_simpy)
            y_simpy_interp = np.interp(time_grid, t_simpy, y_simpy)
            x_ode_interp = np.interp(time_grid, t_ode, x_ode)
            y_ode_interp = np.interp(time_grid, t_ode, y_ode)
        else:
            time_grid = t_ode
            x_simpy_interp = np.interp(time_grid, t_simpy, x_simpy)
            y_simpy_interp = np.interp(time_grid, t_simpy, y_simpy)
            x_ode_interp = x_ode
            y_ode_interp = y_ode
        
        # Plot x trajectories
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_grid, x_ode_interp, color=self.colors['ode'], 
                linewidth=2.5, label='ODE Model', alpha=0.9)
        ax1.plot(time_grid, x_simpy_interp, color=self.colors['simpy'], 
                linewidth=2, label='SimPy Model', alpha=0.8)
        ax1.set_title('Server Queue Level x(t)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Queue Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot y trajectories
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_grid, y_ode_interp, color=self.colors['ode'], 
                linewidth=2.5, label='ODE Model', alpha=0.9)
        ax2.plot(time_grid, y_simpy_interp, color=self.colors['simpy'], 
                linewidth=2, label='SimPy Model', alpha=0.8)
        ax2.set_title('Return Path Level y(t)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Return Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Phase portraits
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(x_ode_interp, y_ode_interp, color=self.colors['ode'], 
                linewidth=2.5, label='ODE Model', alpha=0.9)
        ax3.plot(x_simpy_interp, y_simpy_interp, color=self.colors['simpy'], 
                linewidth=2, label='SimPy Model', alpha=0.8)
        ax3.set_title('Phase Portrait', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Server Level x(t)')
        ax3.set_ylabel('Return Level y(t)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Difference plots
        diff_x = x_ode_interp - x_simpy_interp
        diff_y = y_ode_interp - y_simpy_interp
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time_grid, diff_x, color=self.colors['error'], 
                linewidth=2, label='x difference', alpha=0.8)
        ax4.plot(time_grid, diff_y, color=self.colors['error'], 
                linewidth=2, linestyle='--', label='y difference', alpha=0.8)
        ax4.set_title('Model Differences (ODE - SimPy)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Difference')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Combined trajectories
        ax5 = fig.add_subplot(gs[2, :])
        ax5.plot(time_grid, x_ode_interp, color=self.colors['ode'], 
                linewidth=2.5, label='x(t) - ODE', alpha=0.9)
        ax5.plot(time_grid, y_ode_interp, color=self.colors['ode'], 
                linewidth=2.5, linestyle=':', label='y(t) - ODE', alpha=0.9)
        ax5.plot(time_grid, x_simpy_interp, color=self.colors['simpy'], 
                linewidth=2, label='x(t) - SimPy', alpha=0.8)
        ax5.plot(time_grid, y_simpy_interp, color=self.colors['simpy'], 
                linewidth=2, linestyle=':', label='y(t) - SimPy', alpha=0.8)
        ax5.set_title('All Trajectories Comparison', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Queue Levels')
        ax5.legend(ncol=4, loc='upper right')
        ax5.grid(True, alpha=0.3)
        
        return fig
    
    def plot_sindy_comparison(self, true_solution, discovered_solution, time_points, 
                             comparison_metrics=None, figsize=None):
        """
        Plot comparison between true and SINDy-discovered systems.
        
        Parameters:
        -----------
        true_solution : ndarray
            True system solution
        discovered_solution : ndarray
            SINDy discovered system solution
        time_points : array
            Time points
        comparison_metrics : dict, optional
            Comparison metrics from equation discovery
        figsize : tuple, optional
            Figure size
        """
        if figsize is None:
            figsize = (15, 12)
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Extract trajectories
        x_true = true_solution[:, 0]
        y_true = true_solution[:, 1]
        x_disc = discovered_solution[:, 0]
        y_disc = discovered_solution[:, 1]
        
        # Plot x trajectories comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_points, x_true, color=self.colors['true'], 
                linewidth=2.5, label='True System', alpha=0.9)
        ax1.plot(time_points, x_disc, color=self.colors['sindy'], 
                linewidth=2, linestyle='--', label='SINDy Discovery', alpha=0.8)
        ax1.set_title('Server Level x(t)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('x(t)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot y trajectories comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_points, y_true, color=self.colors['true'], 
                linewidth=2.5, label='True System', alpha=0.9)
        ax2.plot(time_points, y_disc, color=self.colors['sindy'], 
                linewidth=2, linestyle='--', label='SINDy Discovery', alpha=0.8)
        ax2.set_title('Return Level y(t)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('y(t)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Phase portrait comparison
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(x_true, y_true, color=self.colors['true'], 
                linewidth=2.5, label='True System', alpha=0.9)
        ax3.plot(x_disc, y_disc, color=self.colors['sindy'], 
                linewidth=2, linestyle='--', label='SINDy Discovery', alpha=0.8)
        ax3.set_title('Phase Portrait', fontsize=12, fontweight='bold')
        ax3.set_xlabel('x(t)')
        ax3.set_ylabel('y(t)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Error analysis
        error_x = x_true - x_disc
        error_y = y_true - y_disc
        total_error = np.sqrt(error_x**2 + error_y**2)
        
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(time_points, error_x, color=self.colors['error'], linewidth=2)
        ax4.set_title('Error in x(t)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Error')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(time_points, error_y, color=self.colors['error'], linewidth=2)
        ax5.set_title('Error in y(t)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Error')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(time_points, total_error, color=self.colors['error'], linewidth=2)
        ax6.set_title('Total Error', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Time')
        ax6.set_ylabel('√(error_x² + error_y²)')
        ax6.grid(True, alpha=0.3)
        
        # Scatter plots for correlation analysis
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.scatter(x_true, x_disc, alpha=0.6, color=self.colors['sindy'])
        min_val, max_val = min(x_true.min(), x_disc.min()), max(x_true.max(), x_disc.max())
        ax7.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        ax7.set_xlabel('True x(t)')
        ax7.set_ylabel('Predicted x(t)')
        ax7.set_title('x(t) Correlation', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.scatter(y_true, y_disc, alpha=0.6, color=self.colors['sindy'])
        min_val, max_val = min(y_true.min(), y_disc.min()), max(y_true.max(), y_disc.max())
        ax8.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        ax8.set_xlabel('True y(t)')
        ax8.set_ylabel('Predicted y(t)')
        ax8.set_title('y(t) Correlation', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # Metrics display
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        if comparison_metrics is not None:
            metrics_text = f"""
Performance Metrics:

R² Score (x): {comparison_metrics.get('r2_x', 0):.4f}
R² Score (y): {comparison_metrics.get('r2_y', 0):.4f}
Average R²: {comparison_metrics.get('avg_r2', 0):.4f}

MSE (x): {comparison_metrics.get('mse_x', 0):.6f}
MSE (y): {comparison_metrics.get('mse_y', 0):.6f}
Total MSE: {comparison_metrics.get('total_mse', 0):.6f}

Max Error (x): {np.max(np.abs(error_x)):.4f}
Max Error (y): {np.max(np.abs(error_y)):.4f}
Max Total Error: {np.max(total_error):.4f}
            """
            ax9.text(0.1, 0.9, metrics_text, transform=ax9.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        return fig
    
    def plot_parameter_sensitivity(self, parameter_ranges, results_dict, figsize=None):
        """
        Plot parameter sensitivity analysis.
        
        Parameters:
        -----------
        parameter_ranges : dict
            Dictionary with parameter names as keys and ranges as values
        results_dict : dict
            Dictionary with parameter values as keys and metrics as values
        figsize : tuple, optional
            Figure size
        """
        if figsize is None:
            figsize = (15, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        param_names = list(parameter_ranges.keys())
        
        for i, param_name in enumerate(param_names[:4]):  # Plot up to 4 parameters
            if i >= len(axes):
                break
                
            param_values = parameter_ranges[param_name]
            metrics = [results_dict[val] for val in param_values]
            
            # Extract different metrics
            r2_scores = [m.get('avg_r2', 0) for m in metrics]
            mse_scores = [m.get('total_mse', 0) for m in metrics]
            
            ax = axes[i]
            ax2 = ax.twinx()
            
            # Plot R² scores
            line1 = ax.plot(param_values, r2_scores, 'b-o', linewidth=2, 
                           label='Average R²', markersize=6)
            ax.set_xlabel(param_name)
            ax.set_ylabel('Average R²', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            
            # Plot MSE scores
            line2 = ax2.plot(param_values, mse_scores, 'r-s', linewidth=2, 
                            label='Total MSE', markersize=6)
            ax2.set_ylabel('Total MSE', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            ax.set_title(f'Sensitivity to {param_name}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='best')
        
        plt.tight_layout()
        return fig
    
    def create_summary_report(self, ode_model, simpy_model, discovery_model, 
                             comparison_results, figsize=None):
        """
        Create a comprehensive summary report.
        
        Parameters:
        -----------
        ode_model : FluidQueueODE
            ODE model instance
        simpy_model : FluidQueueSimPy
            SimPy model instance  
        discovery_model : EquationDiscovery
            SINDy discovery model instance
        comparison_results : dict
            Comparison results
        figsize : tuple, optional
            Figure size
        """
        if figsize is None:
            figsize = (16, 20)
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle('Fluid Queue System Analysis Summary', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # System parameters display
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        params_text = f"""
System Parameters:
• Arrival Rate (λ): {ode_model.lambda_arrival:.2f}
• Service Rate (μ): {ode_model.mu_service:.2f}  
• Return Probability (p): {ode_model.p_return:.2f}
• Return Rate (γ): {ode_model.gamma_return:.2f}
• Service Capacity (N): {ode_model.N_capacity:.2f}

        """
        ax1.text(0.5, 0.5, params_text, transform=ax1.transAxes, 
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Model equations
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('off')
        equations_text = """
True Governing Equations:

dx/dt = λ - μ·min(x,N) + γ·y
dy/dt = p·μ·min(x,N) - γ·y

Where:
• x(t): Server queue level
• y(t): Return path level
        """
        ax2.text(0.1, 0.9, equations_text, transform=ax2.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        # Discovered equations
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        if discovery_model.discovered_equations:
            discovered_text = f"""
SINDy Discovered Equations:

{discovery_model.discovered_equations[0]}

{discovery_model.discovered_equations[1]}

Model Complexity: {np.sum(np.abs(discovery_model.coefficients) > discovery_model.threshold)} terms
            """
        else:
            discovered_text = "SINDy equations not available"
        
        ax3.text(0.1, 0.9, discovered_text, transform=ax3.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # Performance metrics
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        if comparison_results:
            metrics_text = f"""
Performance Assessment:

Equation Discovery Quality:
• Average R² Score: {comparison_results.get('avg_r2', 0):.4f}
• Total Mean Squared Error: {comparison_results.get('total_mse', 0):.6f}
• R² Score for x(t): {comparison_results.get('r2_x', 0):.4f}
• R² Score for y(t): {comparison_results.get('r2_y', 0):.4f}

Model Interpretation:
• R² > 0.95: Excellent fit
• R² > 0.8: Good fit  
• R² > 0.6: Moderate fit
• R² < 0.6: Poor fit
            """
            ax4.text(0.5, 0.7, metrics_text, transform=ax4.transAxes, 
                    fontsize=12, ha='center', va='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
        
        return fig
