#!/usr/bin/env python3
"""
Main Script for Fluid Queue Simulation and Equation Discovery

This script demonstrates the complete pipeline:
1. Simulate fluid queue using ODE model
2. Simulate fluid queue using SimPy stochastic model
3. Apply PySINDy to discover governing equations
4. Compare and visualize results
"""

import pysindy as ps
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from fluid_queue.fluid_queue_ode import FluidQueueODE
from fluid_queue.fluid_queue_simpy import FluidQueueSimPy
from fluid_queue.equation_discovery import EquationDiscovery
from fluid_queue.visualizer import Visualizer
from fluid_queue.custom_equation_libraries import min_library


def run_complete_analysis(params, save_plots=True, show_plots=False):
    """
    Run the complete fluid queue analysis pipeline.
    
    Parameters:
    -----------
    save_plots : bool
        Whether to save plots to files
    show_plots : bool
        Whether to display plots interactively
    """
    print("="*60)
    print("FLUID QUEUE SIMULATION AND EQUATION DISCOVERY")
    print("="*60)
    
    # Simulation parameters
    t_span = (0, 10)
    initial_state = (0.0, 0.0)
    num_points = 2000
    
    print(f"System Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"Initial State: x(0) = {initial_state[0]}, y(0) = {initial_state[1]}")
    print(f"Time Span: {t_span}")
    print()
    
    # Initialize visualizer
    viz = Visualizer()
    
    # Create output directory
    output_dir = Path("results/lambda_{}__N_{}".format(
        params['lambda_arrival'], params['N_capacity']
    ))
    output_dir.mkdir(exist_ok=True)
    
    # ========================================
    # STEP 1: ODE Model Simulation
    # ========================================
    print("STEP 1: Running ODE Model Simulation...")
    
    ode_model = FluidQueueODE(**params)
    t_ode, solution_ode = ode_model.simulate(t_span, initial_state, num_points)
    
    x_ode = solution_ode[:, 0]
    y_ode = solution_ode[:, 1]
    
    print(f"  ✓ ODE simulation completed")
    print(f"  ✓ Generated {len(t_ode)} time points")
    print(f"  ✓ Initial state: x = {x_ode[0]:.4f}, y = {y_ode[0]:.4f}")
    print(f"  ✓ Final state: x = {x_ode[-1]:.4f}, y = {y_ode[-1]:.4f}")
    
    
    # ========================================
    # STEP 2: SimPy Stochastic Simulation
    # ========================================
    print("STEP 2: Running SimPy Stochastic Simulation...")
    
    simpy_model = FluidQueueSimPy(**params, random_seed=42)
    t_simpy, x_simpy, y_simpy = simpy_model.simulate(
        simulation_time=t_span[1], 
        initial_x=initial_state[0], 
        initial_y=initial_state[1]
    )
    
    print(f"  ✓ SimPy simulation completed")
    print(f"  ✓ Generated {len(t_simpy)} events")
    print(f"  ✓ Initial state: x = {x_simpy[0]:.4f}, y = {y_simpy[0]:.4f}")
    print(f"  ✓ Final state: x = {x_simpy[-1]:.4f}, y = {y_simpy[-1]:.4f}")
    
    # Get simulation statistics
    stats = simpy_model.get_statistics()
    print(f"  ✓ Total arrivals: {stats['total_arrivals']}")
    print(f"  ✓ Total services: {stats['total_services']}")
    print(f"  ✓ Total returns: {stats['total_returns']}")
    print()
    
    # ========================================
    # STEP 3: Model Comparison
    # ========================================
    print("STEP 3: Comparing ODE and SimPy Models...")
    
    # Create common time grid for comparison
    time_grid = np.linspace(t_span[0], t_span[1], num_points)
    
    # Interpolate SimPy data to common grid
    x_simpy_interp, y_simpy_interp = simpy_model.get_interpolated_data(time_grid)
    x_ode_interp = np.interp(time_grid, t_ode, x_ode)
    y_ode_interp = np.interp(time_grid, t_ode, y_ode)
    
    # Calculate comparison metrics
    mse_x = np.mean((x_ode_interp - x_simpy_interp)**2)
    mse_y = np.mean((y_ode_interp - y_simpy_interp)**2)
    
    print(f"  ✓ MSE between models (x): {mse_x:.6f}")
    print(f"  ✓ MSE between models (y): {mse_y:.6f}")
    print()
    
    # Plot model comparison
    fig1 = viz.plot_model_comparison(
        (t_ode, x_ode, y_ode),
        (t_simpy, x_simpy, y_simpy),
        time_grid=time_grid
    )
    
    if save_plots:
        fig1.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        print("  ✓ Model comparison plot saved")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig1)
    
    # ========================================
    # STEP 4: Equation Discovery with PySINDy
    # ========================================
    print("STEP 4: Applying PySINDy for Equation Discovery...")
    
    # Prepare data for SINDy (using ODE data for cleaner derivatives)
    ode_df = ode_model.get_data_frame()
    
    discovery = EquationDiscovery(threshold=0.1, alpha=0.05, max_iter=20)
    
    # Prepare training data
    print("  ✓ Preparing training data...")
    X_train, X_dot_train, t_train = discovery.prepare_data(ode_df)
    
    print(f"  ✓ Training data shape: {X_train.shape}")
    print(f"  ✓ Derivatives shape: {X_dot_train.shape}")
    
    # Fit SINDy model
    print("  ✓ Fitting SINDy model...")
    try:
        poly_lib = ps.PolynomialLibrary(degree=2, include_bias=True, include_interaction=True)
        queue_lib = min_library(N=params['N_capacity'])

        combined_lib = ps.GeneralizedLibrary([poly_lib, queue_lib])
        sindy_model = discovery.fit_sindy_model(X_train, X_dot_train, t_train, combined_lib)
        print("  ✓ SINDy model fitted successfully")
        
        # Print discovered equations
        print("\n  Discovered Equations:")
        print("  " + "="*40)
        for i, eq in enumerate(discovery.discovered_equations):
            var_name = 'x' if i == 0 else 'y'
            print(f"  d{var_name}/dt = {eq}")
        print()
        
    except Exception as e:
        print(f"  ⚠ SINDy fitting failed: {str(e)}")
        discovery = None
        sindy_model = None

    # ========================================
    # STEP 4b: Equation Discovery with PySINDy on SimPy data
    # ========================================
    print("STEP 4b: Applying PySINDy to SimPy data...")

    # Prepare data for SINDy (SimPy data)
    simpy_df = simpy_model.get_data_frame(time_grid)

    discovery_simpy = EquationDiscovery(threshold=0.1, alpha=0.05, max_iter=20)

    X_train_simpy, X_dot_train_simpy, t_train_simpy = discovery_simpy.prepare_data(simpy_df)

    print(f"  ✓ Training data shape (SimPy): {X_train_simpy.shape}")
    print(f"  ✓ Derivatives shape (SimPy): {X_dot_train_simpy.shape}")

    # Fit SINDy model on SimPy data
    print("  ✓ Fitting SINDy model on SimPy data...")
    try:
        poly_lib = ps.PolynomialLibrary(degree=1, include_bias=True, include_interaction=True)
        queue_lib = min_library(N=params['N_capacity'])

        combined_lib = ps.GeneralizedLibrary([poly_lib, queue_lib])
        simpy_sindy_model = discovery_simpy.fit_sindy_model(X_train_simpy, X_dot_train_simpy, t_train_simpy, combined_lib)

        print("  ✓ SINDy model fitted successfully on SimPy data")

        # Extract discovered equations
        print("\n  Discovered Equations (SimPy):")
        for i, eq in enumerate(discovery_simpy.discovered_equations):
            var_name = 'x' if i == 0 else 'y'
            print(f"  d{var_name}/dt = {eq}")
        print()

    except Exception as e:
        print(f"  ⚠ SINDy fitting failed on SimPy data: {str(e)}")
        discovery_simpy = None
        simpy_sindy_model = None
    
    # ========================================
    # STEP 5: Compare True vs Discovered Systems
    # ========================================
    if discovery is not None:
        print("STEP 5: Comparing True vs Discovered Systems...")
        
        # Compare systems
        comparison_results = discovery.compare_with_true_system(
            solution_ode, initial_state, t_ode
        )
        
        print(f"  ✓ R² Score (x): {comparison_results['r2_x']:.4f}")
        print(f"  ✓ R² Score (y): {comparison_results['r2_y']:.4f}")
        print(f"  ✓ Average R²: {comparison_results['avg_r2']:.4f}")
        print(f"  ✓ Total MSE: {comparison_results['total_mse']:.6f}")
        print()
        
        # Plot SINDy comparison
        fig2 = viz.plot_sindy_comparison(
            comparison_results['true_solution'],
            comparison_results['discovered_solution'],
            comparison_results['time'],
            comparison_results
        )
        
        if save_plots:
            fig2.savefig(output_dir / "sindy_comparison.png", dpi=300, bbox_inches='tight')
            print("  ✓ SINDy comparison plot saved")
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig2)
    
    # ========================================
    # STEP 5b: Compare True vs Discovered System (SimPy SINDy)
    # ========================================
    if simpy_sindy_model is not None:
        print("STEP 5b: Comparing True vs SimPy-Discovered Systems...")

        # Compare systems
        comparison_results_simpy = discovery_simpy.compare_with_true_system(
            solution_ode, initial_state, t_ode
        )

        print(f"  ✓ R² Score (x) (SimPy): {comparison_results_simpy['r2_x']:.4f}")
        print(f"  ✓ R² Score (y) (SimPy): {comparison_results_simpy['r2_y']:.4f}")
        print(f"  ✓ Average R² (SimPy): {comparison_results_simpy['avg_r2']:.4f}")
        print(f"  ✓ Total MSE (SimPy): {comparison_results_simpy['total_mse']:.6f}")
        print()

        # Plot SINDy comparison
        fig2b = viz.plot_sindy_comparison(
            comparison_results_simpy['true_solution'],
            comparison_results_simpy['discovered_solution'],
            comparison_results_simpy['time'],
            comparison_results_simpy
        )

        if save_plots:
            fig2b.savefig(output_dir / "sindy_comparison_simpy.png", dpi=300, bbox_inches='tight')
            print("  ✓ SINDy comparison plot (SimPy) saved")

        if show_plots:
            plt.show()
        else:
            plt.close(fig2b)
    
    # ========================================
    # STEP 5c: Compare SimPy training data vs Discovered System (SimPy SINDy)
    # ========================================
    if simpy_sindy_model is not None:
        print("STEP 5c: Comparing SimPy Training Data vs SimPy-Discovered Systems...")

        # Compare systems
        comparison_results_simpy_train = discovery_simpy.compare_with_training_data(
            X_train_simpy, t_train_simpy, t_ode
        )

        print(f"  ✓ R² Score (x) (SimPy Train): {comparison_results_simpy_train['r2_x']:.4f}")
        print(f"  ✓ R² Score (y) (SimPy Train): {comparison_results_simpy_train['r2_y']:.4f}")
        print(f"  ✓ Average R² (SimPy Train): {comparison_results_simpy_train['avg_r2']:.4f}")
        print(f"  ✓ Total MSE (SimPy Train): {comparison_results_simpy_train['total_mse']:.6f}")
        print()

        # Plot SINDy comparison
        fig2c = viz.plot_sindy_comparison(
            X_train_simpy,
            comparison_results_simpy_train['discovered_solution'],
            comparison_results_simpy_train['time'],
            comparison_results_simpy_train
        )

        if save_plots:
            fig2c.savefig(output_dir / "sindy_comparison_simpy_train.png", dpi=300, bbox_inches='tight')
            print("  ✓ SINDy comparison plot (SimPy Training Data) saved")

        if show_plots:
            plt.show()
        else:
            plt.close(fig2c)

    # ========================================
    # STEP 6: Create Summary Report
    # ========================================
    print("STEP 6: Creating Summary Report...")
    
    if discovery is not None:
        fig3 = viz.create_summary_report(
            ode_model, simpy_model, discovery, 
            comparison_results if discovery else None
        )
        
        if save_plots:
            fig3.savefig(output_dir / "summary_report.png", dpi=300, bbox_inches='tight')
            print("  ✓ Summary report saved")
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig3)
    
    print("STEP 6b: Creating Summary Report for SimPy SINDy...")
    if discovery_simpy is not None:
        fig3b = viz.create_summary_report(
            ode_model, simpy_model, discovery_simpy, 
            comparison_results_simpy if discovery_simpy else None
        )

        if save_plots:
            fig3b.savefig(output_dir / "summary_report_simpy.png", dpi=300, bbox_inches='tight')
            print("  ✓ Summary report (SimPy SINDy) saved")

        if show_plots:
            plt.show()
        else:
            plt.close(fig3b)
    
    # ========================================
    # STEP 7: Save Data to Files
    # ========================================
    print("STEP 7: Saving Data to Files...")
    
    # Save ODE data
    ode_df.to_csv(output_dir / "ode_data.csv", index=False)
    print("  ✓ ODE data saved to ode_data.csv")
    
    # Save SimPy data
    simpy_df = simpy_model.get_data_frame(time_grid)
    simpy_df.to_csv(output_dir / "simpy_data.csv", index=False)
    print("  ✓ SimPy data saved to simpy_data.csv")
    
    # Save comparison results
    if discovery is not None:
        comparison_df = pd.DataFrame({
            'time': comparison_results['time'],
            'true_x': comparison_results['true_solution'][:, 0],
            'true_y': comparison_results['true_solution'][:, 1],
            'discovered_x': comparison_results['discovered_solution'][:, 0],
            'discovered_y': comparison_results['discovered_solution'][:, 1]
        })
        comparison_df.to_csv(output_dir / "equation_discovery_results.csv", index=False)
        print("  ✓ Equation discovery results saved")
        
        # Save discovered equations and coefficients
        with open(output_dir / "discovered_equations.txt", 'w') as f:
            f.write("Discovered Equations by PySINDy\n")
            f.write("=" * 40 + "\n\n")
            for i, eq in enumerate(discovery.discovered_equations):
                var_name = 'x' if i == 0 else 'y'
                f.write(f"d{var_name}/dt = {eq}\n")
            
            f.write("\n" + "=" * 40 + "\n")
            f.write("Feature Names:\n")
            for i, name in enumerate(discovery.feature_names):
                f.write(f"{i}: {name}\n")
            
            f.write("\n" + "=" * 40 + "\n")
            f.write("Coefficient Matrix:\n")
            f.write(str(discovery.coefficients))
            
            f.write("\n\n" + "=" * 40 + "\n")
            f.write("Performance Metrics:\n")
            f.write(f"R² Score (x): {comparison_results['r2_x']:.6f}\n")
            f.write(f"R² Score (y): {comparison_results['r2_y']:.6f}\n")
            f.write(f"Average R²: {comparison_results['avg_r2']:.6f}\n")
            f.write(f"Total MSE: {comparison_results['total_mse']:.6f}\n")

            f.write("\n")
            # also save SimPy SINDy results if available
            if discovery_simpy is not None:
                f.write("Discovered Equations by PySINDy (SimPy Data)\n")
                f.write("=" * 40 + "\n\n")
                for i, eq in enumerate(discovery_simpy.discovered_equations):
                    var_name = 'x' if i == 0 else 'y'
                    f.write(f"d{var_name}/dt = {eq}\n")

                f.write("\n" + "=" * 40 + "\n")
                f.write("Feature Names (SimPy):\n")
                for i, name in enumerate(discovery_simpy.feature_names):
                    f.write(f"{i}: {name}\n")

                f.write("\n" + "=" * 40 + "\n")
                f.write("Coefficient Matrix (SimPy):\n")
                f.write(str(discovery_simpy.coefficients))

                f.write("\n\n" + "=" * 40 + "\n")
                f.write("Performance Metrics (SimPy):\n")
                f.write(f"R² Score (x): {comparison_results_simpy['r2_x']:.6f}\n")
                f.write(f"R² Score (y): {comparison_results_simpy['r2_y']:.6f}\n")
                f.write(f"Average R²: {comparison_results_simpy['avg_r2']:.6f}\n")
                f.write(f"Total MSE: {comparison_results_simpy['total_mse']:.6f}\n")
        
        print("  ✓ Discovered equations saved to discovered_equations.txt")
    
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Results saved in: {output_dir.absolute()}")
    print("\nFiles generated:")
    for file_path in sorted(output_dir.glob("*")):
        print(f"  • {file_path.name}")
    print()
    
    # Return results for further analysis
    return {
        'ode_model': ode_model,
        'simpy_model': simpy_model,
        'discovery_model': discovery,
        'comparison_results': comparison_results if discovery else None,
        'parameters': params
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fluid Queue Simulation and Equation Discovery")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively")
    parser.add_argument("--no-save", action="store_true", help="Don't save plots to files")
    
    args = parser.parse_args()

    params = [
        {
            'lambda_arrival': 5.0,
            'mu_service': 3.0,
            'p_return': 0.3,
            'gamma_return': 1.5,
            'N_capacity': 2.0
        },
        {
            'lambda_arrival': 5.0,
            'mu_service': 3.0,
            'p_return': 0.3,
            'gamma_return': 1.5,
            'N_capacity': 4.0
        }
    ]
    
    scalings = [1, 10, 50, 100, 500, 1000]
    for param in params:
        for scale in scalings:
            params_copy = param.copy()
            params_copy['lambda_arrival'] = params_copy['lambda_arrival'] * scale
            params_copy['N_capacity'] = params_copy['N_capacity'] * scale
            print(f"\n=== Running analysis with scaling factor: {scale} ===\n")
            main_results = run_complete_analysis(
                params=params_copy,
                save_plots=not args.no_save,
                show_plots=args.show_plots
            )
    
    print("\n🎉 All analyses completed successfully!")
