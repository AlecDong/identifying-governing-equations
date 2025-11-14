"""
Generate a dataset of fluid queue trajectories (ODE + SimPy) and save as a pickle file.
SimPy trajectories are stored with their original event times (no interpolation).
"""

import numpy as np
import pickle
from pathlib import Path
from model.fluid_queue_ode import FluidQueueODE
from model.fluid_queue_simpy import FluidQueueSimPy

def generate_trajectory_dataset(params_list, num_trajectories=10, t_span=(0,1000), num_points=200000, output_file="dataset/trajectory_dataset_2.pkl"):
    """
    Generate trajectories for multiple parameter sets and save as pickle.

    Parameters:
    -----------
    params_list : list of dict
        Each dict contains system parameters, e.g. {'lambda_arrival': 5, 'mu_service': 3, ...}
    num_trajectories : int
        Number of trajectories per parameter set
    t_span : tuple
        Simulation start and end time (t0, tf)
    num_points : int
        Number of points in uniform time grid for ODE
    output_file : str or Path
        Path to save pickle dataset
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    dataset_records = []
    
    for param_idx, params in enumerate(params_list):
        print(f"\nGenerating trajectories for parameter set {param_idx}: {params}")
        
        for traj_idx in range(num_trajectories):
            print(f"  Trajectory {traj_idx+1}/{num_trajectories}")
            
            # ---- SimPy Simulation (no interpolation) ----
            simpy_model = FluidQueueSimPy(**params, random_seed=traj_idx)
            t_simpy, x_simpy, y_simpy = simpy_model.simulate(simulation_time=t_span[1],
                                                             initial_x=0.0,
                                                             initial_y=0.0)

            time_grid = np.linspace(t_span[0], t_span[1], num_points)
            simpy_df = simpy_model.get_data_frame(time_grid)
            
            # ---- Store trajectory ----
            record = {
                'params': params,
                'trajectory_idx': traj_idx,
                'simpy_time': simpy_df["time"],                       # original event times
                'simpy_states': np.column_stack([simpy_df["x"], simpy_df["y"]])
            }
            dataset_records.append(record)
    
    # ---- Save to pickle ----
    with open(output_file, "wb") as f:
        pickle.dump(dataset_records, f)
    
    print(f"\nSaved {len(dataset_records)} trajectories to {output_file}")
    return dataset_records


def load_trajectory_dataset(pickle_file="dataset/trajectory_dataset.pkl"):
    """Load a trajectory dataset saved with pickle"""
    with open(pickle_file, "rb") as f:
        trajectories = pickle.load(f)
    print(f"Loaded {len(trajectories)} trajectories from {pickle_file}")
    return trajectories


if __name__ == "__main__":
    # Base parameters
    base_params_list = [
        {'lambda_arrival': 5.0, 'mu_service': 3.0, 'p_return': 0.3, 'gamma_return': 1.5, 'N_capacity': 2.0},
        # {'lambda_arrival': 5.0, 'mu_service': 3.0, 'p_return': 0.3, 'gamma_return': 1.5, 'N_capacity': 4.0}
    ]

    # Scaling factors
    scales = [1000]

    # Generate full parameter list with scaled lambda and N
    params_list = []
    for base_params in base_params_list:
        for scale in scales:
            params_scaled = base_params.copy()
            params_scaled['lambda_arrival'] *= scale
            params_scaled['N_capacity'] *= scale
            params_list.append(params_scaled)

    # Print to verify
    print("Generating trajectories for parameter sets:")
    for p in params_list:
        print(p)

    # Generate dataset
    dataset = generate_trajectory_dataset(params_list, num_trajectories=1)
    
    # Example: load dataset back
    loaded_dataset = load_trajectory_dataset()
    
    # Example: inspect first trajectory
    first_traj = loaded_dataset[0]
    print("Parameters:", first_traj['params'])
    print("SimPy states shape:", first_traj['simpy_states'].shape)
    print("SimPy time points (first 10):", first_traj['simpy_time'][:10])
