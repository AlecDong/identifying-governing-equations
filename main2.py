import pickle
import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt

from SINDy.joint_estimation import EquationDiscoveryWithDiffusion
from SINDy.equation_discovery import EquationDiscovery
from SINDy.custom_equation_libraries import min_library


def simulate_drift_only(Xi_drift, feature_library, x0, t_eval):
    """
    Euler integration using the discovered drift coefficients only.
    Xi_drift: (n_features x n_vars) drift coefficients
    feature_library: PySINDy library object
    """
    dt = t_eval[1] - t_eval[0]
    n_vars = len(x0)
    traj = np.zeros((len(t_eval), n_vars))
    traj[0] = x0
    
    for k in range(len(t_eval)-1):
        Theta = feature_library.transform(traj[k:k+1])  # (1, n_features)
        Theta = np.asarray(Theta, dtype=float)
        fk = Theta @ Xi_drift.T  # predicted drift, standardized (1, n_vars)

        traj[k+1] = traj[k] + fk[0] * dt  # Euler step

    return traj

# --------------------
# Stochastic simulation (Euler-Maruyama)
# --------------------
def simulate_sde(Xi_drift, Xi_diff, feature_library, x0, t_eval, n_traj=10):
    dt = t_eval[1] - t_eval[0]
    n_vars = len(x0)
    d_dim = n_vars * n_vars  # diffusion vector length
    all_trajs = np.zeros((n_traj, len(t_eval), n_vars))

    for m in range(n_traj):
        traj = np.zeros((len(t_eval), n_vars))
        traj[0] = x0

        for k in range(len(t_eval)-1):
            Theta = feature_library.transform(traj[k:k+1])
            Theta = np.asarray(Theta, dtype=float)
            
            # Drift
            f_drift = Theta @ Xi_drift  # (1, n_vars)
            
            # Diffusion: reshape from (n_features, n^2) to (n_vars, n_vars)
            D_vec = Theta @ Xi_diff  # (1, n^2)
            D_mat = D_vec.reshape(n_vars, n_vars)
            
            # Generate stochastic increment
            dW = np.random.randn(n_vars) * np.sqrt(dt)  # standard Wiener increment
            
            # Euler-Maruyama step
            traj[k+1] = traj[k] + f_drift[0] * dt + D_mat @ dW
        
        all_trajs[m] = traj

    return all_trajs


# --------------------
# Load dataset
# --------------------
with open('dataset/trajectory_dataset.pkl', 'rb') as f:
    trajectories = pickle.load(f)

traj = trajectories[0]
print(traj["params"])
simpy_time = traj['simpy_time']
simpy_states = traj['simpy_states']
params = traj['params']

data_df = pd.DataFrame({
    'time': simpy_time,
    'x': simpy_states[:, 0],
    'y': simpy_states[:, 1]
})

# --------------------
# Feature Library Setup
# --------------------
poly_lib = ps.PolynomialLibrary(
    degree=2, include_bias=True, include_interaction=True
)
queue_lib = min_library(N=params['N_capacity'])

combined_lib = ps.GeneralizedLibrary([poly_lib, queue_lib])


# ============================================================
# drift + diffusion jointly
# ============================================================
coef_threshold = 0.1
eq_block = EquationDiscoveryWithDiffusion(
    threshold=coef_threshold, alpha=0.05, feature_library=combined_lib
)
eq_block.prepare_data(data_df)
Xi_drift = eq_block.fit_weighted_drift()
print("\nDiscovered drift coefficients:\n", Xi_drift)

eq_block.print_discovered_equations()

# feature_names = eq_block.feature_names
# n_vars = 2
# d_dim = n_vars * n_vars  # 4 for 2D system

# eq_block.pretty_print_equations()


# # ============================================================
# # drift and diffusion separately
# # ============================================================
# eq_disc = EquationDiscoveryWithDiffusion(threshold=coef_threshold, alpha=0.05, feature_library=combined_lib)  # Use your class as defined above
# eq_disc.prepare_data(data_df)
# drift_model, diff_model = eq_disc.fit_drift_and_diffusion()

# print('\n===== Drift model =====')
# drift_model.print()

# print('\n===== Diffusion model =====')
# output_names = ["(x0 x0)'", "(x0 x1)'", "(x1 x0)'", "(x1 x1)'"]
# diff_model.print(lhs=output_names)


# x0 = simpy_states[0]  # initial state
# t_eval = simpy_time
# drift_only_traj = simulate_drift_only(Xi_drift.coef_, combined_lib, x0, t_eval)
# drift_only_sep_traj = simulate_drift_only(drift_model.coefficients(), combined_lib, x0, t_eval)
# real_coefs = np.array([
#     [params["lambda_arrival"], 0, params["gamma_return"], 0, 0, 0, -params["mu_service"], 0],
#     [0, 0, -params["gamma_return"], 0, 0, 0, params["p_return"] * params["mu_service"], 0]
# ])
# real_traj = simulate_drift_only(real_coefs, combined_lib, x0, t_eval)

# # --------------------
# # Plot original vs drift-only
# # --------------------
# plt.figure(figsize=(10,5))
# plt.plot(simpy_time, simpy_states[:, 0], label="x0 (SimPy)", c="blue")
# plt.plot(simpy_time, simpy_states[:, 1], label="x1 (SimPy)", c="orange")
# plt.plot(simpy_time, drift_only_sep_traj[:, 0], '-', label="x0 (Predicted drift sep)", c="green")
# plt.plot(simpy_time, drift_only_sep_traj[:, 1], '-', label="x1 (Predicted drift sep)", c="red")
# plt.plot(simpy_time, drift_only_traj[:, 0], '--', label="x0 (Predicted drift)", c="blue")
# plt.plot(simpy_time, drift_only_traj[:, 1], '--', label="x1 (Predicted drift)", c="orange")
# plt.plot(simpy_time, real_traj[:, 0], ':', label="x0 (True drift)", c="cyan")
# plt.plot(simpy_time, real_traj[:, 1], ':', label="x1 (True drift)", c="magenta")
# plt.xlabel("Time")
# plt.ylabel("State")
# plt.title("SimPy Trajectories vs Drift-only SINDy Prediction")
# plt.legend()
# plt.show()
