import numpy as np
import pysindy as ps
import warnings
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error, r2_score

def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

def weighted_least_squares(X_dot, Theta, W_all):
    """
    Full weighted LS solution for general d×d weight matrices at each timestep.

    X_dot: (N, d)
    Theta: (N, p)
    W_all: (N, d, d)   # full weight matrices

    Returns Xi of shape (p, d)
    """
    Theta = np.asarray(Theta)
    N, d = X_dot.shape
    p = Theta.shape[1]

    # We'll build A of shape (N*d, p*d)
    A = np.zeros((N*d, p*d))
    b = np.zeros(N*d)

    row = 0
    for t in range(N):
        # W^{1/2}
        # Must use symmetric sqrt, not elementwise
        W_sqrt = np.linalg.cholesky(W_all[t])

        # Block: (I_d ⊗ Theta_t)
        Theta_t = Theta[t]  # shape (p,)
        kron_block = np.kron(np.eye(d), Theta_t)  # (d, p*d)

        # Weighted block: W_sqrt @ kron_block
        A[row:row+d, :] = W_sqrt @ kron_block

        # Weighted RHS: W_sqrt @ X_dot[t]
        b[row:row+d] = W_sqrt @ X_dot[t]

        row += d

    # Solve the full LS system
    v, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Reshape vec(Xi) to (p, d)
    Xi = v.reshape(p, d)
    return Xi

class EquationDiscoveryWithDiffusion:
    """
    SINDy-based discovery of drift and diffusion from time series.
    Follows the finite difference approximations from the paper:
      - Drift: ΔX / Δt
      - Diffusion: ΔX_i ΔX_j / (2 Δt)

    let N be number of time steps, d number of variables, p number of features
    """

    def __init__(self, threshold=0.01, alpha=0.05, feature_library=None, feature_names=None):
        self.threshold = threshold
        self.alpha = alpha
        self.feature_library = feature_library or ps.PolynomialLibrary(degree=2, include_bias=True)

        self.feature_names = None

        # Least squares results
        self.Xi_drift = None   # shape (p, d)

        # Data
        self.X = None # shape (N, d)
        self.dt = None # shape (N - 1,)
        self.X_dot = None # shape (N - 1, d)
        self.D = None # shape (N - 1, d, d)

    def prepare_data(self, data_df):
        """Compute drift and diffusion from time series."""
        data_df = data_df.sort_values('time').reset_index(drop=True)
        time_full = data_df['time'].values
        X_full = data_df.drop(columns=['time']).values

        # Finite differences
        dX = np.diff(X_full, axis=0)
        dt = np.diff(time_full).reshape(-1, 1)

        print("Finite difference stats:")
        print("dt: mean =", np.mean(dt), ", std =", np.std(dt))
        print("dX: mean =", np.mean(dX), ", std =", np.std(dX))

        # Drift: ΔX / Δt
        self.X_dot = dX / dt
        print(self.X_dot.shape)

        # Align state and time to (N-1)
        self.X = X_full[:-1]
        self.dt = dt

        # Diffusion: ΔX_i ΔX_j / (2 Δt)
        n_steps, n_vars = self.X_dot.shape
        self.D = np.zeros((n_steps, n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                self.D[:, i, j] = (dX[:, i] * dX[:, j]) / (2 * dt.squeeze())
        print("mean, std of D entries:", np.mean(self.D), np.std(self.D))
        print("mean, std of X_dot entries:", np.mean(self.X_dot), np.std(self.X_dot))
        
        return self.X, self.X_dot, self.D, self.dt

    def build_library(self, X):
        """Compute feature library matrix Θ(X)."""
        return self.feature_library.fit_transform(X)
    
    def fit_weighted_drift(self, weighted=True):
        """Fit drift using weighted least squares with optional diffusion weighting."""

        n_steps, d = self.X_dot.shape
        Theta = self.build_library(self.X)
        p = Theta.shape[1]

        if weighted:
            # Average diffusion over time: (d, d)
            D_avg = np.mean(self.D, axis=0)  # shape (d, d)
            
            # Invert safely with small jitter
            jitter = 1e-6
            W = np.linalg.pinv(D_avg + jitter * np.eye(d))  # shape (d, d)
            
            # Apply same weight to all timesteps
            W_all = np.tile(W, (n_steps, 1, 1))  # shape (N, d, d)

            self.Xi_drift = weighted_least_squares(self.X_dot, Theta, W_all)
        else:
            # Ordinary least squares
            self.Xi_drift, *_ = np.linalg.lstsq(Theta, self.X_dot, rcond=None)

        return self.Xi_drift

    def print_discovered_equations(self):
        """Print the discovered drift equations."""
        if self.Xi_drift is None:
            raise ValueError("Model not fitted yet. Call fit_weighted_drift() first.")

        self.feature_names = self.feature_library.get_feature_names()
        equations = []
        for i in range(self.Xi_drift.shape[1]):
            terms = []
            for j in range(self.Xi_drift.shape[0]):
                coeff = self.Xi_drift[j, i]
                if abs(coeff) >= self.threshold:
                    terms.append(f"({coeff:.4f})*{self.feature_names[j]}")
            equation = " + ".join(terms) if terms else "0"
            equations.append(f"dx_{i}/dt = {equation}")
        
        print("Discovered Drift Equations:")
        for eq in equations:
            print(eq)
        
        self.discovered_equations = equations
        return equations

    def compare_with_true_system(self, solution_ode, initial_state, t_eval):
        """
        Compare discovered system with true system.
        
        Parameters:
        -----------
        solution_ode : ndarray
            True system solution
        initial_state : array_like
            Initial conditions
        t_eval : array_like
            Time points for evaluation
            
        Returns:
        --------
        comparison_results : dict
            Comparison metrics and data
        """        
        # Simulate discovered system
        try:
            discovered_solution = self.simulate_drift_only(initial_state, t_eval)
        except Exception as e:
            print(f"Warning: Could not simulate discovered system: {e}")
            discovered_solution = np.zeros_like(solution_ode)
        
        # Calculate metrics
        mse_x = mean_squared_error(solution_ode[:, 0], discovered_solution[:, 0])
        mse_y = mean_squared_error(solution_ode[:, 1], discovered_solution[:, 1])
        
        r2_x = r2_score(solution_ode[:, 0], discovered_solution[:, 0])
        r2_y = r2_score(solution_ode[:, 1], discovered_solution[:, 1])
        
        return {
            'time': t_eval,
            'true_solution': solution_ode,
            'discovered_solution': discovered_solution,
            'mse_x': mse_x,
            'mse_y': mse_y,
            'r2_x': r2_x,
            'r2_y': r2_y,
            'total_mse': mse_x + mse_y,
            'avg_r2': (r2_x + r2_y) / 2
        }

    def simulate(self, initial_state, t_eval):
        """
        Simulate the discovered system using the estimated drift terms.

        Parameters:
        -----------
        initial_state : array_like
            Initial conditions for the simulation.
        t_eval : array_like
            Time points for evaluation.
        """
        dt = t_eval[1] - t_eval[0]
        n_vars = len(initial_state)
        traj = np.zeros((len(t_eval), n_vars)) # (N, d)
        traj[0] = initial_state

        for k in range(len(t_eval)-1):
            Theta = self.build_library(traj[k:k+1])
            Theta = np.asarray(Theta, dtype=float) # theta is (1, p)
            fk = Theta @ self.Xi_drift # fk is (1, d)

            traj[k+1] = traj[k] + fk[0] * dt  # Euler step

        return traj
