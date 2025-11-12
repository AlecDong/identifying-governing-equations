import numpy as np
import pysindy as ps
import warnings
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error, r2_score

def weighted_stlsq(Theta, Y, threshold=0.1, alpha=0.05, max_iter=20, weights=None):
    """
    Weighted Sequential Thresholded Least Squares (STLSQ) for multi-output regression.
    """
    N, p = Theta.shape
    Y = np.atleast_2d(Y)
    if Y.shape[0] != N:
        Y = Y.T  # rows = samples
    _, n_targets = Y.shape

    if weights is None:
        weights = np.ones(N)

    # Scale Theta and Y by sqrt(weights)
    W_sqrt = np.sqrt(weights)[:, None]  # (N, 1)
    Theta_w = Theta * W_sqrt
    Y_w = Y * W_sqrt

    I = np.eye(p)
    # initial ridge solve
    Xi = solve(Theta_w.T @ Theta_w + alpha * I, Theta_w.T @ Y_w)

    # ensure shape (p, n_targets)
    if Xi.ndim == 1:
        Xi = Xi[:, None]

    # iterative thresholding
    for it in range(max_iter):
        small = np.abs(Xi) < threshold
        Xi[small] = 0.0
        for j in range(n_targets):
            keep = ~small[:, j]
            if np.sum(keep) == 0:
                continue
            Xi[keep, j] = solve(
                Theta_w[:, keep].T @ Theta_w[:, keep] + alpha * np.eye(np.sum(keep)),
                Theta_w[:, keep].T @ Y_w[:, j]
            )

    return Xi

class EquationDiscoveryWithDiffusion:
    """
    SINDy-based discovery of drift and diffusion from time series.
    Follows the finite difference approximations from the paper:
      - Drift: ΔX / Δt
      - Diffusion: ΔX_i ΔX_j / (2 Δt)
    """

    def __init__(self, threshold=0.01, alpha=0.05, feature_library=None):
        self.threshold = threshold
        self.alpha = alpha
        self.feature_library = feature_library or ps.PolynomialLibrary(degree=2, include_bias=True)

        # Fitted models
        self.drift_model = None
        self.diff_model = None
        self.feature_names = None

        # Block-fit results
        self.Xi_drift = None   # shape (p, n)
        self.Xi_diff = None    # shape (p, d)
        self.block_model = None

        # Data
        self.X = None
        self.t = None
        self.X_dot = None
        self.D = None

    def prepare_data(self, data_df):
        """Compute drift and diffusion from time series."""
        data_df = data_df.sort_values('time').reset_index(drop=True)
        time_full = data_df['time'].values
        X_full = data_df[['x', 'y']].values

        # Finite differences
        dX = np.diff(X_full, axis=0)      # (N-1, n)
        dt = np.diff(time_full)           # (N-1,)

        print("Finite difference stats:")
        print("dt: mean =", np.mean(dt), ", std =", np.std(dt))
        print("dX: mean =", np.mean(dX), ", std =", np.std(dX))

        # Drift: ΔX / Δt
        self.X_dot = dX / dt[:, None]

        # Align state and time to (N-1)
        self.X = X_full[:-1]              # (N-1, n)
        self.t = time_full[1:]            # (N-1,)

        # Diffusion: ΔX_i ΔX_j / (2 Δt)
        n_steps, n_vars = self.X_dot.shape
        self.D = np.zeros((n_steps, n_vars * n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                self.D[:, i * n_vars + j] = (dX[:, i] * dX[:, j]) / (2 * dt)
        print("mean, std of D entries:", np.mean(self.D), np.std(self.D))
        print("mean, std of X_dot entries:", np.mean(self.X_dot), np.std(self.X_dot))
        
        return self.X, self.X_dot, self.D, self.t

    def build_library(self, X):
        """Compute feature library matrix Θ(X)."""
        return self.feature_library.fit_transform(X)

    def fit_drift_and_diffusion(self):
        """Fit drift and diffusion using STLSQ sequentially (PySINDy)."""
        if self.X is None or self.X_dot is None or self.D is None:
            raise ValueError("Call prepare_data() before fitting.")

        # Drift model
        self.drift_model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=self.threshold, alpha=self.alpha),
            feature_library=self.feature_library
        )
        self.drift_model.fit(self.X, x_dot=self.X_dot, t=self.t)

        # Diffusion model (treat ΔX_i ΔX_j / 2Δt as targets)
        self.diff_model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=self.threshold, alpha=self.alpha),
            feature_library=self.feature_library
        )
        self.diff_model.fit(self.X, x_dot=self.D, t=self.t)

        self.feature_names = self.feature_library.get_feature_names()
        return self.drift_model, self.diff_model
    
    def fit_drift_and_diffusion_joint(self, verbose=False):
        """
        Fit diffusion first, compute per-sample variance, then fit drift using weighted STLSQ.

        Weighted STLSQ: loss = sum_i w_i * ||Theta_i @ Xi - X_dot_i||^2 + alpha * ||Xi||^2
        """
        if self.X is None or self.X_dot is None or self.D is None:
            raise ValueError("Call prepare_data() before fitting.")

        n_samples, n_vars = self.X.shape

        # prepare feature library (fit on X so transform is consistent)
        Theta = np.asarray(self.build_library(self.X), dtype=float)  # (N, p)
        p = Theta.shape[1]

        # ---- Fit diffusion model (targets = D) ----
        self.diff_model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=self.threshold, alpha=self.alpha),
            feature_library=self.feature_library
        )
        # Convert D to ndarray
        D_array = np.asarray(self.D, dtype=float)
        self.diff_model.fit(self.X, x_dot=D_array, t=self.t)

        Xi_diff = np.asarray(self.diff_model.coefficients())
        self.Xi_diff = Xi_diff

        # ---- Compute per-sample scalar variance from diffusion model ----
        g_hat = Theta @ self.Xi_diff.T  # (N, d)
        s = np.sum(g_hat**2, axis=1)  # scalar variance per sample
        s = np.maximum(s, 1e-12)      # numerical safety
        weights = 1.0 / (s + 1e-6)             # inverse variance weighting
        weights *= len(weights) / np.sum(weights)

        if verbose:
            print(f"Variance stats: mean={np.mean(s):.3g}, std={np.std(s):.3g}, weights mean={np.mean(weights):.3g}")

        # ---- Weighted STLSQ for drift ----
        X_dot_array = np.asarray(self.X_dot, dtype=float)
        Xi_drift = weighted_stlsq(
            Theta,
            X_dot_array,
            threshold=self.threshold,
            alpha=self.alpha,
            max_iter=10,
            weights=weights
        )

        self.Xi_drift = Xi_drift.T

        # Populate drift model object for convenience
        self.drift_model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=self.threshold, alpha=self.alpha),
            feature_library=self.feature_library
        )
        self.drift_model.coef_ = self.Xi_drift  # bypass fitting
        self.feature_names = self.feature_library.get_feature_names()

        if verbose:
            nnz = np.sum(np.abs(np.hstack([self.Xi_drift, self.Xi_diff])) > 1e-12)
            print(f"Done. Nonzero terms: {nnz}")

        return self.drift_model, self.diff_model


    def pretty_print_equations(self, coef_tol=1e-12):
        """
        Pretty print the discovered drift and diffusion formulas from Xi_drift / Xi_diff.
        Must have run fit_drift_and_diffusion() or similar to populate Xi_*.
        """
        if self.Xi_drift is None or self.Xi_diff is None:
            raise ValueError("Run fit_drift_and_diffusion first.")

        # feature names
        try:
            fnames = self.feature_names or self.feature_library.get_feature_names()
        except Exception:
            p = self.Xi_drift.shape[0]
            fnames = [f"f{i}" for i in range(p)]

        n = self.Xi_drift.shape[0]
        # Drift
        print("Drift (f):")
        for var_idx in range(n):
            terms = []
            for k, fname in enumerate(fnames):
                c = self.Xi_drift[var_idx, k]
                if abs(c) > coef_tol:
                    terms.append(f"({c:.6g})*{fname}")
            rhs = " + ".join(terms) if terms else "0"
            print(f"  f_{var_idx}(X) = {rhs}")

        # Diffusion
        print("\nDiffusion (D entries) — estimated from ΔX_i ΔX_j/(2Δt):")
        d = self.Xi_diff.shape[0]
        n_vars = int(np.sqrt(d))
        for i in range(n_vars):
            for j in range(n_vars):
                terms = []
                row = i * n_vars + j
                for k, fname in enumerate(fnames):
                    c = self.Xi_diff[row, k]
                    if abs(c) > coef_tol:
                        terms.append(f"({c:.6g})*{fname}")
                rhs = " + ".join(terms) if terms else "0"
                print(f"  (x{i} x{j})' = {rhs}")
    
    @property
    def discovered_equations(self):
        if self.Xi_drift is None or self.Xi_diff is None:
            raise ValueError("Run fit_drift_and_diffusion first.")
        
        # should return a list with only drift equations
        equations = []
        n = self.Xi_drift.shape[1]
        for var_idx in range(n):
            terms = []
            for k, fname in enumerate(self.feature_names):
                c = self.Xi_drift[k, var_idx]
                if abs(c) > 1e-12:
                    terms.append(f"({c:.6g})*{fname}")
            rhs = " + ".join(terms) if terms else "0"
            equations.append(f"f_{var_idx}(X) = {rhs}")
        return equations
    
    @property
    def coefficients(self):
        if self.Xi_drift is None or self.Xi_diff is None:
            raise ValueError("Run fit_drift_and_diffusion first.")
        return self.Xi_drift

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

    def simulate_drift_only(self, initial_state, t_eval):
        """
        Simulate the discovered system using only the drift terms.
        """
        dt = t_eval[1] - t_eval[0]
        n_vars = len(initial_state)
        traj = np.zeros((len(t_eval), n_vars))
        traj[0] = initial_state

        for k in range(len(t_eval)-1):
            Theta = self.feature_library.transform(traj[k:k+1])  # (1, n_features)
            Theta = np.asarray(Theta, dtype=float)
            fk = Theta @ self.Xi_drift  # predicted drift, standardized (1, n_vars)

            traj[k+1] = traj[k] + fk[0] * dt  # Euler step

        return traj
