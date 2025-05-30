# -*- coding: utf-8 -*-
"""
Created on Fri May 30 13:59:14 2025

@author: mauhl
"""
import numpy as np
from scipy.optimize import minimize_scalar
from functools import lru_cache
from typing import Tuple, Dict, Any, Sequence
import warnings
import labornet as lbn

class LaborMarketCalibrator:
    """Calibrate the speed‐adjustment parameter (kappa) so that modelled
    steady‑state vacancy rates match empirical rates.

    Parameters
    ----------
    delta_u, delta_v : array_like (N,)
        Empirical spontaneous separation and vacancy‑opening probabilities.
    vacancy_rate_emp : array_like (N,)
        Empirical vacancy stock per position, V_i / (E_i + V_i).
    E0, U0, V0 : array_like (N,)
        Initial employment, unemployment, and vacancy stocks.
    D0 : array_like (N,)
        Target demand vector. A sensible default is E0 + U0 + V0.
    t_sim : int
        Number of time steps to run the simulation (long enough to reach
        steady state; ≥ 400 recommended).
    A : ndarray (N, N)
        Occupation‑mobility adjacency matrix used by the matching function.
    tau : float
        Matching‑function scaling parameter.
    link_gamma_u_to_kappa : bool, optional
        If True, set \gamma_u = \kappa * |\delta_u - \delta_v| to keep upward and
        downward adjustment speeds symmetric.  If False (default) use the
        literature rule \gamma_u = 10 \delta_u.
    """

    def __init__(
        self,
        delta_u: np.ndarray,
        delta_v: np.ndarray,
        vacancy_rate_emp: np.ndarray,
        E0: np.ndarray,
        U0: np.ndarray,
        V0: np.ndarray,
        D0: np.ndarray,
        t_sim: int,
        A: np.ndarray,
        tau: float,
        *,
        link_gamma_u_to_kappa: bool = False,
    ) -> None:
        # Convert to np.ndarray and store
        self.delta_u = np.asarray(delta_u, dtype=float)
        self.delta_v = np.asarray(delta_v, dtype=float)
        self.vacancy_rate_emp = np.asarray(vacancy_rate_emp, dtype=float)
        self.E0 = np.asarray(E0, dtype=float)
        self.U0 = np.asarray(U0, dtype=float)
        self.V0 = np.asarray(V0, dtype=float)
        self.D0 = np.asarray(D0, dtype=float)
        self.t_sim = int(t_sim)
        self.A = np.asarray(A, dtype=float)
        self.τ = float(tau)  # labornet expects the Greek letter key
        self.link_gamma_u_to_kappa = link_gamma_u_to_kappa

        # Gap with a positive floor to avoid zeros
        self.gap = np.maximum(1e-8, np.abs(self.delta_u - self.delta_v))

        # Pre‑compute weights once (employment share)
        self.weights = self.E0 / np.sum(self.E0)

        self._validate_inputs()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_inputs(self) -> None:
        arrays: Sequence[np.ndarray] = (
            self.delta_u,
            self.delta_v,
            self.vacancy_rate_emp,
            self.E0,
            self.U0,
            self.V0,
            self.D0,
        )
        n_occ = len(self.delta_u)
        if not all(len(arr) == n_occ for arr in arrays):
            raise ValueError("All occupation‑level arrays must have the same length.")

        if np.any((self.vacancy_rate_emp < 0) | (self.vacancy_rate_emp > 1)):
            warnings.warn("Vacancy rates should lie in [0, 1] – check input scale.")

    # ------------------------------------------------------------------
    # Core objective (memoised for speed)
    # ------------------------------------------------------------------
    @lru_cache(maxsize=None)
    def _objective_function(self, kappa: float) -> float:
        """Mean‑squared error between modelled and empirical vacancy rates."""
        # Guard against pathological values explored by the optimiser
        kappa = float(max(kappa, 1e-3))

        gamma_v = kappa * self.gap
        if self.link_gamma_u_to_kappa:
            gamma_u = gamma_v.copy()
        else:
            gamma_u = np.maximum(1e-6, 10.0 * self.delta_u)

        try:
            U, V, E, *_ = lbn.run_numerical_solution(
                lbn.fire_and_hire_workers,
                self.t_sim,
                self.delta_u,
                self.delta_v,
                gamma_u,
                gamma_v,
                employment_0=self.E0,
                unemployment_0=self.U0,
                vacancies_0=self.V0,
                target_demand_function=self._const_demand,
                D_0=self.D0,
                D_f=self.D0,
                t_shock=9999,
                k=0,
                t_halfsig=0,
                matching=lbn.matching_probability,
                A_matrix=self.A,
                τ=self.τ,
            )
        except Exception as exc:
            warnings.warn(f"Simulation failed for κ={kappa:.4g}: {exc}")
            return 1e6

        total_positions = V[-1] + E[-1]
        mask = total_positions > 1e-12
        vac_rate_model = np.zeros_like(total_positions)
        vac_rate_model[mask] = V[-1][mask] / total_positions[mask]

        # Weighted MSE (same weights used for R²)
        mse = np.sum(self.weights * (vac_rate_model - self.vacancy_rate_emp) ** 2)
        return float(mse)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calibrate(
        self, *, kappa_bounds: Tuple[float, float] = (1.0, 50.0), coarse_grid: int = 25
    ) -> Dict[str, Any]:
        """Search for the κ that minimises the weighted vacancy‑rate MSE.

        Parameters
        ----------
        kappa_bounds : tuple, optional
            (low, high) search interval for κ.
        coarse_grid : int, optional
            Number of equally‑spaced κ values to evaluate before scalar optimisation.
        """
        k_low, k_high = map(float, kappa_bounds)
        if k_low <= 0 or k_high <= k_low:
            raise ValueError("kappa_bounds must be positive and ordered (low < high).")

        # ------------------------------------------------------------------
        # 1. Coarse grid – get a good starting interval
        # ------------------------------------------------------------------
        grid = np.linspace(k_low, k_high, coarse_grid)
        grid_mse = np.array([self._objective_function(k) for k in grid])
        best_idx = int(grid_mse.argmin())
        # Create a tight bracket around the best grid point (± one grid step)
        step = (k_high - k_low) / (coarse_grid - 1)
        bracket_low = max(k_low, grid[best_idx] - step)
        bracket_high = min(k_high, grid[best_idx] + step)

        # ------------------------------------------------------------------
        # 2. Scalar optimiser (bounded Brent)
        # ------------------------------------------------------------------
        opt = minimize_scalar(
            self._objective_function,
            bounds=(bracket_low, bracket_high),
            method="bounded",
            options={"xatol": 1e-3},
        )
        if not opt.success:
            warnings.warn("Optimization did not converge – result may be suboptimal.")
        kappa_star: float = float(opt.x)

        # ------------------------------------------------------------------
        # 3. Diagnostics at optimum
        # ------------------------------------------------------------------
        final_mse = float(self._objective_function(kappa_star))
        gamma_v_star = kappa_star * self.gap
        gamma_u_star = gamma_v_star.copy() if self.link_gamma_u_to_kappa else np.maximum(
            1e-6, 10.0 * self.delta_u
        )
        U, V, E, *_ = lbn.run_numerical_solution(
            lbn.fire_and_hire_workers,
            self.t_sim,
            self.delta_u,
            self.delta_v,
            gamma_u_star,
            gamma_v_star,
            employment_0=self.E0,
            unemployment_0=self.U0,
            vacancies_0=self.V0,
            target_demand_function=self._const_demand,
            D_0=self.D0,
            D_f=self.D0,
            t_shock=9999,
            k=0,
            t_halfsig=0,
            matching=lbn.matching_probability,
            A_matrix=self.A,
            τ=self.τ,
        )
        vac_rate_final = V[-1] / (V[-1] + E[-1])
        r_squared = self._weighted_r_squared(vac_rate_final)

        return {
            "kappa_optimal": kappa_star,
            "mse": final_mse,
            "rmse": np.sqrt(final_mse),
            "r_squared": r_squared,
            "optimization_result": opt,
            "predicted_vacancy_rates": vac_rate_final,
            "gamma_v_optimal": gamma_v_star,
            "gamma_u_optimal": gamma_u_star,
            "grid_kappa": grid,
            "grid_mse": grid_mse,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _const_demand(self, *_):  # noqa: D401 – simple callback
        """Constant‐in‐time target demand function for steady‑state runs."""
        return self.D0

    def _weighted_r_squared(self, model_vacancy: np.ndarray) -> float:
        """Weighted R² using the same employment weights as the MSE."""
        resid = model_vacancy - self.vacancy_rate_emp
        ss_res = np.sum(self.weights * resid ** 2)
        ss_tot = np.sum(self.weights * (self.vacancy_rate_emp - np.average(self.vacancy_rate_emp, weights=self.weights)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

import numpy as np
from labor_market_calibrator import LaborMarketCalibrator   # the file in the canvas
import labornet as lbn                                       # your core model

# --- 1.  Prepare / load all occupation-level arrays -------------------------
delta_u            = ...   # shape (N,)
delta_v            = ...
vacancy_rate_emp   = ...
E0, U0, V0         = ...   # initial stocks, each (N,)
D0                 = E0 + U0 + V0    # target demand including vacancies

# Other modelwide settings
t_sim  = 600                 # iterations until steady state
A      = ...                 # adjacency / matching matrix
tau    = 0.15                # matching Cobb-Douglas exponent, for example

# --- 2.  Instantiate the calibrator ----------------------------------------
calibrator = LaborMarketCalibrator(
    delta_u, delta_v, vacancy_rate_emp,
    E0, U0, V0, D0,
    t_sim, A, tau,
    link_gamma_u=False         # set True if you also want γ_u ∝ |δ_u − δ_v|
)

# --- 3.  Run the calibration (optimises κ) ---------------------------------
results = calibrator.calibrate(kappa_bounds=(5, 20))

# --- 4.  Inspect results ----------------------------------------------------
print(f"Optimal κ   : {results['kappa_optimal']:.3f}")
print(f"RMSE        : {results['rmse']:.5f}")
print(f"Weighted R² : {results['r_squared']:.3f}")

# Model-predicted vacancy rates are in:
vac_rate_model = results['predicted_vacancy_rates']

# You can now plot predicted vs. empirical, or run further simulations
