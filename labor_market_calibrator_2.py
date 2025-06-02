# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 13:07:00 2025

@author: mauhl
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
import warnings
import labornet as lbn

# ---------------------------------------------------------------------------
# Type aliases & constants
# ---------------------------------------------------------------------------
NDArrayF = npt.NDArray[np.floating]
EPS_DENOM: float = 1e-12      # avoid 0/0 in vacancy rate
RULE_GAMMA_U: float = 10.0    # γ_u default multiplier
MIN_GAMMA_V: float = 1e-4     # lower bound for γ_v inside optimiser
T_SHOCK_LARGE: int   = 9999   # ensures "no shock" during steady‑state search

@dataclass
class DeltaGammaTopKCalibrator:
    """Block‑coordinate descent fitter for δ_v and γ_v (Top‑K occupations)."""

    # required arrays --------------------------------------------------------
    delta_u: NDArrayF
    delta_v_init: NDArrayF
    gamma_v_init: NDArrayF
    vacancy_emp: NDArrayF                 # empirical V/(E+V) in steady state
    E0: NDArrayF
    U0: NDArrayF
    V0: NDArrayF
    D0: NDArrayF

    # model parameters -------------------------------------------------------
    t_sim: int
    A: NDArrayF
    tau: int

    # fitter hyper‑parameters -----------------------------------------------
    top_k: int = 100
    bounds_delta: Tuple[float, float] = (1e-6, 0.10)
    bounds_gamma: Tuple[float, float] = (MIN_GAMMA_V, 50.0)
    max_sweeps: int = 5
    tol: float = 1e-4

    # internal state (set in __post_init__) ----------------------------------
    delta_v: NDArrayF = field(init=False)
    gamma_v: NDArrayF = field(init=False)
    top_idx: NDArrayF = field(init=False)
    weights: NDArrayF = field(init=False)
    rmse_log: List[float] = field(init=False, default_factory=list)
    _cache: Dict[Tuple[int, float, float], float] = field(init=False, default_factory=dict)

    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.delta_u = np.asarray(self.delta_u, dtype=float)
        self.delta_v = np.asarray(self.delta_v_init, dtype=float).copy()
        self.gamma_v = np.asarray(self.gamma_v_init, dtype=float).copy()
        self.vacancy_emp = np.asarray(self.vacancy_emp, dtype=float)
        self.E0 = np.asarray(self.E0, dtype=float)
        self.U0 = np.asarray(self.U0, dtype=float)
        self.V0 = np.asarray(self.V0, dtype=float)
        self.D0 = np.asarray(self.D0, dtype=float)

        self._check_inputs()
        self._select_top_k()
        self.weights = self.E0[self.top_idx] / self.E0[self.top_idx].sum()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _check_inputs(self) -> None:
        n = len(self.delta_u)
        arrays_to_check = [
            (self.delta_v, "delta_v"), (self.gamma_v, "gamma_v"),
            (self.vacancy_emp, "vacancy_emp"), (self.E0, "E0"),
            (self.U0, "U0"), (self.V0, "V0"), (self.D0, "D0"),
        ]
        for arr, name in arrays_to_check:
            if len(arr) != n:
                raise ValueError(f"Array '{name}' length {len(arr)} ≠ {n}")
        if self.top_k > n:
            raise ValueError("top_k exceeds number of occupations")
        if self.A.shape[0] != n or self.A.shape[1] != n:
            raise ValueError("Adjacency matrix A must be (N×N)")
        if self.max_sweeps <= 0:
            raise ValueError("max_sweeps must be positive")

    def _select_top_k(self) -> None:
        self.top_idx = np.argsort(self.E0)[::-1][: self.top_k]

    # ------------------------------------------------------------------
    # Model wrapper
    # ------------------------------------------------------------------
    def _run_model_with_params(self, delta_v: NDArrayF, gamma_v: NDArrayF) -> Tuple[NDArrayF, NDArrayF, NDArrayF]:
        """Run the labornet solver once and return final (U, V, E)."""
        try:
            U, V, E, *_ = lbn.run_numerical_solution(
                lbn.fire_and_hire_workers,
                self.t_sim,
                self.delta_u,
                delta_v,
                np.maximum(1e-6, RULE_GAMMA_U * self.delta_u),  # γ_u (kept fixed)
                gamma_v,
                employment_0=self.E0,
                unemployment_0=self.U0,
                vacancies_0=self.V0,
                target_demand_function=lambda t, *args: self.D0,
                D_0=self.D0,
                D_f=self.D0,
                t_shock=T_SHOCK_LARGE,
                k=0,
                t_halfsig=0,
                matching=lbn.matching_probability,
                A_matrix=self.A,
                τ=self.tau,
            )
            return U[-1], V[-1], E[-1]
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Model failed: {exc}")
            # Return zeros to yield large error in objective
            n = len(self.delta_u)
            return (np.zeros(n), np.zeros(n), np.zeros(n))

    # ------------------------------------------------------------------
    # Objective for a single occupation  (pure, cached)
    # ------------------------------------------------------------------
    def _objective_occ(self, idx: int, params: NDArrayF) -> float:
        dvi, gvi = params
        dvi = float(np.clip(dvi, *self.bounds_delta))
        gvi = float(np.clip(gvi, *self.bounds_gamma))

        cache_key = (idx, dvi, gvi)
        if cache_key in self._cache:
            return self._cache[cache_key]

        delta_v_tmp = self.delta_v.copy()
        gamma_v_tmp = self.gamma_v.copy()
        delta_v_tmp[idx] = dvi
        gamma_v_tmp[idx] = gvi

        # run model ------------------------------------------------------
        _, V_fin, E_fin = self._run_model_with_params(delta_v_tmp, gamma_v_tmp)
        denom = V_fin + E_fin
        vac_rate_model = np.where(denom > EPS_DENOM, V_fin / denom, 0.0)

        # weighted squared error on top‑K only ---------------------------
        diff = vac_rate_model[self.top_idx] - self.vacancy_emp[self.top_idx]
        w_mse = np.sum(self.weights * diff**2)

        self._cache[cache_key] = w_mse
        return w_mse

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self) -> Tuple[NDArrayF, NDArrayF, List[float]]:
        """Run block‑coordinate descent; return updated δ_v, γ_v, RMSE log."""
        for sweep in range(self.max_sweeps):
            rmse_start = self._current_rmse()
            for idx in self.top_idx:
                x0 = np.array([
                    self.delta_v[idx],
                    self.gamma_v[idx],
                ])
                res = minimize(
                    lambda p: self._objective_occ(idx, p),
                    x0=x0,
                    bounds=[self.bounds_delta, self.bounds_gamma],
                    method="L-BFGS-B",
                )
                self.delta_v[idx], self.gamma_v[idx] = res.x
            rmse_end = self._current_rmse()
            self.rmse_log.append(rmse_end)
            if abs(rmse_end - rmse_start) < self.tol:
                break
        return self.delta_v.copy(), self.gamma_v.copy(), self.rmse_log

    # ------------------------------------------------------------------
    def _current_rmse(self) -> float:
        """Compute weighted RMSE for current δ_v & γ_v on the top‑K set."""
        _, V_fin, E_fin = self._run_model_with_params(self.delta_v, self.gamma_v)
        denom = V_fin + E_fin
        vac_rate_model = np.where(denom > EPS_DENOM, V_fin / denom, 0.0)
        diff = vac_rate_model[self.top_idx] - self.vacancy_emp[self.top_idx]
        return float(np.sqrt(np.sum(self.weights * diff**2)))

# ---------------------------------------------------------------------------
# Example usage (dummy data) – safe to run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    N = 10
    rng = np.random.default_rng(42)
    delta_u = rng.uniform(0.01, 0.03, N)
    delta_v0 = rng.uniform(0.005, 0.02, N)
    gamma_v0 = rng.uniform(0.5, 5, N)
    vacancy_emp = rng.uniform(0.02, 0.05, N)
    E0 = rng.integers(1_000, 10_000, N).astype(float)
    U0 = rng.integers(100, 800, N).astype(float)
    V0 = rng.integers(50, 400, N).astype(float)
    D0 = E0 + U0 + V0
    A = rng.uniform(0, 1, (N, N))

    calibrator = DeltaGammaTopKCalibrator(
        delta_u,
        delta_v0,
        gamma_v0,
        vacancy_emp,
        E0, U0, V0, D0,
        t_sim=200,
        A=A,
        tau=3,
        top_k=5,
    )

    dv_fit, gv_fit, rmse_log = calibrator.fit()
    print("Final weighted RMSE:", rmse_log[-1])
