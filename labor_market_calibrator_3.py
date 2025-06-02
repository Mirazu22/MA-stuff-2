from __future__ import annotations

import struct
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Sequence, Dict

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

import labornet as lbn

# Module‑level constants
MIN_GAMMA = 1e-4            # absolute floor for γ_v
R_MIN = 5.0                 # lower bound for ratio r = γ_v/δ_v
R_MAX = 20.0                # upper bound for ratio r = γ_v/δ_v
DELTA_V_MIN = 1e-6          # lower bound for δ_v
DELTA_V_MAX = 0.10          # upper bound for δ_v
T_SHOCK_LARGE = 9999        # effectively no shock
EPS_DENOM = 1e-12           # to avoid division by zero in vacancy rates
CONV_TOL = 1e-6             # convergence tol for horizon detection
CACHE_MAX_SIZE = 1000       # max entries in manual cache

NDArrayFloat = npt.NDArray[np.floating]


class DeltaGammaTopKCalibrator:
    """Calibrate δ_v and γ_v for the largest *K* occupations.

    The calibrator performs block‑coordinate descent, fitting `(δ_v, γ_v)`
    pairs for one occupation at a time while holding the others fixed.  To
    speed up wall‑clock time the per‑occupation minimisations are executed in
    parallel using a thread‑pool and a manual cache to avoid redundant calls
    to the (expensive) agent‑based solver in *labornet*.
    """

    def __init__(
        self,
        delta_u: NDArrayFloat,
        delta_v_init: NDArrayFloat,
        gamma_v_init: NDArrayFloat,
        vacancy_emp: NDArrayFloat,
        E0: NDArrayFloat,
        U0: NDArrayFloat,
        V0: NDArrayFloat,
        D0: NDArrayFloat,
        t_sim: int,
        A: NDArrayFloat,
        tau: int,
        top_k: int = 100,
        n_workers: int | None = None,
        max_sweeps: int = 5,
        tol_rmse: float = 1e-4,
    ):
        """Parameters
        ----------
        delta_u
            Spontaneous separation probabilities (length *N*).
        delta_v_init
            Initial guesses for spontaneous vacancy probabilities.
        gamma_v_init
            Initial guesses for directed vacancy speeds.
        vacancy_emp
            Empirical steady‑state vacancy rates (``V / (E+V)``).
        E0, U0, V0
            Initial employment, unemployment and vacancy stocks.
        D0
            Target labour demand (``E0 + U0 + V0``).
        t_sim
            Maximum time‑steps for the *labornet* solver.
        A
            Matching/adjacency matrix (``N × N``).
        tau
            Long‑term unemployment threshold (passed straight to *labornet*).
        top_k
            Number of largest occupations (by employment) to calibrate.
        n_workers
            Thread‑pool size.  Defaults to ``min(top_k, active_threads)``.
        max_sweeps
            Maximum block‑coordinate sweeps.
        tol_rmse
            Stopping tolerance on RMSE improvement between sweeps.
        """

        # --------------------------- input validation -----------------------
        n_occ = len(delta_u)
        if A.shape != (n_occ, n_occ):
            raise ValueError(
                f"A matrix shape {A.shape} does not match number of occupations {n_occ}."
            )
        if not (
            len(delta_v_init)
            == len(gamma_v_init)
            == len(vacancy_emp)
            == n_occ
        ):
            raise ValueError("All input arrays must have the same length.")
        if not (1 <= top_k <= n_occ):
            raise ValueError(f"top_k must be between 1 and {n_occ}.")
        if max_sweeps < 1:
            raise ValueError("max_sweeps must be >= 1.")

        # --------------------------- store copies ---------------------------
        self.delta_u = delta_u.copy()
        self.delta_v = delta_v_init.copy()
        self.gamma_v = gamma_v_init.copy()
        self.vac_emp = vacancy_emp.copy()
        self.E0 = E0.copy()
        self.U0 = U0.copy()
        self.V0 = V0.copy()
        self.D0 = D0.copy()
        self.A = A.copy()
        self.tau = tau
        self.t_sim_full = t_sim

        # sort occupations by employment size (descending)
        self.sorted_idx = np.argsort(-self.E0)
        self.top_idx = self.sorted_idx[:top_k]

        top_employment = self.E0[self.top_idx]
        self.top_weights = top_employment / top_employment.sum()

        self.n_workers = (
            n_workers
            if n_workers is not None
            else min(len(self.top_idx), max(1, threading.active_count()))
        )
        self.max_sweeps = max_sweeps
        self.tol_rmse = tol_rmse

        # manual cache: key (bytes) → (U, V, E)
        self._cache: Dict[bytes, Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]] = {}
        self._cache_lock = threading.Lock()

        # effective horizon based on 95 % convergence criterion
        self.t_sim_eff = self._compute_short_horizon()

    # ---------------------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------------------

    def _compute_short_horizon(self) -> int:
        """Detect a shorter horizon *t* where ≥95 % of occupations have converged."""
        gamma_u_vec = 10.0 * self.delta_u  # heuristic γ_u

        try:
            U, V, E, _, _ = lbn.run_numerical_solution(
                lbn.fire_and_hire_workers,
                self.t_sim_full,
                self.delta_u,
                self.delta_v,
                gamma_u_vec,  # γ_u (new)
                self.gamma_v,
                self.E0,
                self.U0,
                self.V0,
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
        except Exception as exc:
            warnings.warn(
                f"Baseline model run failed: {exc}. Using full horizon {self.t_sim_full}."
            )
            return self.t_sim_full

        # max absolute change in total filled ‑vs‑ vacant positions
        tot_change = np.abs(np.diff(E + V, axis=0))  # (t‑1, N)
        converged_counts = np.sum(tot_change < CONV_TOL, axis=1)
        target = int(0.95 * len(self.E0))
        idxs = np.where(converged_counts >= target)[0]
        if idxs.size:
            # pad with extra steps to make sure we are in steady state
            horizon = int(idxs[0] + 20)
            return min(horizon, self.t_sim_full)
        return self.t_sim_full // 2

    # ------------------------------------------------------------------
    def _run_model(
        self,
        delta_v_vec: NDArrayFloat,
        gamma_v_vec: NDArrayFloat,
        t_steps: int,
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Wrapper around *labornet* with manual memoisation."""

        gamma_u_vec = 10.0 * self.delta_u  # simple broadcast heuristic

        # build a *stable* cache key: round floats to 12 decimals → bytes
        key = (
            np.round(delta_v_vec, 12).tobytes()
            + np.round(gamma_v_vec, 12).tobytes()
            + struct.pack("<Q", t_steps)  # 64‑bit to avoid overflow
        )

        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]

        # -------------------------- run solver -----------------------------
        U, V, E, _, _ = lbn.run_numerical_solution(
            lbn.fire_and_hire_workers,
            t_steps,
            self.delta_u,
            delta_v_vec,
            gamma_u_vec,  # γ_u
            gamma_v_vec,
            self.E0,
            self.U0,
            self.V0,
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

        with self._cache_lock:
            if len(self._cache) >= CACHE_MAX_SIZE:
                # FIFO eviction of 100 oldest entries
                for old_key in list(self._cache.keys())[:100]:
                    del self._cache[old_key]
            self._cache[key] = (U, V, E)
        return U, V, E

    # ------------------------------------------------------------------
    def _objective_single(self, idx: int, params: Sequence[float]) -> float:
        """Weighted squared error for a single occupation (internal to optimiser)."""
        dvi, ri = params
        dvi = float(np.clip(dvi, DELTA_V_MIN, DELTA_V_MAX))
        ri = float(np.clip(ri, R_MIN, R_MAX))

        # prepare parameter vectors for this trial
        delta_v_tmp = self.delta_v.copy()
        delta_v_tmp[idx] = dvi
        gamma_v_tmp = self.gamma_v.copy()
        gamma_v_tmp[idx] = max(ri * dvi, MIN_GAMMA)

        # simulate
        U_fin, V_fin, E_fin = self._run_model(delta_v_tmp, gamma_v_tmp, self.t_sim_eff)
        vac_model = V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], EPS_DENOM)

        weight = self.top_weights[np.where(self.top_idx == idx)[0][0]]
        diff = vac_model[idx] - self.vac_emp[idx]
        return float(weight * diff * diff)

    # ------------------------------------------------------------------
    def _fit_one_occ(self, idx: int) -> Tuple[int, float, float]:
        """Local 2‑D minimisation for occupation *idx* → returns *(idx, δ_v*, γ_v*)*."""
        du = self.delta_u[idx]
        dv0 = self.delta_v[idx]
        ri0 = 10.0  # fallback γ/δ ratio
        ed = self.E0[idx]
        vd = self.vac_emp[idx]
        dd = self.D0[idx]

        # analytic γ guess (eq. S60 in SI) — fall back to ratio 10 if undefined
        gap = abs(du - dv0)
        denom_term = 1.0 - dv0
        target_frac = 1.0 - (dd - vd) / ed if ed > 0 else 0.5
        if denom_term > 0 and 0 < target_frac < 1 and gap > 0:
            gamma_guess = gap / (denom_term * target_frac)
            ri0 = float(np.clip(gamma_guess / dv0, R_MIN, R_MAX))

        bounds = [(DELTA_V_MIN, DELTA_V_MAX), (R_MIN, R_MAX)]
        res = minimize(
            lambda x: self._objective_single(idx, x),
            x0=np.array([dv0, ri0], dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
        )
        dvi_opt, ri_opt = res.x
        gamma_opt = max(ri_opt * dvi_opt, MIN_GAMMA)
        return idx, float(dvi_opt), float(gamma_opt)

    # ------------------------------------------------------------------
    def fit(self) -> Tuple[NDArrayFloat, NDArrayFloat, List[float]]:
        """Run block‑coordinate descent and return fitted vectors and RMSE log."""
        rmse_log: List[float] = []

        for sweep in range(self.max_sweeps):
            # ---------------- per‑occupation optimisation (parallel) --------
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(self._fit_one_occ, int(idx)): idx for idx in self.top_idx
                }
                for fut in as_completed(futures):
                    idx, dv_new, gv_new = fut.result()
                    self.delta_v[idx] = dv_new
                    self.gamma_v[idx] = gv_new

            # ---------------- evaluate new RMSE ---------------------------
            U_fin, V_fin, E_fin = self._run_model(
                self.delta_v, self.gamma_v, self.t_sim_eff
            )
            vac_model_topk = (
                V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], EPS_DENOM)
            )[self.top_idx]
            vac_emp_topk = self.vac_emp[self.top_idx]
            mse = float(np.sum(self.top_weights * (vac_model_topk - vac_emp_topk) ** 2))
            rmse = float(np.sqrt(mse))
            rmse_log.append(rmse)

            if sweep > 0 and abs(rmse_log[-2] - rmse) < self.tol_rmse:
                break

        return self.delta_v.copy(), self.gamma_v.copy(), rmse_log


# ---------------------------------------------------------------------------
# example usage (quick self‑test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    N = 10
    np.random.seed(0)
    delta_u = np.random.uniform(0.01, 0.05, size=N)
    delta_v_init = np.random.uniform(0.01, 0.05, size=N)
    gamma_v_init = np.maximum(MIN_GAMMA, 10.0 * delta_v_init)
    E0 = np.random.randint(50, 500, size=N).astype(float)
    U0 = np.random.randint(5, 50, size=N).astype(float)
    V0 = np.random.randint(1, 20, size=N).astype(float)
    D0 = E0 + U0 + V0
    vac_emp = V0 / np.maximum(V0 + E0, EPS_DENOM)
    A = np.eye(N)
    tau = 3
    t_sim = 200

    calibrator = DeltaGammaTopKCalibrator(
        delta_u,
        delta_v_init,
        gamma_v_init,
        vac_emp,
        E0,
        U0,
        V0,
        D0,
        t_sim,
        A,
        tau,
        top_k=5,
        n_workers=4,
    )
    dv_fit, gv_fit, rmse_history = calibrator.fit()
    print("Fitted δ_v:", dv_fit)
    print("Fitted γ_v:", gv_fit)
    print("RMSE log:", rmse_history)

