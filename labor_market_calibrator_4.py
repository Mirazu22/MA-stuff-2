from __future__ import annotations

import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

import labornet as lbn

# ---------------------------------------------------------------------------
# Types & constants
# ---------------------------------------------------------------------------
NDArrayFloat = npt.NDArray[np.floating]

DELTA_V_MIN = 1e-6   # lower bound for δ_v
DELTA_V_MAX = 0.10   # upper bound for δ_v
T_SHOCK_LARGE = 9_999  # effectively no demand shock
EPS_DENOM = 1e-12
CACHE_MAX_SIZE = 1_000


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------
class DeltaVTopKCalibrator:
    """Calibrate δ_v for the *K* largest occupations with γ_v = 10⋅δ_v.

    Parameters
    ----------
    delta_u : array_like
        Spontaneous separation probabilities (length *N*).
    delta_v_init : array_like
        Initial guesses for spontaneous vacancy probabilities.
    vacancy_emp : array_like
        Empirical steady‑state vacancy rates (``V / (E+V)``).
    E0, U0, V0 : array_like
        Initial employment, unemployment and vacancy stocks.
    D0 : array_like
        Target labour demand (``E0 + U0 + V0``).
    t_sim : int
        *Explicit* simulation horizon supplied by the caller.
    A : ndarray
        Adjacency / matching matrix (``N × N``).
    tau : int
        Long‑term unemployment threshold passed to *labornet*.
    top_k : int, optional
        Number of largest occupations (by employment) to calibrate.
    n_workers : int | None, optional
        Size of the thread‑pool executing per‑occupation fits.
    max_sweeps : int, optional
        Maximum block‑coordinate descent sweeps.
    tol_rmse : float, optional
        Early‑stopping tolerance on RMSE improvement between sweeps.
    """

    def __init__(
        self,
        delta_u: NDArrayFloat,
        delta_v_init: NDArrayFloat,
        vacancy_emp: NDArrayFloat,
        E0: NDArrayFloat,
        U0: NDArrayFloat,
        V0: NDArrayFloat,
        D0: NDArrayFloat,
        t_sim: int,
        A: NDArrayFloat,
        tau: int,
        *,
        top_k: int = 100,
        n_workers: int | None = None,
        max_sweeps: int = 5,
        tol_rmse: float = 1e-4,
    ) -> None:
        # ---- input validation ------------------------------------------------
        n_occ = len(delta_u)
        if A.shape != (n_occ, n_occ):
            raise ValueError("A matrix shape does not match number of occupations.")
        if not (len(delta_v_init) == len(vacancy_emp) == n_occ):
            raise ValueError("All input arrays must have the same length.")
        if not (1 <= top_k <= n_occ):
            raise ValueError("top_k must be between 1 and number of occupations.")
        if max_sweeps < 1:
            raise ValueError("max_sweeps must be >= 1.")

        # ---- store copies ----------------------------------------------------
        self.delta_u = np.asarray(delta_u, float).copy()
        self.delta_v = np.clip(np.asarray(delta_v_init, float), DELTA_V_MIN, DELTA_V_MAX)
        self.vac_emp = np.asarray(vacancy_emp, float).copy()
        self.E0 = np.asarray(E0, float).copy()
        self.U0 = np.asarray(U0, float).copy()
        self.V0 = np.asarray(V0, float).copy()
        self.D0 = np.asarray(D0, float).copy()
        self.A = np.asarray(A, float).copy()
        self.tau = int(tau)

        # largest occupations by employment
        self.sorted_idx = np.argsort(-self.E0)
        self.top_idx = self.sorted_idx[:top_k]
        emp_top = self.E0[self.top_idx]
        self.top_weights = emp_top / emp_top.sum()

        # execution params
        self.t_sim = int(t_sim)
        self.n_workers = n_workers or min(len(self.top_idx), max(1, threading.active_count()))
        self.max_sweeps = max_sweeps
        self.tol_rmse = tol_rmse

        # simple FIFO cache for model runs
        self._cache: dict[bytes, tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]] = {}
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _run_model(
        self,
        delta_v_vec: NDArrayFloat,
        t_steps: int,
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Wrapper around *labornet.run_numerical_solution* with memoisation."""
        gamma_u_vec = 10.0 * self.delta_u
        gamma_v_vec = 10.0 * delta_v_vec

        key = np.round(delta_v_vec, 12).tobytes()

        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]

        try:
            U, V, E, _, _ = lbn.run_numerical_solution(
                lbn.fire_and_hire_workers,
                t_steps,
                self.delta_u,
                delta_v_vec,
                gamma_u_vec,
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
        except Exception as exc:
            warnings.warn(f"Model run failed: {exc}")
            raise

        with self._cache_lock:
            if len(self._cache) >= CACHE_MAX_SIZE:
                for old_key in list(self._cache)[:100]:
                    del self._cache[old_key]
            self._cache[key] = (U, V, E)
        return U, V, E

    # ------------------------------------------------------------------
    def _objective_single(self, idx: int, dv: float) -> float:
        dv = float(np.clip(dv, DELTA_V_MIN, DELTA_V_MAX))
        delta_v_tmp = self.delta_v.copy()
        delta_v_tmp[idx] = dv

        # run deterministic solver
        U_fin, V_fin, E_fin = self._run_model(delta_v_tmp, self.t_sim)
        vac_model = V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], EPS_DENOM)

        weight = self.top_weights[np.where(self.top_idx == idx)[0][0]]
        diff = vac_model[idx] - self.vac_emp[idx]
        return float(weight * diff * diff)

    # ------------------------------------------------------------------
    def _fit_one_occ(self, idx: int) -> Tuple[int, float]:
        dv0 = self.delta_v[idx]
        res = minimize_scalar(
            lambda x: self._objective_single(idx, x),
            bounds=(DELTA_V_MIN, DELTA_V_MAX),
            method="bounded",
        )
        return idx, float(res.x)

    # ------------------------------------------------------------------
    def fit(self) -> Tuple[NDArrayFloat, NDArrayFloat, List[float]]:
        """Run block‑coordinate descent and return fitted δ_v, γ_v and RMSE log."""
        rmse_log: List[float] = []

        for sweep in range(self.max_sweeps):
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {executor.submit(self._fit_one_occ, int(idx)): idx for idx in self.top_idx}
                for fut in as_completed(futures):
                    idx, dv_new = fut.result()
                    self.delta_v[idx] = dv_new

            # keep γ_v internally consistent
            self.gamma_v = 10.0 * self.delta_v

            # evaluate RMSE across top‑K
            U_fin, V_fin, E_fin = self._run_model(self.delta_v, self.t_sim)
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
# Quick self‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    N = 10
    delta_u = np.random.uniform(0.01, 0.05, size=N)
    delta_v_init = np.random.uniform(0.01, 0.05, size=N)
    gamma_v_init = 10.0 * delta_v_init  # ignored but kept for API compatibility
    E0 = np.random.randint(50, 500, size=N).astype(float)
    U0 = np.random.randint(5, 50, size=N).astype(float)
    V0 = np.random.randint(1, 20, size=N).astype(float)
    D0 = E0 + U0 + V0
    vac_emp = V0 / np.maximum(V0 + E0, EPS_DENOM)
    A = np.eye(N)
    tau = 3

    calibrator = DeltaVTopKCalibrator(
        delta_u,
        delta_v_init,
        vac_emp,
        E0,
        U0,
        V0,
        D0,
        t_sim=200,
        A=A,
        tau=tau,
        top_k=5,
    )
    dv_fit, gv_fit, rmse_history = calibrator.fit()
    print("Fitted δ_v:", dv_fit)
    print("Fitted γ_v:", gv_fit)
    print("RMSE log:", rmse_history)
