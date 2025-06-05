from __future__ import annotations

import dataclasses
import threading
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

import labornet as lbn

# ---------------------------------------------------------------------------
# Types & constants
# ---------------------------------------------------------------------------
NDArrayFloat = npt.NDArray[np.floating]


@dataclasses.dataclass(frozen=True)
class CalibrationConfig:
    """Configuration parameters for the calibration process."""
    delta_v_min: float = 1e-6
    delta_v_max: float = 0.10
    gamma_multiplier: float = 10.0
    cache_max_size: int = 1000
    eps_denom: float = 1e-12
    t_shock_large: int = 9999
    cache_cleanup_batch: int = 100


# Default configuration instance
DEFAULT_CONFIG = CalibrationConfig()


# ---------------------------------------------------------------------------
# Thread-safe LRU Cache
# ---------------------------------------------------------------------------
class LRUCache:
    """Thread-safe LRU cache implementation."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self.cache: OrderedDict[
            bytes, Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]
        ] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: bytes) -> Optional[Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]]:
        """Thread-safe: return cache[key] (if present) and mark it as “recently used.”"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        return None

    def put(self, key: bytes, value: Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]) -> None:
        """Thread-safe: put value in cache, evicting oldest if necessary."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            elif len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
            self.cache[key] = value


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
        Empirical steady-state vacancy rates (``V / (E+V)``).
    E0, U0, V0 : array_like
        Initial employment, unemployment and vacancy stocks.
    D0 : array_like
        Target labour demand (``E0 + U0 + V0``).
    t_sim : int
        *Explicit* simulation horizon supplied by the caller.
    A : ndarray
        Adjacency / matching matrix (``N × N``).
    tau : int
        Long-term unemployment threshold passed to *labornet*.
    top_k : int, optional
        Number of largest occupations (by employment) to calibrate.
    n_workers : int | None, optional
        Size of the thread-pool executing per-occupation fits.
    max_sweeps : int, optional
        Maximum block-coordinate descent sweeps.
    tol_rmse : float, optional
        Early-stopping tolerance on RMSE improvement between sweeps.
    config : CalibrationConfig, optional
        Configuration object with calibration parameters.
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
        config: CalibrationConfig | None = None,
    ) -> None:

        self.config = config or DEFAULT_CONFIG

        # ---- input validation ------------------------------------------------
        self._validate_inputs(
            delta_u, delta_v_init, vacancy_emp, E0, U0, V0, D0, A, tau, top_k, max_sweeps, t_sim
        )

        n_occ = len(delta_u)

        # ---- store copies ----------------------------------------------------
        self.delta_u = np.asarray(delta_u, float).copy()
        self.delta_v = np.clip(
            np.asarray(delta_v_init, float),
            self.config.delta_v_min,
            self.config.delta_v_max,
        )
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

        # Emit warnings if any E0, U0, or V0 in top-K are zero
        for idx in self.top_idx:
            if self.E0[idx] <= 0:
                warnings.warn(
                    f"Top-K occupation {idx} has E0 = {self.E0[idx]}. "
                    "Zero employment may lead to degenerate calibration."
                )
            if self.V0[idx] <= 0:
                warnings.warn(
                    f"Top-K occupation {idx} has V0 = {self.V0[idx]}. "
                    "Zero initial vacancies may drive δ_v to its lower bound."
                )
            if self.U0[idx] <= 0:
                warnings.warn(
                    f"Top-K occupation {idx} has U0 = {self.U0[idx]}. "
                    "Zero initial unemployment may influence steady-state flows."
                )

        # Create efficient mapping from index to weight
        self.idx_to_weight: Dict[int, float] = dict(zip(self.top_idx, self.top_weights))

        # execution params
        self.t_sim = int(t_sim)
        self.n_workers = n_workers or min(len(self.top_idx), max(1, threading.active_count()))
        self.max_sweeps = max_sweeps
        self.tol_rmse = tol_rmse

        # thread-safe LRU cache for model runs
        self._cache = LRUCache(self.config.cache_max_size)

    def _validate_inputs(
        self,
        delta_u: NDArrayFloat,
        delta_v_init: NDArrayFloat,
        vacancy_emp: NDArrayFloat,
        E0: NDArrayFloat,
        U0: NDArrayFloat,
        V0: NDArrayFloat,
        D0: NDArrayFloat,
        A: NDArrayFloat,
        tau: int,
        top_k: int,
        max_sweeps: int,
        t_sim: int,
    ) -> None:
        """Validate all input parameters."""
        n_occ = len(delta_u)

        # Shape validation
        if A.shape != (n_occ, n_occ):
            raise ValueError("A matrix shape does not match number of occupations.")
        if not (
            len(delta_v_init) == len(vacancy_emp) == len(E0) == len(U0) == len(V0) == len(D0) == n_occ
        ):
            raise ValueError("All input arrays must have the same length.")

        # Range validation
        if not (1 <= top_k <= n_occ):
            raise ValueError("top_k must be between 1 and number of occupations.")
        if max_sweeps < 1:
            raise ValueError("max_sweeps must be >= 1.")
        if tau < 0:
            raise ValueError("tau must be non-negative.")
        if t_sim <= 0:
            raise ValueError("t_sim must be a positive integer.")

        # Probability validation
        delta_u_arr = np.asarray(delta_u)
        delta_v_arr = np.asarray(delta_v_init)
        vacancy_emp_arr = np.asarray(vacancy_emp)

        if np.any(delta_u_arr < 0) or np.any(delta_u_arr > 1):
            raise ValueError("delta_u must be in [0, 1].")
        if np.any(delta_v_arr < 0) or np.any(delta_v_arr > 1):
            raise ValueError("delta_v_init must be in [0, 1].")
        if np.any(vacancy_emp_arr < 0) or np.any(vacancy_emp_arr > 1):
            raise ValueError("vacancy_emp must be in [0, 1].")

        # Stock validation
        E0_arr, U0_arr, V0_arr, D0_arr = map(np.asarray, [E0, U0, V0, D0])
        if np.any(E0_arr < 0) or np.any(U0_arr < 0) or np.any(V0_arr < 0):
            raise ValueError("Employment, unemployment, and vacancy stocks must be non-negative.")
        if np.any(D0_arr <= 0):
            raise ValueError("Target demand must be positive.")

        # Ensure D0 == E0 + U0 + V0 (within tolerance)
        if not np.allclose(D0_arr, E0_arr + U0_arr + V0_arr, rtol=1e-8, atol=1e-8):
            raise ValueError("D0 must equal E0 + U0 + V0 (within floating-point tolerance).")

        # Check for NaN/infinite values
        arrays_to_check = [delta_u_arr, delta_v_arr, vacancy_emp_arr, E0_arr, U0_arr, V0_arr, D0_arr, A]
        array_names = ["delta_u", "delta_v_init", "vacancy_emp", "E0", "U0", "V0", "D0", "A"]

        for arr, name in zip(arrays_to_check, array_names):
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} contains non-finite values (NaN or inf).")

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _run_model(
        self,
        delta_v_vec: NDArrayFloat,
        t_steps: int,
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Wrapper around *labornet.run_numerical_solution* with memoisation."""

        gamma_u_vec = self.config.gamma_multiplier * self.delta_u
        gamma_v_vec = self.config.gamma_multiplier * delta_v_vec

        key = np.round(delta_v_vec, 12).tobytes()
        # Optionally, you could append t_steps or gamma_multiplier to key if needed

        # Check cache first
        cached_result = self._cache.get(key)
        if cached_result is not None:
            return cached_result

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
                t_shock=self.config.t_shock_large,
                k=0,
                t_halfsig=0,
                matching=lbn.matching_probability,
                A_matrix=self.A,
                τ=self.tau,
            )
        except Exception as exc:
            raise RuntimeError(f"labornet solver failed for δ_v={delta_v_vec}: {exc}") from exc

        result = (U, V, E)
        self._cache.put(key, result)
        return result

    # ------------------------------------------------------------------
    def _objective_single(self, idx: int, dv: float) -> float:
        """Compute objective function for a single occupation."""
        dv = float(np.clip(dv, self.config.delta_v_min, self.config.delta_v_max))
        delta_v_tmp = self.delta_v.copy()
        delta_v_tmp[idx] = dv

        # run deterministic solver
        U_fin, V_fin, E_fin = self._run_model(delta_v_tmp, self.t_sim)
        vac_model = V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], self.config.eps_denom)

        # Use efficient lookup for weight
        weight = self.idx_to_weight.get(idx)
        if weight is None:
            raise ValueError(f"Index {idx} not found in top_idx")

        diff = vac_model[idx] - self.vac_emp[idx]
        return float(weight * diff * diff)

    # ------------------------------------------------------------------
    def _fit_one_occ(self, idx: int) -> Tuple[int, float]:
        """Fit delta_v for one occupation."""
        res = minimize_scalar(
            lambda x: self._objective_single(idx, x),
            bounds=(self.config.delta_v_min, self.config.delta_v_max),
            method="bounded",
        )
        return idx, float(res.x)

    # ------------------------------------------------------------------
    def fit(self) -> Tuple[NDArrayFloat, NDArrayFloat, List[float]]:
        """Run block-coordinate descent and return fitted δ_v, γ_v and RMSE log."""
        rmse_log: List[float] = []

        total_occ = len(self.top_idx)
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for sweep in range(self.max_sweeps):
                print(f"Starting sweep {sweep+1}/{self.max_sweeps}...")

                # (1) Run all per‐occupation minimizations in parallel
                futures = {executor.submit(self._fit_one_occ, int(idx)): idx for idx in self.top_idx}

                completed = 0
                for fut in as_completed(futures):
                    idx, dv_new = fut.result()
                    self.delta_v[idx] = dv_new
                    completed += 1
                    print(f"    [{completed}/{total_occ}] Occupation {idx} fitted")

                # (2) Single “full‐delta_v” model run for RMSE
                U_fin, V_fin, E_fin = self._run_model(self.delta_v, self.t_sim)

                vac_model_topk = (
                    V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], self.config.eps_denom)
                )[self.top_idx]
                vac_emp_topk = self.vac_emp[self.top_idx]
                mse = float(np.sum(self.top_weights * (vac_model_topk - vac_emp_topk) ** 2))
                rmse = float(np.sqrt(mse))
                rmse_log.append(rmse)

                print(f"Sweep {sweep+1} complete. RMSE = {rmse:.6f}")

                # Early stopping check
                if sweep > 0 and abs(rmse_log[-2] - rmse) < self.tol_rmse:
                    print("Early stopping: RMSE improvement below tolerance.")
                    break

        # Return final gamma_v consistently computed
        final_gamma_v = self.config.gamma_multiplier * self.delta_v
        return self.delta_v.copy(), final_gamma_v.copy(), rmse_log


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    N = 10
    delta_u = np.random.uniform(0.01, 0.05, size=N)
    delta_v_init = np.random.uniform(0.01, 0.05, size=N)
    E0 = np.random.randint(50, 500, size=N).astype(float)
    U0 = np.random.randint(5, 50, size=N).astype(float)
    V0 = np.random.randint(1, 20, size=N).astype(float)
    D0 = E0 + U0 + V0
    vac_emp = V0 / np.maximum(V0 + E0, DEFAULT_CONFIG.eps_denom)
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
