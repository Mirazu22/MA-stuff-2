from __future__ import annotations

import dataclasses
import threading
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

import labornet as lbn

NDArrayFloat = npt.NDArray[np.floating]

@dataclasses.dataclass(frozen=True)
class CalibrationConfig:
    delta_v_min: float = 1e-6
    delta_v_max: float = 0.10
    gamma_multiplier: float = 10.0
    cache_max_size: int = 1000
    eps_denom: float = 1e-12
    t_shock_large: int = 9999
    cache_cleanup_batch: int = 100
    damping_alpha: float = 0.2
    inner_loop_max: int = 3
    inner_loop_tol: float = 1e-5
    cache_max_memory_mb: float = 50.0

DEFAULT_CONFIG = CalibrationConfig()

class LRUCache:
    def __init__(self, maxsize: int, max_memory_mb: float = 50.0) -> None:
        self.maxsize = maxsize
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[bytes, Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]] = OrderedDict()
        self.lock = threading.Lock()
        self.current_memory = 0

    def _array_size(self, arrays: Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]) -> int:
        return sum(arr.nbytes for arr in arrays)

    def get(self, key: bytes) -> Optional[Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        return None

    def put(self, key: bytes, value: Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]) -> None:
        item_size = self._array_size(value)

        if item_size > self.max_memory_bytes:
            warnings.warn(f"Item of size {item_size/1024**2:.2f}MB exceeds max memory; skipping cache.")
            return

        with self.lock:
            if key in self.cache:
                self.current_memory -= self._array_size(self.cache[key])
                self.cache.move_to_end(key)
            else:
                while (len(self.cache) >= self.maxsize or 
                       self.current_memory + item_size > self.max_memory_bytes) and self.cache:
                    _, old_value = self.cache.popitem(last=False)
                    self.current_memory -= self._array_size(old_value)

            self.cache[key] = value
            self.current_memory += item_size

class DeltaVTopKCalibrator:
    def __init__(self, delta_u: NDArrayFloat, delta_v_init: NDArrayFloat, vacancy_emp: NDArrayFloat,
                 E0: NDArrayFloat, U0: NDArrayFloat, V0: NDArrayFloat, D0: NDArrayFloat,
                 t_sim: int, A: NDArrayFloat, tau: int, *, top_k: int = 100,
                 n_workers: int | None = None, max_sweeps: int = 5, tol_rmse: float = 1e-4,
                 config: CalibrationConfig | None = None) -> None:

        self.config = config or DEFAULT_CONFIG
        self._validate_inputs(delta_u, delta_v_init, vacancy_emp, E0, U0, V0, D0, A, tau, top_k, max_sweeps, t_sim)
        n_occ = len(delta_u)

        self.delta_u = np.asarray(delta_u, float).copy()
        self.delta_v = np.clip(np.asarray(delta_v_init, float), self.config.delta_v_min, self.config.delta_v_max)
        self.vac_emp = np.asarray(vacancy_emp, float).copy()
        self.E0 = np.asarray(E0, float).copy()
        self.U0 = np.asarray(U0, float).copy()
        self.V0 = np.asarray(V0, float).copy()
        self.D0 = np.asarray(D0, float).copy()
        self.A = np.asarray(A, float).copy()
        self.tau = int(tau)

        self.sorted_idx = np.argsort(-self.E0)
        self.top_idx = self.sorted_idx[:top_k]
        emp_top = self.E0[self.top_idx]
        self.top_weights = emp_top / emp_top.sum()

        for idx in self.top_idx:
            if self.E0[idx] <= 0:
                warnings.warn(f"Top-K occupation {idx} has E0 = {self.E0[idx]}.")
            if self.V0[idx] <= 0:
                warnings.warn(f"Top-K occupation {idx} has V0 = {self.V0[idx]}.")
            if self.U0[idx] <= 0:
                warnings.warn(f"Top-K occupation {idx} has U0 = {self.U0[idx]}.")

        self.idx_to_weight: Dict[int, float] = dict(zip(self.top_idx, self.top_weights))
        self.t_sim = int(t_sim)
        self.n_workers = n_workers or min(len(self.top_idx), max(1, threading.active_count()))
        self.max_sweeps = max_sweeps
        self.tol_rmse = tol_rmse
        self._cache = LRUCache(self.config.cache_max_size, self.config.cache_max_memory_mb)

    def _validate_inputs(self, delta_u, delta_v_init, vacancy_emp, E0, U0, V0, D0, A, tau, top_k, max_sweeps, t_sim):
        n_occ = len(delta_u)
        if A.shape != (n_occ, n_occ):
            raise ValueError("A matrix shape does not match number of occupations.")
        if not (len(delta_v_init) == len(vacancy_emp) == len(E0) == len(U0) == len(V0) == len(D0) == n_occ):
            raise ValueError("All input arrays must have the same length.")
        if not (1 <= top_k <= n_occ):
            raise ValueError("top_k must be between 1 and number of occupations.")
        if max_sweeps < 1:
            raise ValueError("max_sweeps must be >= 1.")
        if tau < 0:
            raise ValueError("tau must be non-negative.")
        if t_sim <= 0:
            raise ValueError("t_sim must be a positive integer.")
        for arr in [delta_u, delta_v_init, vacancy_emp, E0, U0, V0, D0, A]:
            if not np.isfinite(arr).all():
                raise ValueError("Input arrays contain non-finite values.")

    def _run_model(self, delta_v_vec: NDArrayFloat, t_steps: int) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        gamma_u_vec = self.config.gamma_multiplier * self.delta_u
        gamma_v_vec = self.config.gamma_multiplier * delta_v_vec
        key = np.round(delta_v_vec, 12).tobytes()
        cached_result = self._cache.get(key)
        if cached_result is not None:
            return cached_result

        print("Running model simulation with updated delta_v...")
        U, V, E, _, _ = lbn.run_numerical_solution(
            lbn.fire_and_hire_workers, t_steps,
            self.delta_u, delta_v_vec, gamma_u_vec, gamma_v_vec,
            self.E0, self.U0, self.V0,
            target_demand_function=lambda t, *args: self.D0,
            D_0=self.D0, D_f=self.D0, t_shock=self.config.t_shock_large,
            k=0, t_halfsig=0, matching=lbn.matching_probability,
            A_matrix=self.A, τ=self.tau)

        result = (U, V, E)
        self._cache.put(key, result)
        return result

    def _objective_single(self, idx: int, dv: float, base_dv: NDArrayFloat) -> float:
        dv = float(np.clip(dv, self.config.delta_v_min, self.config.delta_v_max))
        delta_v_tmp = base_dv.copy()
        delta_v_tmp[idx] = dv
        U_fin, V_fin, E_fin = self._run_model(delta_v_tmp, self.t_sim)
        vac_model = V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], self.config.eps_denom)
        weight = self.idx_to_weight.get(idx)
        diff = vac_model[idx] - self.vac_emp[idx]
        return float(weight * diff * diff)

    def _fit_one_occ(self, idx: int, base_dv: NDArrayFloat) -> Tuple[int, float]:
        print(f"  Calibrating occupation {idx}...")
        res = minimize_scalar(lambda x: self._objective_single(idx, x, base_dv),
                              bounds=(self.config.delta_v_min, self.config.delta_v_max), method="bounded")
        return idx, float(res.x)

    def fit(self) -> Tuple[NDArrayFloat, NDArrayFloat, List[float]]:
        rmse_log: List[float] = []

        for sweep in range(self.max_sweeps):
            print(f"\n=== Starting sweep {sweep+1}/{self.max_sweeps} ===")

            for inner_loop in range(self.config.inner_loop_max):
                print(f"  Inner loop {inner_loop+1}/{self.config.inner_loop_max}...")
                max_delta = 0.0
                base_dv = self.delta_v.copy()
                for idx in self.top_idx:
                    _, dv_new = self._fit_one_occ(idx, base_dv)
                    dv_old = self.delta_v[idx]
                    dv_updated = (1 - self.config.damping_alpha) * dv_old + self.config.damping_alpha * dv_new
                    self.delta_v[idx] = dv_updated
                    max_delta = max(max_delta, abs(dv_updated - dv_old))
                print(f"    Max delta_v change in inner loop: {max_delta:.2e}")
                if max_delta < self.config.inner_loop_tol:
                    print("    Inner loop converged.")
                    break

            print("  Running full model for RMSE evaluation...")
            U_fin, V_fin, E_fin = self._run_model(self.delta_v, self.t_sim)
            vac_model_topk = (V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], self.config.eps_denom))[self.top_idx]
            vac_emp_topk = self.vac_emp[self.top_idx]
            mse = float(np.sum(self.top_weights * (vac_model_topk - vac_emp_topk) ** 2))
            rmse = float(np.sqrt(mse))
            rmse_log.append(rmse)
            print(f"=== Sweep {sweep+1} complete. RMSE = {rmse:.6f} ===")

            if sweep > 0 and abs(rmse_log[-2] - rmse) < self.tol_rmse:
                print("Early stopping: RMSE improvement below tolerance.")
                break

        final_gamma_v = self.config.gamma_multiplier * self.delta_v
        return self.delta_v.copy(), final_gamma_v.copy(), rmse_log

    def report_calibration(self, delta_v_init: NDArrayFloat) -> None:
        """Prints how δ_v changed for each top-K occupation and the resulting vacancy rates.

        Parameters
        ----------
        delta_v_init : array_like
            The original δ_v values passed into the constructor.
        """
        # Run the model one last time on the fully calibrated δ_v to get final steady‐state
        print("\n=== Final calibration report ===")
        U_fin, V_fin, E_fin = self._run_model(self.delta_v, self.t_sim)
        vac_model = V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], self.config.eps_denom)

        for idx in self.top_idx:
            init_val = float(delta_v_init[idx])
            final_val = float(self.delta_v[idx])
            model_rate = float(vac_model[idx])
            target = float(self.vac_emp[idx])

            if target != 0:
                pct_diff = 100.0 * (model_rate - target) / target
            else:
                pct_diff = float('nan')

            print(
                f"Occupation {idx}: "
                f"δ_v changed from {init_val:.6f} → {final_val:.6f}; "
                f"model vac_rate = {model_rate:.6f}, "
                f"target = {target:.6f}, "
                f"% difference = {pct_diff:.2f}%"
            )
