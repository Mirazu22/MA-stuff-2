from __future__ import annotations

"""Delta‑v calibrator with robust fallbacks.

This version patches several critical issues that surfaced during review:
1. **Correct bottom‑N selection** – now based purely on the employment vector.
2. **Single‑thread execution** – no multithreading diversity; `n_workers` is hard‑wired to 1.
3. **LRU cache accounting** – memory tallies stay consistent on key updates.
4. **Stable cache keys** – quantise δ_v to integer ticks to avoid FP drift.
   (tick size == `config.delta_v_min`.)
All previous functionality – grid‑search fallback, occupation locking, bottom‑N seeding – is preserved.
"""

import dataclasses
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

import labornet as lbn

NDArrayFloat = npt.NDArray[np.floating]


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CalibrationConfig:
    # --- core bounds ----------------------------------------------------------
    delta_v_min: float = 1e-6
    delta_v_max: float = 0.10
    gamma_multiplier: float = 10.0
    # --- cache ---------------------------------------------------------------
    cache_max_size: int = 1000
    cache_max_memory_mb: float = 50.0
    # --- numerical epsilons --------------------------------------------------
    eps_denom: float = 1e-12
    t_shock_large: int = 9999
    # --- optimiser behaviour -------------------------------------------------
    damping_alpha: float = 0.2
    inner_loop_max: int = 3
    inner_loop_tol: float = 1e-5
    # --- grid‑search fallback -------------------------------------------------
    grid_search_points: int = 15
    # --- bottom‑N seeding -----------------------------------------------------
    bottom_n: int = 0          # 0 disables feature
    similarity_metric: str = "abs_distance"


DEFAULT_CONFIG = CalibrationConfig()


# -----------------------------------------------------------------------------
# SIMPLE LRU CACHE WITH FIXED ACCOUNTING
# -----------------------------------------------------------------------------

class LRUCache:
    def __init__(self, maxsize: int, max_memory_mb: float) -> None:
        self.maxsize = maxsize
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[bytes, Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]] = (
            OrderedDict()
        )
        self.current_memory = 0

    @staticmethod
    def _array_size(arrays: Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]) -> int:
        return sum(a.nbytes for a in arrays)

    def get(
        self, key: bytes
    ) -> Optional[Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(
        self, key: bytes, value: Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]
    ) -> None:
        item_size = self._array_size(value)
        if item_size > self.max_memory_bytes:
            warnings.warn(
                f"Item of size {item_size / 1024 ** 2:.2f} MB exceeds cache limit; skipping."
            )
            return

        if key in self.cache:  # replace existing → remove first, then re‑insert
            self.current_memory -= self._array_size(self.cache[key])
            del self.cache[key]

        # evict LRU until space fits
        while (
            self.cache
            and (len(self.cache) >= self.maxsize or self.current_memory + item_size > self.max_memory_bytes)
        ):
            _, old_val = self.cache.popitem(last=False)
            self.current_memory -= self._array_size(old_val)

        self.cache[key] = value
        self.current_memory += item_size


# -----------------------------------------------------------------------------
# MAIN CALIBRATOR
# -----------------------------------------------------------------------------

class DeltaVTopKCalibrator:
    """Calibrates δ_v for top‑K occupations with robust fall‑backs."""

    # ---------------------------------------------------------------------
    # construction helpers
    # ---------------------------------------------------------------------

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
        n_workers: int | None = None,  # kept for API compatibility, but ignored.
        max_sweeps: int = 5,
        tol_rmse: float = 1e-4,
        config: CalibrationConfig | None = None,
    ) -> None:

        self.config = config or DEFAULT_CONFIG
        self._validate_inputs(
            delta_u,
            delta_v_init,
            vacancy_emp,
            E0,
            U0,
            V0,
            D0,
            A,
            tau,
            top_k,
            max_sweeps,
            t_sim,
        )

        # --- store immutable data ----------------------------------------
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
        self.t_sim = int(t_sim)

        # --- top‑K index bookkeeping -------------------------------------
        self.sorted_idx = np.argsort(-self.E0)  # descending employment
        self.top_idx = self.sorted_idx[:top_k]
        emp_top = self.E0[self.top_idx]
        self.top_weights = emp_top / emp_top.sum()
        self.idx_to_weight: Dict[int, float] = dict(zip(self.top_idx, self.top_weights))

        # --- calibration flow control ------------------------------------
        self.max_sweeps = max_sweeps
        self.tol_rmse = tol_rmse

        # --- cache + locking ---------------------------------------------
        self._cache = LRUCache(self.config.cache_max_size, self.config.cache_max_memory_mb)
        self._locked_idx: Set[int] = set()  # occupations that failed optimisation

        # --- execution mode ----------------------------------------------
        self.n_workers: int = 1  # explicit single‑thread mode

    # ---------------------------------------------------------------------
    # validation & helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        delta_u,
        delta_v_init,
        vacancy_emp,
        E0,
        U0,
        V0,
        D0,
        A,
        tau,
        top_k,
        max_sweeps,
        t_sim,
    ) -> None:
        n_occ = len(delta_u)
        if A.shape != (n_occ, n_occ):
            raise ValueError("A matrix shape does not match number of occupations.")
        arrays = [delta_v_init, vacancy_emp, E0, U0, V0, D0]
        if not all(len(a) == n_occ for a in arrays):
            raise ValueError("All input arrays must have the same length.")
        if not (1 <= top_k <= n_occ):
            raise ValueError("top_k must be between 1 and number of occupations.")
        if max_sweeps < 1:
            raise ValueError("max_sweeps must be >= 1.")
        if tau < 0:
            raise ValueError("tau must be non‑negative.")
        if t_sim <= 0:
            raise ValueError("t_sim must be positive.")
        for arr in [delta_u] + arrays + [A]:
            if not np.isfinite(arr).all():
                raise ValueError("Input arrays contain non‑finite values.")

    # ---------------------------------------------------------------------
    # model runner with stable cache key
    # ---------------------------------------------------------------------

    def _run_model(
        self, delta_v_vec: NDArrayFloat, t_steps: int
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Run labornet simulation with caching."""
        gamma_u_vec = self.config.gamma_multiplier * self.delta_u
        gamma_v_vec = self.config.gamma_multiplier * delta_v_vec

        key = (
            np.round(delta_v_vec / self.config.delta_v_min).astype(np.int32).tobytes()
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

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
        result = (U, V, E)
        self._cache.put(key, result)
        return result

    # ---------------------------------------------------------------------
    # objective helpers
    # ---------------------------------------------------------------------

    def _objective_single(
        self, idx: int, dv: float, base_dv: NDArrayFloat
    ) -> float:
        dv = float(np.clip(dv, self.config.delta_v_min, self.config.delta_v_max))
        delta_v_tmp = base_dv.copy()
        delta_v_tmp[idx] = dv
        U_fin, V_fin, E_fin = self._run_model(delta_v_tmp, self.t_sim)
        vac_model = V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], self.config.eps_denom)
        weight = self.idx_to_weight.get(idx, 1.0)
        diff = vac_model[idx] - self.vac_emp[idx]
        return float(weight * diff * diff)

    # ---------------------------------------------------------------------
    # grid‑search fallback + optimiser wrapper
    # ---------------------------------------------------------------------

    def _grid_search(
        self, idx: int, base_dv: NDArrayFloat
    ) -> Tuple[float, float]:
        """Return (best_dv, best_err) from uniform grid."""
        pts = np.linspace(
            self.config.delta_v_min,
            self.config.delta_v_max,
            self.config.grid_search_points,
        )
        errs = [self._objective_single(idx, x, base_dv) for x in pts]
        best_i = int(np.argmin(errs))
        return float(pts[best_i]), float(errs[best_i])

    def _fit_one_occ(
        self, idx: int, base_dv: NDArrayFloat
    ) -> Tuple[int, float]:
        """Optimise δ_v for a single occupation with robust fallback."""
        if idx in self._locked_idx:
            return idx, base_dv[idx]

        # first-pass solver
        res = minimize_scalar(
            lambda x: self._objective_single(idx, x, base_dv),
            bounds=(self.config.delta_v_min, self.config.delta_v_max),
            method="bounded",
        )
        old_dv = base_dv[idx]
        new_dv = float(res.x)
        # accept if solver converged and changed value meaningfully
        if res.success and abs(new_dv - old_dv) > 1e-12:
            return idx, new_dv
        # test for no-change + vacancy error
        no_change = abs(new_dv - old_dv) < 1e-12
        U_fin, V_fin, E_fin = self._run_model(np.where(np.arange(len(base_dv))==idx, new_dv, base_dv), self.t_sim)
        model_rate = (V_fin[-1][idx] / max(V_fin[-1][idx] + E_fin[-1][idx], self.config.eps_denom))
        emp_rate = self.vac_emp[idx]
        rel_error = abs(model_rate - emp_rate) / max(emp_rate, self.config.eps_denom)
        if not res.success or (no_change and rel_error > 0.30):
            # fallback stage 1: grid search
            dv_grid, err_grid = self._grid_search(idx, base_dv)
            # fallback stage 2: refined solver around grid best
            half_span = 0.25 * (self.config.delta_v_max - self.config.delta_v_min)
            lo = max(self.config.delta_v_min, dv_grid - half_span)
            hi = min(self.config.delta_v_max, dv_grid + half_span)
            res2 = minimize_scalar(
                lambda x: self._objective_single(idx, x, base_dv),
                bounds=(lo, hi),
                method="bounded",
            )
            if res2.success and abs(res2.x - dv_grid) > 1e-12:
                return idx, float(res2.x)
            warnings.warn(f"Locking occupation {idx} – optimiser stalled.")
            self._locked_idx.add(idx)
            return idx, dv_grid
        # otherwise accept new_dv
        return idx, new_dv

    # ---------------------------------------------------------------------
    # bottom‑N seeding
    # ---------------------------------------------------------------------

    def _seed_bottom_n(self) -> None:
        N = self.config.bottom_n
        if N <= 0:
            return

        non_top_idx = [i for i in range(len(self.E0)) if i not in self.top_idx]
        bottom_idx = sorted(non_top_idx, key=lambda x: self.E0[x])[:N]

        for b_idx in bottom_idx:
            distances = np.abs(self.vac_emp[self.top_idx] - self.vac_emp[b_idx])
            nearest_top = int(self.top_idx[int(np.argmin(distances))])
            self.delta_v[b_idx] = self.delta_v[nearest_top]

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------

    def fit(self) -> Tuple[NDArrayFloat, NDArrayFloat, List[float]]:
        rmse_log: List[float] = []

        for sweep in range(self.max_sweeps):
            print(f"\n=== Sweep {sweep + 1}/{self.max_sweeps} ===")
            for inner in range(self.config.inner_loop_max):
                max_delta = 0.0
                base_dv = self.delta_v.copy()

                for idx in self.top_idx:
                    if idx in self._locked_idx:
                        continue
                    _, dv_new = self._fit_one_occ(idx, base_dv)
                    dv_old = self.delta_v[idx]
                    dv_updated = (
                        (1 - self.config.damping_alpha) * dv_old
                        + self.config.damping_alpha * dv_new
                    )
                    self.delta_v[idx] = dv_updated
                    max_delta = max(max_delta, abs(dv_updated - dv_old))

                print(f"  max |Δδ_v| = {max_delta:.2e}")
                if max_delta < self.config.inner_loop_tol:
                    break

            U_fin, V_fin, E_fin = self._run_model(self.delta_v, self.t_sim)
            vac_final = V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], self.config.eps_denom)
            model_top = vac_final[self.top_idx]
            emp_top = self.vac_emp[self.top_idx]
            mse = float(np.sum(self.top_weights * (model_top - emp_top) ** 2))
            rmse = float(np.sqrt(mse))
            rmse_log.append(rmse)
            print(f"  RMSE = {rmse:.6e}")
            if sweep > 0 and abs(rmse_log[-2] - rmse) < self.tol_rmse:
                print("  Early‑stop: RMSE improvement below tolerance.")
                break

        self._seed_bottom_n()

        final_gamma_v = self.config.gamma_multiplier * self.delta_v
        return self.delta_v.copy(), final_gamma_v.copy(), rmse_log

    # ---------------------------------------------------------------------
    # diagnostics
    # ---------------------------------------------------------------------

    def report_calibration(self, delta_v_init: NDArrayFloat) -> None:
        print("\n=== Final calibration report ===")
        U_fin, V_fin, E_fin = self._run_model(self.delta_v, self.t_sim)
        vac_model = V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], self.config.eps_denom)

        for idx in self.top_idx:
            init_val = float(delta_v_init[idx])
            final_val = float(self.delta_v[idx])
            model_rate = float(vac_model[idx])
            target = float(self.vac_emp[idx])
            pct_diff = (
                float("nan") if target == 0 else 100.0 * (model_rate - target) / target
            )
            print(
                f"occ {idx:>3d}: δ_v {init_val:.6f} ➜ {final_val:.6f}; "
                f"model {model_rate:.6f}, target {target:.6f} ({pct_diff:.2f} %)"
            )
