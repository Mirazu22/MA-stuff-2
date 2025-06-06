# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 13:54:57 2025

@author: mauhl
"""

    def fit(self) -> Tuple[NDArrayFloat, NDArrayFloat, List[float]]:
        """Run L-BFGS-B on the top-K occupations instead of block-coordinate descent."""
        from scipy.optimize import minimize, Bounds

        # Objective: given a top-K delta_v vector (length = len(top_idx)),
        # run the model once, compute weighted RMSE on vacancy rates of those top-K occupations.
        def objective(delta_v_topk: NDArrayFloat) -> float:
            full_delta_v = self.delta_v.copy()
            full_delta_v[self.top_idx] = delta_v_topk

            U_fin, V_fin, E_fin = self._run_model(full_delta_v, self.t_sim)
            vac_rates = V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], self.config.eps_denom)

            diff_topk = vac_rates[self.top_idx] - self.vac_emp[self.top_idx]
            weighted_mse = np.sum(self.top_weights * (diff_topk ** 2))
            return float(np.sqrt(weighted_mse))

        # Initial guess: current delta_v values for top-K occupations, clipped to bounds
        x0_raw = self.delta_v[self.top_idx].copy()
        x0 = np.clip(x0_raw, self.config.delta_v_min, self.config.delta_v_max)

        # Bounds: enforce [delta_v_min, delta_v_max] for each top-K entry
        K = len(self.top_idx)
        lower = np.full(K, self.config.delta_v_min)
        upper = np.full(K, self.config.delta_v_max)
        bounds = Bounds(lower, upper)

        # Run L-BFGS-B
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 200,
                "ftol": 1e-9,
            }
        )

        # Check optimization success
        if not res.success:
            raise RuntimeError(f"L-BFGS-B did not converge: {res.message}")

        # Update delta_v for the top-K occupations
        self.delta_v[self.top_idx] = res.x
        final_gamma_v = self.config.gamma_multiplier * self.delta_v

        # Compute final RMSE for logging/return
        U_fin, V_fin, E_fin = self._run_model(self.delta_v, self.t_sim)
        vac_rates = V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], self.config.eps_denom)
        diff_topk = vac_rates[self.top_idx] - self.vac_emp[self.top_idx]
        weighted_mse = np.sum(self.top_weights * (diff_topk ** 2))
        final_rmse = float(np.sqrt(weighted_mse))

        return self.delta_v.copy(), final_gamma_v.copy(), [final_rmse]
