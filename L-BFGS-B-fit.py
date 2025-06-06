    def fit(self) -> Tuple[NDArrayFloat, NDArrayFloat, List[float]]:
        """Run L-BFGS-B on the top-K occupations, printing diagnostics to help tune maxiter, ftol, etc."""
        from scipy.optimize import minimize, Bounds

        # Objective: for a candidate top-K δ_v vector, run the model once and compute weighted RMSE.
        def objective(delta_v_topk: NDArrayFloat) -> float:
            candidate = self.delta_v.copy()
            candidate[self.top_idx] = delta_v_topk

            U_fin, V_fin, E_fin = self._run_model(candidate, self.t_sim)
            vac_rates = V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], self.config.eps_denom)

            diff = vac_rates[self.top_idx] - self.vac_emp[self.top_idx]
            weighted_mse = np.sum(self.top_weights * (diff ** 2))
            return float(np.sqrt(weighted_mse))

        # 1) Build and clip the initial guess for top-K δ_v
        x0_raw = self.delta_v[self.top_idx].copy()
        x0 = np.clip(x0_raw, self.config.delta_v_min, self.config.delta_v_max)

        # 2) Set up bounds for each top-K entry
        K = len(self.top_idx)
        lower = np.full(K, self.config.delta_v_min)
        upper = np.full(K, self.config.delta_v_max)
        bounds = Bounds(lower, upper)

        # 3) Run L-BFGS-B
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 5 * K,   # e.g. 5 × 50 = 250 for top_k=50
                "ftol":   1e-9,     # change if you see very small final improvement
            }
        )

        # 4) Print solver diagnostics for tuning
        print("L-BFGS-B results:")
        print(f"  success: {res.success}")
        print(f"  status : {res.status}")
        print(f"  nit    : {res.nit}")
        print(f"  fun    : {res.fun:.6e}")        # final objective value (RMSE)
        print(f"  message: {res.message}")

        # 5) Check for convergence failure
        if not res.success:
            raise RuntimeError(f"L-BFGS-B did not converge: {res.message}")

        # 6) Update self.delta_v with the optimized top-K block
        self.delta_v[self.top_idx] = res.x
        final_gamma_v = self.config.gamma_multiplier * self.delta_v

        # 7) (Optional) compute final RMSE for consistency
        U_fin, V_fin, E_fin = self._run_model(self.delta_v, self.t_sim)
        vac_rates = V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], self.config.eps_denom)
        diff = vac_rates[self.top_idx] - self.vac_emp[self.top_idx]
        weighted_mse = np.sum(self.top_weights * (diff ** 2))
        final_rmse = float(np.sqrt(weighted_mse))

        return self.delta_v.copy(), final_gamma_v.copy(), [final_rmse]

