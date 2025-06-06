def _fit_one_occ(self, idx: int) -> Tuple[int, float]:
    # snapshot “other” δ_v values before starting the 1D search
    baseline = self.delta_v.copy()

    def obj(dv: float) -> float:
        dv_clipped = float(np.clip(dv, self.config.delta_v_min, self.config.delta_v_max))
        dv_vec = baseline.copy()
        dv_vec[idx] = dv_clipped

        U_fin, V_fin, E_fin = self._run_model(dv_vec, self.t_sim)
        vac_model = V_fin[-1] / np.maximum(V_fin[-1] + E_fin[-1], self.config.eps_denom)

        weight = self.idx_to_weight[idx]
        diff = vac_model[idx] - self.vac_emp[idx]
        return float(weight * diff * diff)

    res = minimize_scalar(
        obj,
        bounds=(self.config.delta_v_min, self.config.delta_v_max),
        method="bounded",
    )

    return idx, float(res.x)
