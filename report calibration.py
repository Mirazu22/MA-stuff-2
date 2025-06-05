# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 15:06:30 2025

@author: mauhl
"""

def report_calibration(self, delta_v_init: NDArrayFloat) -> None:
    """Prints how δ_v changed for each top-K occupation and the resulting vacancy rates.

    Parameters
    ----------
    delta_v_init : array_like
        The original δ_v values passed into the constructor.
    """
    # Run the model one last time on the fully calibrated δ_v to get final steady‐state
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
