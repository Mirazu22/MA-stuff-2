# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 12:00:29 2025

@author: mauhl
"""

import numpy as np

def period_occ_u_rate(E, U, time_start, time_end, labels=None):
    """
    Computes the average unemployment rate for each occupation over [time_start:time_end),
    but first checks if any occupation has (e+u)=0 in that window and prints them out.

    Parameters
    ----------
    E : ndarray, shape (T, N)
        Employment over time for each of N occupations.
    U : ndarray, shape (T, N)
        Unemployment over time for each of N occupations.
    time_start, time_end : int
        The slice of time over which to average (time_start inclusive, time_end exclusive).
    labels : array-like of str, shape (N,), optional
        If provided, must contain the occupation‐names. If None, only indices get printed.

    Returns
    -------
    rates : ndarray, shape (N,)
        100 * (sum_{t=time_start..time_end-1} U[t, i]) / (sum_{t=time_start..time_end-1} (E[t, i] + U[t, i]))
        for each occupation i.  If denominator is zero, that entry will become np.inf or np.nan
        (after printing a warning).
    """

    # 1) slice out the window
    e = E[time_start:time_end, :]   # shape = (time_end - time_start, N)
    u = U[time_start:time_end, :]   # same shape

    # 2) sum along the time‐axis (axis=0) to get an array of length N
    num = np.sum(u, axis=0)          # ∑_t U[t,i]
    den = np.sum(e + u, axis=0)      # ∑_t (E[t,i] + U[t,i])

    # 3) detect which occupations have den == 0
    zero_mask = (den == 0)           # boolean array of length N
    if np.any(zero_mask):
        culprits = np.nonzero(zero_mask)[0]   # array of indices where den==0
        if labels is not None:
            bad_list = [(i, labels[i]) for i in culprits]
            print("⚠️  Zero‐denominator in period_occ_u_rate for occupations (idx, label):", bad_list)
        else:
            print("⚠️  Zero‐denominator in period_occ_u_rate for occupation indices:", culprits.tolist())

    # 4) do the division (NumPy will emit inf / nan for den==0, but we’ve already logged)
    rates = 100.0 * (num / den)
    return rates
