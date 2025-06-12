# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 13:54:44 2025

@author: mauhl
"""

import numpy as np

def adjust_self_loops(transition_matrix: np.ndarray,
                      employment: np.ndarray,
                      num_lowest: int,
                      new_loop_prob: float) -> np.ndarray:
    """
    For the `num_lowest` occupations by employment, set the self-loop
    probability (diagonal entry) to `new_loop_prob` and re-normalize
    the rest of each row so that the matrix remains row-stochastic.

    Parameters
    ----------
    transition_matrix : (n, n) array
        Row-stochastic job-to-job transition matrix.
    employment : (n,) array
        Employment counts for each occupation.
    num_lowest : int
        Number of lowest-employment occupations to adjust.
    new_loop_prob : float
        Desired self-loop probability (0 <= new_loop_prob <= 1).

    Returns
    -------
    np.ndarray
        A new (n, n) row-stochastic matrix with adjusted self-loops.
    """
    # Sanity checks
    n = transition_matrix.shape[0]
    if transition_matrix.shape != (n, n):
        raise ValueError("transition_matrix must be square")
    if employment.shape[0] != n:
        raise ValueError("employment length must match matrix dimensions")
    if not (0 <= new_loop_prob <= 1):
        raise ValueError("new_loop_prob must be between 0 and 1")
    if num_lowest < 1 or num_lowest > n:
        raise ValueError("num_lowest must be between 1 and n occupations")

    # Work on a copy so we don't overwrite the original
    P = transition_matrix.astype(float).copy()

    # Find indices of the num_lowest occupations
    low_idx = np.argsort(employment)[:num_lowest]

    for i in low_idx:
        old_diagonal = P[i, i]
        # Extract off-diagonal probabilities
        off_diag = P[i, :].copy()
        off_diag[i] = 0.0
        off_sum = off_diag.sum()

        # If there is any off-diagonal mass to redistribute
        if off_sum > 0:
            # Scale off-diagonals so they sum to (1 - new_loop_prob)
            scale = (1.0 - new_loop_prob) / off_sum
            P[i, :] = off_diag * scale
            P[i, i] = new_loop_prob
        else:
            # Edge case: row was originally all self-loop.
            # We'll leave it as a pure self-loop.
            P[i, :] = 0.0
            P[i, i] = 1.0

    return P

import numpy as np

def adjust_delta_u(delta_u: np.ndarray,
                   employment: np.ndarray,
                   num_lowest: int,
                   new_value: float) -> np.ndarray:
    """
    For the `num_lowest` occupations by employment, set their entries in
    `delta_u` to `new_value`.

    Parameters
    ----------
    delta_u : (n,) array
        Original vector of delta_u for each occupation.
    employment : (n,) array
        Employment counts for each occupation.
    num_lowest : int
        Number of lowest-employment occupations whose delta_u will be replaced.
    new_value : float
        The value to assign to delta_u for those occupations.

    Returns
    -------
    np.ndarray
        A new (n,) delta_u array with specified entries replaced.
    """
    # Sanity checks
    n = delta_u.shape[0]
    if employment.shape[0] != n:
        raise ValueError("employment length must match delta_u length")
    if num_lowest < 1 or num_lowest > n:
        raise ValueError("num_lowest must be between 1 and n occupations")

    # Copy so original is untouched
    new_delta = delta_u.astype(float).copy()

    # Find indices of the num_lowest occupations
    low_idx = np.argsort(employment)[:num_lowest]

    # Assign new value
    new_delta[low_idx] = new_value

    return new_delta
