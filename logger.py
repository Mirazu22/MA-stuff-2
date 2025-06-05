# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 11:46:17 2025

@author: mauhl
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)          # configure once, at top of file
logger = logging.getLogger(__name__)

def log_extinct_occ(E, U, t0, t1, labels=None):
    """
    Print / log the occupations whose employment+unemployment
    sum to zero in the interval [t0, t1).

    Parameters
    ----------
    E, U : ndarray (t_sim, n_occ)
        Employment and unemployment paths.
    t0, t1 : int
        Start (inclusive) and end (exclusive) of the window.
    labels : 1-D iterable of str, optional
        Occupation names; if None, only indices are printed.

    Returns
    -------
    idx : 1-D ndarray
        Indices of the extinct occupations.
    """
    # Vectorised one-liner: sum over the window and test for zero
    den = (E[t0:t1] + U[t0:t1]).sum(axis=0)
    idx = np.flatnonzero(den == 0)

    if idx.size:
        if labels is None:
            logger.warning("Extinct occupations: %s", idx.tolist())
        else:
            culprits = [(i, labels[i]) for i in idx]
            logger.warning("Extinct occupations (idx, label): %s", culprits)
    return idx

# inside save_percentage_change(), just before you call lbn.percentage_change_u
dead = log_extinct_occ(E, U,
                       t_steady_start, t_steady_end,
                       labels=df_labs["label"].values)
