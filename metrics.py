from __future__ import division
import numpy as np


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def computeViolationRatio(patt_gph, patt_seq):
    return 1 - sum(map(patt_gph.hasPattern, patt_seq)) / len(patt_seq)


def computeLikelihood(patt_gph, patt_seq, penalty=1):
    log_lklhood = map(np.log,
                      filter(lambda x: x != 0.0,
                             map(patt_gph.getPatternProbability, patt_seq)))
    n_misses = len(patt_seq) - len(log_lklhood)

    if n_misses == len(patt_seq):
        log_lklhood = np.log(1.0 / patt_gph.n_trans) * len(patt_seq)
        # log_lklhood = np.log(1e-64)*len(patt_seq)
    else:
        log_lklhood = np.sum(log_lklhood)

    return np.sum((log_lklhood,
                  np.log((len(patt_seq) - n_misses) / len(patt_seq))))

def computeBhattCoeff(x, y, bins=20):
    """ Compute the Bhattacharyya distance between samples of two random variables.

    Parameters
    ----------
    x, y : np.ndarray
        Samples of two random variables.
    bins : int
        Number of bins to use when approximating densities.

    Returns
    -------
    bhatt_coeff : float
        Bhattacharyya coefficient.
    """

    # Find histogram range - min to max
    bounds = [min(min(x), min(y)), max(max(x), max(y))]

    # Compute histograms
    x_hist = np.histogram(x, bins=bins, range=bounds)[0]
    y_hist = np.histogram(y, bins=bins, range=bounds)[0]

    # Normalize
    x_hist = x_hist.astype(float)/x_hist.sum()
    y_hist = y_hist.astype(float)/y_hist.sum()

    # Compute Bhattacharyya coefficient
    return np.sum(np.sqrt(x_hist*y_hist))