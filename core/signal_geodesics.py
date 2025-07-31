import numpy as np

def compute_signal_geodesic(path):
    """
    Computes the arc length of a signal path (geodesic length).
    """
    diffs = np.diff(path)
    return np.sum(np.sqrt(1 + diffs**2))
