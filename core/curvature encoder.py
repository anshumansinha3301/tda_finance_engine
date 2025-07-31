import numpy as np

def ricci_curvature(signal):
    """
    Applies a discrete curvature approximation to signal windows.
    """
    curvature = []
    for i in range(1, len(signal) - 1):
        left = signal[i - 1]
        center = signal[i]
        right = signal[i + 1]
        k = (left - 2 * center + right)
        curvature.append(k)
    return np.array(curvature)
