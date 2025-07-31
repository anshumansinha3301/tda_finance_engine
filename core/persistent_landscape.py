import numpy as np

def landscape_from_barcode(diagram, resolution=100):
    """
    Builds a persistent landscape: piecewise functions from diagram.
    """
    landscape = np.zeros(resolution)
    for _, (b, d) in diagram:
        if d == float("inf"):
            continue
        for i in range(resolution):
            x = i / resolution
            height = max(0, min(x - b, d - x))
            landscape[i] += height
    return landscape
