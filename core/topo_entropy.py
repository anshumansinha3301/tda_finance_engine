import numpy as np

def compute_topological_entropy(persistence_diagram):
    """
    Approximates entropy over lifetime of persistent features.
    """
    lifetimes = [pt[1][1] - pt[1][0] for pt in persistence_diagram if pt[1][1] < float("inf")]
    probs = np.array(lifetimes) / np.sum(lifetimes)
    return -np.sum(probs * np.log(probs + 1e-10))
