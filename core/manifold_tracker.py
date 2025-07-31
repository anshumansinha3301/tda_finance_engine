import numpy as np
from scipy.spatial.distance import pdist, squareform

def compute_riemannian_embedding(data, sigma=1.0):
    """
    Computes a Riemannian embedding using Gaussian kernels over the price manifold.
    """
    dist_matrix = squareform(pdist(data))
    kernel = np.exp(-dist_matrix**2 / (2 * sigma**2))
    return kernel / np.sum(kernel, axis=1, keepdims=True)
