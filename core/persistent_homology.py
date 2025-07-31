import numpy as np
from gudhi import RipsComplex

def compute_persistence_diagram(time_series, max_dim=2, epsilon=0.5):
    """
    Converts a 1D time series into a point cloud and computes persistent homology.
    """
    pts = np.array([[i, val] for i, val in enumerate(time_series)])
    rips = RipsComplex(points=pts, max_edge_length=epsilon)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    return st.persistence()
