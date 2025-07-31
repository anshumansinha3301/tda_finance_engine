from gudhi import SimplexTree

def filter_persistent_pairs(diagram, threshold=0.1):
    """
    Filters persistence diagram points above a certain lifetime.
    """
    return [pt for pt in diagram if abs(pt[1][1] - pt[1][0]) > threshold]

def get_homology_dimension_counts(diagram):
    """
    Counts how many features exist per homology dimension (H0, H1, H2).
    """
    from collections import Counter
    dims = [dim for dim, _ in diagram]
    return dict(Counter(dims))
