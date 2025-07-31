def average_barcode_length(diagram):
    """
    Computes the mean lifetime of persistent features.
    """
    lengths = [b - a for _, (a, b) in diagram if b < float("inf")]
    return sum(lengths) / len(lengths) if lengths else 0

def longest_barcode(diagram):
    """
    Returns the maximum persistent feature lifetime.
    """
    return max([b - a for _, (a, b) in diagram if b < float("inf")], default=0)
