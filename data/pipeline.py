import numpy as np
from scipy.signal import savgol_filter

def preprocess_market_data(data):
    """
    Normalizes and smooths raw price series.
    """
    smoothed = savgol_filter(data, 11, 3)
    return (smoothed - np.mean(smoothed)) / np.std(smoothed)
