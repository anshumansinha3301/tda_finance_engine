import numpy as np
import pywt

def wavelet_transform(data, wavelet='db1'):
    """
    Applies wavelet decomposition to a signal.
    """
    coeffs = pywt.wavedec(data, wavelet)
    return np.concatenate(coeffs)
