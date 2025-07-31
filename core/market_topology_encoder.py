import numpy as np
from sklearn.decomposition import PCA
from core.persistent_homology import compute_persistence_diagram
from core.barcode_statistics import average_barcode_length, longest_barcode

class MarketTopologyEncoder:
    """
    Converts raw price series into topological and geometric feature vectors.
    """

    def __init__(self, tda_dim=2, max_points=300):
        self.tda_dim = tda_dim
        self.max_points = max_points

    def preprocess(self, price_series):
        """
        Standardizes input and truncates to max_points.
        """
        series = np.array(price_series)
        series = (series - np.mean(series)) / (np.std(series) + 1e-8)
        return series[-self.max_points:]

    def delay_embedding(self, series, dimension=3, delay=2):
        """
        Converts time series into delay-embedded coordinates.
        """
        N = len(series) - (dimension - 1) * delay
        embedded = np.zeros((N, dimension))
        for i in range(dimension):
            embedded[:, i] = series[i * delay:i * delay + N]
        return embedded

    def extract_topo_features(self, embedded):
        """
        Runs persistent homology and computes statistics.
        """
        diagram = compute_persistence_diagram(embedded[:, 0], max_dim=self.tda_dim)
        return {
            "avg_barcode_length": average_barcode_length(diagram),
            "max_barcode": longest_barcode(diagram),
            "feature_count": len(diagram)
        }

    def encode_series(self, price_series):
        """
        Full pipeline: preprocess, embed, run TDA, return features.
        """
        series = self.preprocess(price_series)
        embedded = self.delay_embedding(series)
        features = self.extract_topo_features(embedded)
        return features
