import numpy as np
from scipy.ndimage import gaussian_filter1d
from core.signal_geodesics import compute_signal_geodesic
from core.topo_entropy import compute_topological_entropy

class AlphaSurfaceSimulator:
    """
    Simulates hypothetical 'alpha surfaces' from synthetic market environments.
    """

    def __init__(self, smoothing=1.5, noise_level=0.2):
        self.smoothing = smoothing
        self.noise_level = noise_level

    def generate_base_signal(self, length=500, freq=0.1):
        """
        Creates a sine-based synthetic signal with noise and trend.
        """
        t = np.linspace(0, 2 * np.pi * freq * length, length)
        base = np.sin(t) + 0.05 * t + np.random.normal(0, self.noise_level, length)
        return gaussian_filter1d(base, sigma=self.smoothing)

    def compute_surface_metrics(self, surface):
        """
        Derives curvature and geodesic complexity metrics from the surface.
        """
        geo_len = compute_signal_geodesic(surface)
        curvature = np.gradient(np.gradient(surface))
        curvature_std = np.std(curvature)
        return {
            "geodesic_length": geo_len,
            "curvature_std": curvature_std
        }

    def simulate_and_analyze(self, n_samples=10):
        """
        Generates multiple alpha surfaces and evaluates each.
        """
        results = []
        for _ in range(n_samples):
            surf = self.generate_base_signal()
            metrics = self.compute_surface_metrics(surf)
            results.append(metrics)
        return results
