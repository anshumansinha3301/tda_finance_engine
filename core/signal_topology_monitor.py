import numpy as np
import matplotlib.pyplot as plt
from core.alpha_phase_space import build_phase_space_embedding, project_phase_space
from core.persistent_homology import compute_persistence_diagram
from core.barcode_statistics import longest_barcode, average_barcode_length

class SignalTopologyMonitor:
    """
    Monitors topology of a dynamic financial signal in real-time fashion.
    """

    def __init__(self, window_size=100, step=10):
        self.window_size = window_size
        self.step = step
        self.history = []

    def rolling_windows(self, signal):
        """
        Splits signal into rolling windows of fixed size.
        """
        return [signal[i:i+self.window_size]
                for i in range(0, len(signal) - self.window_size, self.step)]

    def compute_window_features(self, window):
        """
        For a single window, extract persistent diagram + summary stats.
        """
        emb = build_phase_space_embedding(window)
        diag = compute_persistence_diagram(emb[:, 0], max_dim=1, epsilon=0.6)
        return {
            "longest_barcode": longest_barcode(diag),
            "avg_barcode_length": average_barcode_length(diag),
            "barcode_count": len(diag)
        }

    def process(self, signal):
        """
        Process all rolling windows and record topological summaries.
        """
        windows = self.rolling_windows(signal)
        self.history = [self.compute_window_features(w) for w in windows]
        return self.history

    def plot_metric_over_time(self, metric="longest_barcode"):
        """
        Visualizes evolution of a topological metric over time.
        """
        values = [h[metric] for h in self.history]
        plt.plot(values, label=metric)
        plt.title(f"Evolution of {metric}")
        plt.xlabel("Window Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()
