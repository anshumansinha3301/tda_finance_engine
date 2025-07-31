import numpy as np
import matplotlib.pyplot as plt

from core.market_topology_encoder import MarketTopologyEncoder
from core.alpha_surface_simulator import AlphaSurfaceSimulator
from core.signal_topology_monitor import SignalTopologyMonitor

def load_synthetic_data(length=500):
    """
    Generate synthetic price data with trend + noise for demo purposes.
    """
    np.random.seed(42)
    trend = np.linspace(0, 10, length)
    noise = np.random.normal(0, 1.0, length)
    cycle = 2 * np.sin(np.linspace(0, 10 * np.pi, length))
    return trend + cycle + noise

def main():
    print("🚀 Starting NeuroTopological Alpha Simulation")

    price_data = load_synthetic_data()
    print(f"[+] Loaded time series of length: {len(price_data)}")

    # Step 2: Encode topological features
    encoder = MarketTopologyEncoder()
    topo_features = encoder.encode_series(price_data)
    print(f"[+] Extracted topological features: {topo_features}")

    # Step 3: Simulate alpha surfaces
    simulator = AlphaSurfaceSimulator()
    surfaces = simulator.simulate_and_analyze(n_samples=3)
    for i, metrics in enumerate(surfaces):
        print(f"[+] Surface {i + 1}: {metrics}")

    # Step 4: Monitor topological dynamics over time
    monitor = SignalTopologyMonitor(window_size=100, step=25)
    monitor.process(price_data)
    monitor.plot_metric_over_time(metric="longest_barcode")
    monitor.plot_metric_over_time(metric="avg_barcode_length")

    print("✅ Analysis complete.")

if __name__ == "__main__":
    main()
