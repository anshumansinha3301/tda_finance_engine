# 🧠 NeuroTopological Alpha Surfaces (NTAS)

> A hybrid framework combining **topological data analysis**, **manifold learning**, and **neuro-symbolic modeling** to infer latent alpha-generating structures from financial time series.

---

## 📌 Overview

**NTAS** is an experimental research framework that hypothesizes the presence of **latent geometric structures ("alpha surfaces")** in financial markets — structures that evolve in hidden topological phase spaces and may encode long-term predictive signals.

This project aims to extract those latent shapes using tools from:
- 🌀 **Topological Data Analysis (TDA)** — persistent homology and barcode landscapes
- 📐 **Differential Geometry** — phase space embeddings, curvature, geodesics
- 🧠 **Neuro-symbolic AI** — LSTMs fused with topological attention
- 🔁 **Self-supervised inference loops** — no labels, just structured temporal evolution

The architecture is modular, research-driven, and easily extensible for both deep learning and quantitative finance tasks.

---

## 🏗️ Architecture

```
neurotopo-alpha/
│
├── core/                   # Core mathematics, TDA, curvature, simulation
│   ├── alpha_surface_simulator.py
│   ├── market_topology_encoder.py
│   ├── persistent_homology.py
│   └── signal_topology_monitor.py
│
├── models/                # Neural and neuro-symbolic model architectures
│   ├── topo_lstm.py
│   └── symbnet.py
│
├── data/                  # Preprocessing, transforms, and loaders
│   ├── pipeline.py
│   └── transforms.py
│
├── experiments/           # Notebooks for analysis, research, visualization
│   ├── latent_phase_experiment.ipynb
│   └── alpha_geometry.ipynb
│
├── config/                # Configs and hyperparameters
│   └── hyperparams.yaml
│
├── main.py                # Main execution script
├── README.md              # This file
├── setup.py               # Packaging and dependencies
└── LICENSE                # MIT License
```

---

## 🔧 Requirements

Install Python packages:
```bash
pip install -r requirements.txt
```

Core dependencies:
- `numpy`, `scipy`, `matplotlib`
- `torch` — neural networks
- `gudhi` — persistent homology
- `pywt` — wavelet transforms
- `sklearn` — manifold projection

---

## 🚀 Usage

Run the full pipeline:
```bash
python main.py
```

What it does:
- Generates synthetic market data
- Computes topological summaries of rolling windows
- Simulates synthetic alpha surfaces using curvature and geodesics
- Visualizes evolution of persistent structures over time

Explore results:
- Notebooks in `/experiments` show diagrams and surface plots
- TDA is applied via delay embeddings and Rips filtrations

---

## 🧠 Conceptual Foundation

### 🧩 1. **Alpha Surfaces**
Latent multi-dimensional manifolds that evolve from the geometry of market price data. These are not directly observable, but can be inferred via:
- Curvature analysis
- Phase space embeddings
- Persistent homology features

### 🔗 2. **Persistent Homology**
Persistent barcodes track the "lifespan" of topological features across scale. In finance, they may represent:
- Cycles of volatility
- Regime shifts
- Fractal structures in price action

### 🔭 3. **Manifold + TDA Fusion**
We delay-embed price series into geometric clouds, extract persistent features, and encode those into:
- Summary statistics
- Barcode landscapes
- Attention weights for neural models

### 🤖 4. **Topo-Aware LSTM**
A custom LSTM that injects persistent homology features into its hidden state using a learned gate. This allows time-series models to consider **structural complexity**, not just magnitude and momentum.

---

## 🔬 Applications

| Domain               | Use Case                                                                 |
|----------------------|--------------------------------------------------------------------------|
| Quantitative Finance | Alpha factor modeling, anomaly detection, volatility regime analysis     |
| Time Series Research | Nonlinear manifold tracking, structure-aware forecasting                 |
| Signal Processing    | Geometric noise compression, chaos detection (via Lyapunov exponents)    |
| Machine Learning     | Feature engineering, unsupervised representation of time series geometry |

---

## 👨‍🔬 Author

**Anshuman Sinha**  
Data Scientist at Xyphor Advisors  
Exploring the convergence of mathematical structure, neural computation, and market evolution.  
GitHub: [anshumansinha3301](https://github.com/anshumansinha3301)

---

## 📖 Citations & Theory

- Carlsson, G. (2009). *Topology and data.* Bulletin of the AMS  
- Gidea & Katz (2018). *Topological data analysis of financial time series.*  
- Vidal et al. (2020). *Neuro-symbolic artificial intelligence: The state of the art.*  
- Peters et al. (2022). *Riemannian manifold learning for trading regimes.*

---

## 🪪 License

MIT License — free to use, fork, and contribute with attribution.

---

## 🏁 Final Notes

NTAS is not a production trading system.  
It is a research-grade framework for exploring the **hidden structure of market behavior**.  
Use it to push the boundary of how we model reality in high-dimensional chaotic systems.

> "What we see is not all that exists."  
> — The Topologist's Creed
