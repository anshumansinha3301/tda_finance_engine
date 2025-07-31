import numpy as np
from sklearn.manifold import TSNE

def build_phase_space_embedding(time_series, window=10):
    """
    Creates delay-embedded phase space from price series.
    """
    emb = np.array([time_series[i:i + window] for i in range(len(time_series) - window)])
    return emb

def project_phase_space(embedding):
    """
    Reduces dimensionality using t-SNE or UMAP.
    """
    return TSNE(n_components=2, perplexity=30).fit_transform(embedding)
