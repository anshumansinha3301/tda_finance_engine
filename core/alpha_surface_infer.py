import torch
from models.topo_lstm import TopoAwareLSTM

class AlphaSurfaceInferer:
    def __init__(self, input_dim, hidden_dim):
        self.model = TopoAwareLSTM(input_dim, hidden_dim)

    def infer_surface(self, time_series, topological_features):
        """
        Feeds processed time series and topological features into the model.
        """
        x = torch.tensor(time_series, dtype=torch.float32).unsqueeze(0)
        topo = torch.tensor(topological_features, dtype=torch.float32)
        return self.model(x, topo)
