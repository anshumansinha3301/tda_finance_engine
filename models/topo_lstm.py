import torch
import torch.nn as nn

class TopoAwareLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.topo_gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, persistence_input):
        """
        LSTM with injected persistent homology as bias signal.
        """
        lstm_out, _ = self.lstm(x)
        topo_bias = torch.tanh(self.topo_gate(persistence_input))
        return lstm_out + topo_bias.unsqueeze(1)
