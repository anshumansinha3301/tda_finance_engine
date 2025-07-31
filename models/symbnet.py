import torch
import torch.nn as nn

class SymbolicPatternNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        """
        Pattern net inspired by symbolic representations.
        """
        x = self.act(self.fc1(x))
        return self.fc2(x)
