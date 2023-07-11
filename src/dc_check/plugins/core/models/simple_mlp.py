import torch
from torch import nn

from pydantic import validate_arguments


class SimpleMLP(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, input_dim: int = 4, output_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = nn.Linear(input_dim, 128)
        self.hidden_layer1 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.input_layer(x))
        out = self.relu(self.hidden_layer1(out))
        out = self.output_layer(out)
        if self.output_dim == 1:
            out = out.squeeze()
        return out
