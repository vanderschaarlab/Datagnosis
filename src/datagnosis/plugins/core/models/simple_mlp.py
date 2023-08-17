# third party
import torch
from pydantic import validate_call
from torch import nn


class SimpleMLP(nn.Module):
    @validate_call
    def __init__(self, input_dim: int = 4, output_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = nn.Linear(input_dim, 128)
        self.hidden_layer1 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = self.relu(self.input_layer(x))
        out = self.relu(self.hidden_layer1(out))
        out = self.output_layer(out)
        if self.output_dim == 1:
            out = out.squeeze()
        return out
