# third party
import torch
import torch.nn as nn

from pydantic import validate_arguments


# This is a PyTorch implementation of a LSTM model with configurable number of layers, hidden
# dimensions, and dropout, used for classification tasks.
class LSTM(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_layer: int = 2,
        n_class: int = 2,
        dropout: float = 0.0,
    ):
        super(LSTM, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, n_class)

        # initialisation
        self.lstm = nn.LSTM(
            in_dim,
            hidden_dim,
            n_layer,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=False,
        )
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                param.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out
