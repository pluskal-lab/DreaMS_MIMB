import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Simple MLP for binary classification.
    """

    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout: float = 0.5):
        super().__init__()
        layers = []
        dims = [input_dim, *hidden_dims]
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], 1))  # single logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)