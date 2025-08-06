# import torch
# from torch import nn
# from dreams.api import PreTrainedModel
# from dreams.models.dreams.dreams import DreaMS as DreaMSModel
#
# class DreamsClassifier(nn.Module):
#     """
#     Wraps DreaMS encoder and adds a classification head
#     """
#     def __init__(
#         self,
#         ckpt_path: str,
#         n_highest_peaks: int = 60,
#         embedding_dim: int = 1024,
#         num_classes: int = 2,
#         dropout: float = 0.1,
#         train_encoder: bool = False,
#     ):
#         super().__init__()
#         # load DreaMS encoder
#         self.spec_encoder = PreTrainedModel.from_ckpt(
#             ckpt_path=ckpt_path,
#             ckpt_cls=DreaMSModel,
#             n_highest_peaks=n_highest_peaks,
#         ).model
#         # freeze encoder if desired
#         for p in self.spec_encoder.parameters():
#             p.requires_grad = train_encoder
#
#         out_dim = 1 if num_classes == 2 else num_classes
#         hidden_dims = [512, 512, 512, 256]
#         layers = []
#         prev_dim = embedding_dim
#         for hdim in hidden_dims:
#             layers += [
#                 nn.Linear(prev_dim, hdim),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#             ]
#             prev_dim = hdim
#         # final projection
#         layers.append(nn.Linear(prev_dim, out_dim))
#
#         self.classifier = nn.Sequential(*layers)
#         self.num_classes = num_classes
#
#     def forward(self, spec: torch.Tensor) -> torch.Tensor:
#         # spec: [B, peaks, 2]
#         x = self.spec_encoder(spec)[:, 0, :]  # [B, embedding_dim]
#         logits = self.classifier(x)
#         return logits.squeeze(-1)  # collapse to [B] for binary


import torch
from torch import nn
from dreams.api import PreTrainedModel
from dreams.models.dreams.dreams import DreaMS as DreaMSModel


class ResidualBlock(nn.Module):
    """
    A simple residual MLP block: Linear -> BN -> ReLU -> Dropout -> Linear -> BN + skip connection
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.bn2(out)
        return self.act(out + residual)


class DreamsClassifier(nn.Module):
    """
    Wraps DreaMS encoder and adds a more powerful classification head
    with batch norm and residual connections.
    """
    def __init__(
        self,
        ckpt_path: str,
        n_highest_peaks: int = 60,
        embedding_dim: int = 1024,
        num_classes: int = 2,
        dropout: float = 0.1,
        train_encoder: bool = False,
    ):
        super().__init__()
        # 1) load pre-trained DreaMS encoder
        self.spec_encoder = PreTrainedModel.from_ckpt(
            ckpt_path=ckpt_path,
            ckpt_cls=DreaMSModel,
            n_highest_peaks=n_highest_peaks,
        ).model
        # optionally freeze encoder
        for p in self.spec_encoder.parameters():
            p.requires_grad = train_encoder

        # 2) build classification head
        hidden_dims = [1024, 1024, 1024, 256]
        self.layers = nn.ModuleList()
        prev_dim = embedding_dim
        for hdim in hidden_dims:
            if hdim == prev_dim:
                # use a residual block when dims match
                self.layers.append(ResidualBlock(hdim, dropout))
            else:
                # simple linear block when dims change
                self.layers.append(nn.Sequential(
                    nn.Linear(prev_dim, hdim),
                    nn.BatchNorm1d(hdim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ))
            prev_dim = hdim
        # final output layer
        out_dim = 1 if num_classes == 2 else num_classes
        self.head = nn.Linear(prev_dim, out_dim)
        self.num_classes = num_classes

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec shape: [B, peaks, 2]
        x = self.spec_encoder(spec)[:, 0, :]  # [B, embedding_dim]
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x)
        # squeeze for binary classification
        return logits.squeeze(-1)  # [B] if binary, [B, C] otherwise

