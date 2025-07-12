import torch
from torch import nn
import torch.nn.functional as F
from dreams.api import PreTrainedModel
from dreams.models.dreams.dreams import DreaMS as DreaMSModel


class DreamsClassifier(nn.Module):
    """
    Wraps DreaMS encoder and adds a classification head for N-way outputs.
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
        # load DreaMS
        self.spec_encoder = PreTrainedModel.from_ckpt(
            ckpt_path=ckpt_path,
            ckpt_cls=DreaMSModel,
            n_highest_peaks=n_highest_peaks,
        ).model
        # freeze encoder if needed
        for p in self.spec_encoder.parameters():
            p.requires_grad = train_encoder
        # classification head
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        out_dim = 1 if num_classes == 2 else num_classes
        self.lin_out = nn.Linear(embedding_dim, out_dim)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: [B, peaks, 2]
        x = self.spec_encoder(spec)[:, 0, :]  # [B, emb]
        x = self.dropout(x)
        out = self.lin_out(x)
        # collapse last dim for binary
        return out.squeeze(-1)


# class DreamsClassifier(nn.Module):
#     """
#     Wraps a frozen or trainable DreaMS encoder + binary classification head.
#     """
#
#     def __init__(
#         self,
#         ckpt_path: str,
#         n_highest_peaks: int = 60,
#         embedding_dim: int = 1024,
#         dropout: float = 0.1,
#         train_encoder: bool = False,
#     ):
#         """
#         Args:
#             ckpt_path: path to the pre-trained SSL checkpoint (e.g. 'ssl_model.ckpt' or 'embedding_model.ckpt')
#             n_highest_peaks: how many peaks DreaMS should ingest
#             embedding_dim: dimension of the output token embeddings (default 1024)
#             dropout: dropout before the final head
#             train_encoder: whether to fine-tune the DreaMS encoder or freeze it
#         """
#         super().__init__()
#         # load the pre-trained DreaMS model
#         self.spec_encoder = PreTrainedModel.from_ckpt(
#             ckpt_path=ckpt_path,
#             ckpt_cls=DreaMSModel,
#             n_highest_peaks=n_highest_peaks,
#         ).model
#
#         # freeze or unfreeze the encoder
#         for p in self.spec_encoder.parameters():
#             p.requires_grad = train_encoder
#
#         # classification head
#         self.dropout = nn.Dropout(dropout)
#         self.lin_out = nn.Linear(embedding_dim, 1)
#
#     def forward(self, spec: torch.Tensor) -> torch.Tensor:
#         """
#         spec: [batch_size, n_peaks, 2]  (mz, intensity)
#         returns: [batch_size]  logits for binary classification
#         """
#         # DreaMS returns [batch, seq_len, dim]; take the [CLS]-like first token
#         x = self.spec_encoder(spec)[:, 0, :]    # [B, 1024]
#         x = self.dropout(x)
#         return self.lin_out(x).squeeze(-1)      # [B]