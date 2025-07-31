import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

class FocalLoss(nn.Module):
    """
    Focal loss for binary and multi-class classification,
    implemented AMP-safe by using binary_cross_entropy_with_logits.
    """
    def __init__(
        self,
        gamma: float,
        alpha=None,
        binary: bool = False,
        return_softmax_out: bool = False,
    ):
        super().__init__()
        self.binary = binary
        self.gamma = gamma
        self.alpha = alpha
        self.return_softmax_out = return_softmax_out
        if alpha is not None:
            if binary:
                assert 0 < alpha < 1, "For binary, alpha must be scalar in (0,1)"
            else:
                raise NotImplementedError("Alpha weighting only for binary currently.")

    def forward(self, inputs, targets):
        if not self.binary:
            # multi-class: inputs are logits, targets one-hot
            p = F.softmax(inputs, dim=-1)
            loss = F.nll_loss(p.log(), torch.argmax(targets, dim=-1), reduction='none')
            if self.gamma != 0:
                p_t = (targets * p).sum(dim=-1)
                loss = ((1 - p_t) ** self.gamma) * loss
            return (loss, p) if self.return_softmax_out else loss
        else:
            # binary: inputs are logits, targets floats 0/1
            # ensure shapes
            logits = inputs.squeeze(-1)
            targets = targets.squeeze(-1).float()

            # compute probabilities for weighting
            p = torch.sigmoid(logits)

            # base alpha weight
            weight = torch.ones_like(targets)
            if self.alpha is not None:
                weight = torch.where(targets > 0.5, self.alpha, 1 - self.alpha)

            # focal modulation
            if self.gamma > 0:
                p_t = torch.where(targets > 0.5, p, 1 - p)
                weight = weight * (1 - p_t) ** self.gamma

            # use logits + BCE with logits for AMP safety
            loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                weight=weight.detach(),
                reduction='mean'
            )
            return loss



# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.cuda.amp import autocast
#
#
# class FocalLoss(nn.Module):
#     """
#     Focal loss for binary and multi-class classification.
#     """
#     def __init__(self, gamma: float, alpha=None, binary: bool = False, return_softmax_out: bool = False):
#         super().__init__()
#         self.binary = binary
#         self.gamma = gamma
#         self.alpha = alpha
#         self.return_softmax_out = return_softmax_out
#         if alpha is not None:
#             if binary:
#                 assert 0 < alpha < 1, "For binary, alpha must be scalar in (0,1)"
#             else:
#                 raise NotImplementedError("Alpha weighting only for binary currently.")
#
#     def forward(self, inputs, targets):
#         if not self.binary:
#             # multi-class: inputs are logits, targets one-hot
#             p = F.softmax(inputs, dim=-1)
#             loss = F.nll_loss(p.log(), torch.argmax(targets, dim=-1), reduction='none')
#             if self.gamma != 0:
#                 p_t = (targets * p).sum(dim=-1)
#                 loss = ((1 - p_t) ** self.gamma) * loss
#             return (loss, p) if self.return_softmax_out else loss
#         else:
#             # binary: inputs are probabilities (0..1), targets floats 0/1
#             # ensure shapes match
#             if inputs.ndim > targets.ndim:
#                 inputs = inputs.squeeze(-1)
#             if targets.ndim < inputs.ndim:
#                 targets = targets.unsqueeze(-1).squeeze(-1)
#
#             weight = torch.ones_like(targets)
#             if self.alpha is not None:
#                 mask = targets > 0.5
#                 weight[mask] = self.alpha
#                 weight[~mask] = 1 - self.alpha
#             if self.gamma > 0:
#                 weight = weight * torch.where(
#                     targets > 0.5,
#                     (1 - inputs) ** self.gamma,
#                     inputs ** self.gamma
#                 )
#             # binary_cross_entropy expects inputs and targets same shape
#             inputs = inputs.squeeze(-1)
#             targets = targets.squeeze(-1)
#             # disable autocast for binary_cross_entropy under AMP
#             with autocast(enabled=False):
#                 bce = F.binary_cross_entropy(
#                     inputs,
#                     targets,
#                     weight=weight.detach(),
#                     reduction='mean'
#                 )
#             return bce
