import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Optional
from torchmetrics import Accuracy, AUROC, AveragePrecision, F1Score, Recall


from benchmark.models.dreams_classifier import DreamsClassifier
from benchmark.models.loss import FocalLoss


class LitClassifier(pl.LightningModule):
    """
    Lightning module that supports binary or multiclass with optional focal loss.
    """
    def __init__(
        self,
        ckpt_path: str,
        num_classes: int = 2,
        n_highest_peaks: int = 60,
        lr: float = 1e-4,
        dropout: float = 0.1,
        train_encoder: bool = False,
        use_focal: bool = False,
        gamma: float = 0.5,
        alpha: float = None,
        pos_weight: float = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        # model
        self.model = DreamsClassifier(
            ckpt_path=self.hparams.ckpt_path,
            n_highest_peaks=self.hparams.n_highest_peaks,
            embedding_dim=1024,
            num_classes=self.hparams.num_classes,
            dropout=self.hparams.dropout,
            train_encoder=self.hparams.train_encoder,
        )
        # loss
        if self.hparams.use_focal:
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma,
                alpha=self.hparams.alpha,
                binary=(self.hparams.num_classes == 2),
                return_softmax_out=False,
            )
        else:
            if self.hparams.num_classes == 2:
                weight = torch.tensor(self.hparams.pos_weight) if self.hparams.pos_weight else None
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
            else:
                self.criterion = nn.CrossEntropyLoss()
        # metrics
        if self.hparams.num_classes == 2:
            self.train_acc = Accuracy(task="binary", threshold=0.5)
            self.val_acc   = Accuracy(task="binary", threshold=0.5)
            self.val_auc   = AUROC(task="binary")
            self.val_pr    = AveragePrecision(task="binary")
            self.val_f1    = F1Score(task="binary")
            self.val_recall = Recall(task="binary", threshold=0.5)

            self.test_acc = Accuracy(task="binary", threshold=0.5)
            self.test_auc = AUROC(task="binary")
            self.test_pr = AveragePrecision(task="binary")
            self.test_f1 = F1Score(task="binary")
            self.test_recall = Recall(task="binary", threshold=0.5)
        else:
            self.train_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
            self.val_acc   = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
            self.val_auc   = AUROC(task="multiclass", num_classes=self.hparams.num_classes)
            self.val_pr    = AveragePrecision(task="multiclass", num_classes=self.hparams.num_classes, average="macro")
            self.val_f1    = F1Score(task="multiclass", num_classes=self.hparams.num_classes, average="macro")

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.model(spec)

    def step(self, batch, stage: str):
        x, y = batch['spec'], batch['label']
        logits = self(x)
        # binary logits shape [B], multiclass [B, C]
        # compute loss
        if self.hparams.use_focal:
            if self.hparams.num_classes == 2:
                p = torch.sigmoid(logits)
                loss = self.criterion(p, y)
            else:
                tgt = F.one_hot(y.long(), num_classes=self.hparams.num_classes).float()
                loss = self.criterion(logits, tgt).mean()
        else:
            if self.hparams.num_classes == 2:
                loss = self.criterion(logits, y)
            else:
                loss = self.criterion(logits, y.long())
        # predictions & logging
        if self.hparams.num_classes == 2:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
        else:
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        self.log(f"{stage}_loss", loss, prog_bar=(stage!='train'))
        acc = self.train_acc if stage=='train' else self.val_acc
        self.log(f"{stage}_acc", acc(probs if self.hparams.num_classes>2 else preds, y.long()), prog_bar=True)
        if stage == 'val':
            self.log('val_auc', self.val_auc(probs, y.long()), prog_bar=True)
            self.log('val_pr',  self.val_pr(probs, y.long()), prog_bar=True)
            self.log('val_f1',  self.val_f1(probs, y.long()), prog_bar=True)
            self.log('val_recall', self.val_recall(preds, y.long()), prog_bar=True)
        if stage == 'test':
            self.log('test_auc', self.test_auc(probs, y.long()), prog_bar=True)
            self.log('test_pr', self.test_pr(probs, y.long()), prog_bar=True)
            self.log('test_f1', self.test_f1(probs, y.long()), prog_bar=True)
            self.log('test_recall', self.test_recall(preds, y.long()), prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        # scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=self.hparams.lr,
        #     total_steps=self.trainer.estimated_stepping_batches,
        #     pct_start=0.1,
        #     div_factor=10.0,
        #     final_div_factor=100.0,
        # )
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }



# class LitBinaryClassifier(pl.LightningModule):
#     """
#     LightningModule for fine-tuning any binary classifier on spectra tasks,
#     with metrics suited for imbalanced data (balanced accuracy via macro accuracy).
#     """
#     def __init__(
#         self,
#         ckpt_path: str,
#         n_highest_peaks: int = 60,
#         lr: float = 1e-4,
#         dropout: float = 0.1,
#         train_encoder: bool = False,
#         pos_weight: Optional[float] = None,
#     ):
#         super().__init__()
#         # Save all init args to self.hparams
#         self.save_hyperparameters()
#
#         # DreaMS-based classifier head
#         self.model = DreamsClassifier(
#             ckpt_path=self.hparams.ckpt_path,
#             n_highest_peaks=self.hparams.n_highest_peaks,
#             dropout=self.hparams.dropout,
#             train_encoder=self.hparams.train_encoder,
#         )
#
#         # Balanced accuracy: macro-average of the binary task
#         self.train_bal_acc = Accuracy(task="binary", threshold=0.5, average="macro")
#         self.val_bal_acc   = Accuracy(task="binary", threshold=0.5, average="macro")
#
#         # Additional validation metrics
#         self.val_auc = AUROC(task="binary")
#         self.val_pr  = AveragePrecision(task="binary")
#         self.val_f1  = F1Score(task="binary")
#
#         # Positive class weight for BCE loss
#         self.pos_weight = self.hparams.pos_weight
#
#     def forward(self, spec: torch.Tensor) -> torch.Tensor:
#         return self.model(spec)
#
#     def _shared_step(self, batch, stage: str):
#         x, y = batch['spec'], batch['label']
#         logits = self(x)
#         # Compute loss
#         if self.pos_weight is not None:
#             pos_w = torch.tensor(self.pos_weight, device=self.device)
#             loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w)
#         else:
#             loss = F.binary_cross_entropy_with_logits(logits, y)
#
#         preds = torch.sigmoid(logits)
#         # Log loss
#         self.log(f'{stage}_loss', loss, prog_bar=(stage!='train'))
#         # Log balanced accuracy
#         bal_acc = self.train_bal_acc if stage=='train' else self.val_bal_acc
#         self.log(f'{stage}_bal_acc', bal_acc(preds, y.int()), prog_bar=True)
#
#         # Validation-only metrics
#         if stage == 'val':
#             self.log('val_auc', self.val_auc(preds, y.int()),       prog_bar=True)
#             self.log('val_pr',  self.val_pr(preds, y.int()),        prog_bar=True)
#             self.log('val_f1',  self.val_f1(preds, y.int()),        prog_bar=True)
#         return loss
#
#     def training_step(self, batch, batch_idx):
#         return self._shared_step(batch, 'train')
#
#     def validation_step(self, batch, batch_idx):
#         return self._shared_step(batch, 'val')
#
#     def test_step(self, batch, batch_idx):
#         # Reuse validation logic on test
#         return self._shared_step(batch, 'val')
#
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
#
#     def configure_optimizers(self):
#         # 1) AdamW with weight decay
#         optimizer = optim.AdamW(
#             self.parameters(),
#             lr=self.hparams.lr,
#             weight_decay=1e-5
#         )
#
#         # 2) One-cycle LR scheduler
#         scheduler = OneCycleLR(
#             optimizer,
#             max_lr=self.hparams.lr,
#             total_steps=self.trainer.estimated_stepping_batches,
#             pct_start=0.1,
#             div_factor=10.0,
#             final_div_factor=100.0,
#         )
#
#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': {
#                 'scheduler': scheduler,
#                 'interval': 'step',
#             }
#         }
