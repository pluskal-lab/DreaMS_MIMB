import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from typing import Optional
from torchmetrics import Accuracy, AUROC, AveragePrecision, F1Score


from benchmark.models.dreams_classifier import DreamsClassifier


class LitBinaryClassifier(pl.LightningModule):
    """
    LightningModule for fine-tuning any binary classifier on spectra tasks,
    with metrics suited for imbalanced data (balanced accuracy via macro accuracy).
    """
    def __init__(
        self,
        ckpt_path: str,
        n_highest_peaks: int = 60,
        lr: float = 1e-4,
        dropout: float = 0.1,
        train_encoder: bool = False,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        # Save all init args to self.hparams
        self.save_hyperparameters()

        # DreaMS-based classifier head
        self.model = DreamsClassifier(
            ckpt_path=self.hparams.ckpt_path,
            n_highest_peaks=self.hparams.n_highest_peaks,
            dropout=self.hparams.dropout,
            train_encoder=self.hparams.train_encoder,
        )

        # Balanced accuracy: macro-average of the binary task
        self.train_bal_acc = Accuracy(task="binary", threshold=0.5, average="macro")
        self.val_bal_acc   = Accuracy(task="binary", threshold=0.5, average="macro")

        # Additional validation metrics
        self.val_auc = AUROC(task="binary")
        self.val_pr  = AveragePrecision(task="binary")
        self.val_f1  = F1Score(task="binary")

        # Positive class weight for BCE loss
        self.pos_weight = self.hparams.pos_weight

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.model(spec)

    def _shared_step(self, batch, stage: str):
        x, y = batch['spec'], batch['label']
        logits = self(x)
        # Compute loss
        if self.pos_weight is not None:
            pos_w = torch.tensor(self.pos_weight, device=self.device)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y)

        preds = torch.sigmoid(logits)
        # Log loss
        self.log(f'{stage}_loss', loss, prog_bar=(stage!='train'))
        # Log balanced accuracy
        bal_acc = self.train_bal_acc if stage=='train' else self.val_bal_acc
        self.log(f'{stage}_bal_acc', bal_acc(preds, y.int()), prog_bar=True)

        # Validation-only metrics
        if stage == 'val':
            self.log('val_auc', self.val_auc(preds, y.int()),       prog_bar=True)
            self.log('val_pr',  self.val_pr(preds, y.int()),        prog_bar=True)
            self.log('val_f1',  self.val_f1(preds, y.int()),        prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        # Reuse validation logic on test
        return self._shared_step(batch, 'val')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def configure_optimizers(self):
        # 1) AdamW with weight decay
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-5
        )

        # 2) One-cycle LR scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
