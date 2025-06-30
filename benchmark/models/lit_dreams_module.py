import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from benchmark.models.dreams_classifier import DreamsClassifier


class LitDreamsClassifier(pl.LightningModule):
    """
    LightningModule for fine-tuning DreamsClassifier on binary tasks.
    """

    def __init__(
        self,
        ckpt_path: str,
        n_highest_peaks: int = 60,
        lr: float = 1e-4,
        dropout: float = 0.1,
        train_encoder: bool = False,
    ):
        """
        Args:
            ckpt_path: path to the DreaMS checkpoint
            n_highest_peaks: number of peaks
            lr: learning rate
            dropout: dropout before head
            train_encoder: whether to fine-tune the encoder
        """
        super().__init__()
        self.save_hyperparameters(ignore=['train_encoder'])
        self.model = DreamsClassifier(
            ckpt_path=ckpt_path,
            n_highest_peaks=n_highest_peaks,
            dropout=dropout,
            train_encoder=train_encoder,
        )

        self.train_acc = BinaryAccuracy()
        self.val_acc   = BinaryAccuracy()
        self.val_auc   = BinaryAUROC()

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.model(spec)

    def training_step(self, batch, batch_idx):
        x, y = batch['spec'], batch['label']
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        preds = torch.sigmoid(logits)

        bs = x.shape[0]
        self.log('train_loss', loss,    batch_size=bs)
        self.log('train_acc',  self.train_acc(preds, y.int()),
                 batch_size=bs, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['spec'], batch['label']
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        preds = torch.sigmoid(logits)

        bs = x.shape[0]
        self.log('val_loss', loss,    batch_size=bs, prog_bar=True)
        self.log('val_acc',  self.val_acc(preds, y.int()),
                 batch_size=bs, prog_bar=True)
        self.log('val_auc',  self.val_auc(preds, y.int()),
                 batch_size=bs, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch['spec'], batch['label']
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        preds = torch.sigmoid(logits)

        bs = x.shape[0]
        self.log('test_loss', loss,    batch_size=bs)
        self.log('test_acc',  self.val_acc(preds, y.int()),
                 batch_size=bs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)