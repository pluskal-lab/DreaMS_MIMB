import torch
import torch.nn.functional as F
import pytorch_lightning as pl
# either:
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
# or (if you prefer the unified API):
# from torchmetrics import Accuracy, AUROC


class LitClassifier(pl.LightningModule):
    """
    Lightning wrapper around a binary classifier.
    """

    def __init__(self, model, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

        # Option 1: use the specialized metrics
        self.train_acc = BinaryAccuracy()
        self.val_acc   = BinaryAccuracy()
        self.val_auc   = BinaryAUROC()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        specs, labels = batch["spec"], batch["label"]
        bs = specs.shape[0]
        logits = self(specs)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        preds = torch.sigmoid(logits)

        self.log("train_loss", loss, batch_size=bs)
        self.log("train_acc", self.train_acc(preds, labels.int()),
                 batch_size=bs, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        logits = self(batch["spec"])
        loss   = F.binary_cross_entropy_with_logits(logits, batch["label"])
        preds  = torch.sigmoid(logits)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc",  self.val_acc(preds, batch["label"].int()), prog_bar=True)
        self.log("val_auc",  self.val_auc(preds, batch["label"].int()), prog_bar=True)

    def test_step(self, batch, _):
        logits = self(batch["spec"])
        loss   = F.binary_cross_entropy_with_logits(logits, batch["label"])
        preds  = torch.sigmoid(logits)

        self.log("test_loss", loss)
        # You can reuse your val_acc here or create a fresh one
        self.log("test_acc",  self.val_acc(preds, batch["label"].int()))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)