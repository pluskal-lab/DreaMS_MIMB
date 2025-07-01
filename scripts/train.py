import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from benchmark.data.datasets import ChlorineDetectionDataset
from benchmark.data.data_module import BenchmarkDataModule
from benchmark.models.classifier import MLPClassifier
from benchmark.models.lit_module import LitClassifier
from benchmark.utils import infer_spec_dim

from massspecgym.data.transforms import SpecBinner
from paths import PROJECT_ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (0 for CPU)")
    parser.add_argument(
        "--logger", choices=["tb", "wandb"], default="tb", help="Logger choice"
    )
    args = parser.parse_args()

    # 1. data & transforms
    spec_transform = SpecBinner(max_mz=1000.0, bin_width=1.0)
    dataset = ChlorineDetectionDataset(
        spec_transform=spec_transform,
        pth=PROJECT_ROOT / "data" / "massspecgym" / "MassSpecGym.mgf",
    )
    dm = BenchmarkDataModule(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 2. model
    input_dim = infer_spec_dim(spec_transform)
    model = MLPClassifier(input_dim=input_dim)
    lit_model = LitClassifier(model)

    # 3. logger
    if args.logger == "tb":
        logger = TensorBoardLogger("tb_logs", name="chlorine_detect")
    else:
        logger = WandbLogger(project="chlorine-detect")

    # 4. trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else None,
        strategy="ddp" if args.gpus > 1 else None,
        logger=logger,
    )

    # 5. run
    trainer.fit(lit_model, dm)
    trainer.test(lit_model, dm)

if __name__ == "__main__":
    main()