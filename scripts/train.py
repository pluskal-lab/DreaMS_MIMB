#!/usr/bin/env python
import sys
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from paths import PROJECT_ROOT

from benchmark.data.datasets import BinaryDetectionDataset
from benchmark.data.data_module import BenchmarkDataModule
from benchmark.models import MODEL_REGISTRY

@hydra.main(config_path="../configs", config_name="test_config")
def main(cfg):
    # Convert full Hydra config to dict for WandB
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # 1) WandB logger
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=cfg.logger.name,
        config=config_dict
    )

    # 2) Callbacks
    ckpt_dir = PROJECT_ROOT / cfg.callbacks.checkpoint.dirpath
    checkpoint_cb = ModelCheckpoint(
        monitor=cfg.callbacks.checkpoint.monitor,
        mode=cfg.callbacks.checkpoint.mode,
        save_top_k=cfg.callbacks.checkpoint.save_top_k,
        dirpath=str(ckpt_dir)
    )
    lr_monitor = LearningRateMonitor(logging_interval=cfg.callbacks.lr_monitor.logging_interval)

    # 3) Annotate MGF
    input_mgf = PROJECT_ROOT / cfg.data.mgf_path
    output_mgf = PROJECT_ROOT / cfg.data.labeled_mgf
    BinaryDetectionDataset.annotate_mgf(
        input_pth=input_mgf,
        output_pth=output_mgf,
        label_fn=lambda md: float(cfg.data.label_element in md.get("formula", ""))
    )

    # 4) DataModule
    spec_transform = hydra.utils.instantiate(cfg.data.transform)
    dataset = hydra.utils.instantiate(
        cfg.data.dataset,
        pth=output_mgf,
        spec_transform=spec_transform,
    )
    datamodule = hydra.utils.instantiate(
        cfg.data.datamodule,
        dataset=dataset,
    )

    # 5) Model instantiation
    model_cls = MODEL_REGISTRY[cfg.model.name]
    # Resolve checkpoint path
    ckpt_path = Path(cfg.model.hparams.ckpt_path)
    if not ckpt_path.is_absolute():
        cfg.model.hparams.ckpt_path = str(PROJECT_ROOT / ckpt_path)
    hparams = OmegaConf.to_container(cfg.model.hparams, resolve=True)
    model = model_cls(**hparams)

        # 6) Trainer hardware args
    if torch.cuda.is_available():
        accel_kwargs = {'accelerator': 'gpu', 'devices': torch.cuda.device_count()}
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        accel_kwargs = {'accelerator': 'mps', 'devices': 1}
    else:
        accel_kwargs = {'accelerator': 'cpu', 'devices': 1}

    # 7) Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        precision=cfg.trainer.precision,
        **accel_kwargs
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        precision=cfg.trainer.precision,
        **accel_kwargs
    )

    # 8) Fit & Test
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == '__main__':
    main()
