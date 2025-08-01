#!/usr/bin/env python

import torch
_orig_torch_load = torch.load

def _patched_torch_load(f, *args, **kwargs):
    # Always allow full unpickling for trusted checkpoints
    kwargs['weights_only'] = False
    return _orig_torch_load(f, *args, **kwargs)

torch.load = _patched_torch_load

# Add project root to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from paths import PROJECT_ROOT

from datetime import datetime

import hydra
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import wandb

from benchmark.data.datasets import BinaryDetectionDataset
from benchmark.data.data_module import BenchmarkDataModule
from benchmark.models import MODEL_REGISTRY

@hydra.main(config_path="../configs", config_name="test_config")
def main(cfg):
    # Convert full Hydra config to dict for WandB
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # 1) WandB logger
    # prepend a timestamp so each run is unique
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{ts}_{cfg.logger.name}"

    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=run_name,
        config=config_dict
    )

    # 2) Callbacks
    ckpt_dir = PROJECT_ROOT / cfg.callbacks.checkpoint.dirpath
    checkpoint_cb = ModelCheckpoint(
        monitor=cfg.callbacks.checkpoint.monitor,
        mode=cfg.callbacks.checkpoint.mode,
        save_top_k=cfg.callbacks.checkpoint.save_top_k,
        save_last=True,
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

    # 6) Trainer
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger = wandb_logger,
        callbacks = [checkpoint_cb, lr_monitor],
    )

    # 8) Fit & Test
    try:
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule=datamodule)
    finally:
        wandb.finish()

if __name__ == '__main__':
    main()
