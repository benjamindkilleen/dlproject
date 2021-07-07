#!/usr/bin/env python3

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
import dlproject
from dlproject.models import Classifier
from dlproject.datasets import MNISTDataModule


@dlproject.register_experiment
def mnist(cfg):
    pl.seed_everything(cfg.seed)

    # The ** converts the dictionary inside to key, value pairs.
    dm = MNISTDataModule(**cfg.dataset)
    model = Classifier(**cfg.model)
    dm.prepare_data()
    dm.setup(stage="fit")

    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)


@dlproject.register_experiment
def imagenet(cfg):
    pass


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    dlproject.run(cfg)


if __name__ == "__main__":
    main()
