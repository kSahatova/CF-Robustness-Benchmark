import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from src.cf_methods.coin.utils import Logger
from src.cf_methods.coin.utils import get_experiment_folder_path


class BaseTrainer:
    def __init__(
        self, opt: edict, model: nn.Module, continue_path: Optional[str] = None
    ) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.opt = opt
        self.model = model.to(self.device)
        self.batches_done = 0
        self.current_epoch = 0

        continue_path = Path(continue_path) if continue_path else None
        if continue_path and continue_path.suffix == ".pth":
            self.logging_dir = continue_path.parent.parent
            self.ckpt_name = continue_path.name
        elif continue_path:
            self.logging_dir = Path(continue_path)
            self.ckpt_name = None
        else:
            self.logging_dir = Path(
                get_experiment_folder_path(
                    opt.logging_dir, opt.get("experiment_name", "exp")
                )
            )
            self.ckpt_name = None
        self.vis_dir = self.logging_dir / "visualizations"
        self.ckpt_dir = self.logging_dir / "checkpoints"
        self.logger = Logger(self.logging_dir)
        if continue_path is not None:
            assert self.logging_dir.exists(), (
                f"Unable to find model directory {continue_path}"
            )
            self.restore_state()
        else:
            self.vis_dir.mkdir(exist_ok=True)
            self.ckpt_dir.mkdir(exist_ok=True)
            # self.logger.reset()
        logging.info(f"Using model: {self.model.__class__.__name__}")

    @abstractmethod
    def restore_state(self):
        """Restore trainer's state based on the latest checkpoint from `self.ckpt_dir`"""

    @abstractmethod
    def save_state(self) -> str:
        """Persists trainer's state to the `self.ckpt_dir`"""

    @abstractmethod
    def training_epoch(self, loader: torch.utils.data.DataLoader) -> dict:
        """Runs a training epoch for a given loader and returns the results, which can include epoch loss, metrics, etc"""

    @torch.no_grad()
    def validation_epoch(self, loader: torch.utils.data.DataLoader) -> dict:
        """Runs a validation epoch for a given loader and returns the results, which can include epoch loss, metrics, etc"""

    @abstractmethod
    def get_dataloaders(self) -> tuple[torch.utils.data.DataLoader]:
        pass

    def fit(self, dataloaders: List[DataLoader]):
        """Starts the training of the provided  model using train and validation dataloaders"""
        train_loader, val_loader = dataloaders
        for _ in range(self.current_epoch, self.opt.n_epochs):
            _ = self.training_epoch(train_loader)
            _ = self.validation_epoch(val_loader)
            if self.current_epoch % self.opt.checkpoint_freq == 0:
                ckpt_path = self.save_state()
                self.logger.info(
                    f"Saved checkpoint parameters at epoch {self.current_epoch}: {ckpt_path}"
                )
            self.current_epoch += 1
        ckpt_path = self.save_state()
        self.logger.info(
            f"Saved checkpoint parameters at epoch {self.current_epoch}: {ckpt_path}"
        )
