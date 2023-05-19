"""
This script trains and evaluates a FFNN on the semantic security game.
The data is either feature vectors or raw ciphertexts.
This script is used to tune and find the optimal hyper parameters, which are then hardcoded in the src/attacking/pipeline.py script.
"""
import argparse
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import relic
import sklearn.model_selection
import sklearn.pipeline
import torch

from .. import logging
from . import semantic_security

logger = logging.init(__name__)

Kwargs = Dict[str, Any]


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    relic.cli.lib.shared.add_filter_options(parser)
    return parser


class FFNN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, p_dropout: float):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p_dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.mlp(x)


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_criterion,
        optimizer_cls: str,
        optimizer_lr,
        optimizer_weight_decay,
    ):
        super().__init__()
        self.model = model
        self.loss_criterion = loss_criterion
        self.optimizer_cls = optimizer_cls
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay

    def training_step(self, batch, batch_idx):
        x, true_y = batch
        logits = self.model(x)
        loss = self.loss_criterion(logits, true_y.view(*logits.shape))
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, true_y = batch
        logits = self.model(x)
        loss = self.loss_criterion(logits, true_y.view(*logits.shape))
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx):
        logits = self.model(batch)
        predictions = torch.zeros_like(logits)
        predictions[torch.sigmoid(logits) > 0.5] = 1
        return predictions

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            self.parameters(),
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )
        return optimizer


class SemanticSecurityModel(sklearn.base.BaseEstimator):
    def __init__(
        self,
        *,
        # Model parameters
        model_input_dim: int,
        model_hidden_dim: int = 1024,
        model_p_dropout: float = 0.5,
        # Loss parameters
        loss_cls="torch.nn.BCEWithLogitsLoss",
        # Optimizer parameters
        optimizer_cls=torch.optim.Adam,
        optimizer_lr: float = 1e-4,
        optimizer_weight_decay: float = 1e-2,
        # Training parameters
        training_batch_size: int = 16,
        training_max_epochs: int = 10_000,
        gpus: int = 1,
        # Keyword args
        loss_kwargs: Optional[Kwargs] = None,
    ):
        if loss_kwargs is None:
            loss_kwargs = {}

        self.model_input_dim = model_input_dim
        self.model_hidden_dim = model_hidden_dim
        self.model_p_dropout = model_p_dropout
        self.loss_cls = loss_cls
        self.optimizer_cls = optimizer_cls
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.training_batch_size = training_batch_size
        self.training_max_epochs = training_max_epochs
        self.gpus = gpus
        self.loss_kwargs = loss_kwargs
        self.trainer = None
        self.dataloader_kwargs = {
            "num_workers": 16,
            "drop_last": True,
            "batch_size": self.training_batch_size,
        }

    def train_dataloader(self, X, y):
        dataset = [(xi, yi) for xi, yi in zip(X, y)]
        return torch.utils.data.DataLoader(
            dataset, shuffle=True, **self.dataloader_kwargs
        )

    def test_dataloader(self, X, y):
        dataset = [(xi, yi) for xi, yi in zip(X, y)]
        return torch.utils.data.DataLoader(
            dataset, shuffle=False, **self.dataloader_kwargs
        )

    def predict_dataloader(self, X):
        return torch.utils.data.DataLoader(
            torch.tensor(X), shuffle=False, **self.dataloader_kwargs
        )

    def fit(self, X, y):
        pl.seed_everything(42, workers=True)

        self.model = LightningModel(
            FFNN(
                self.model_input_dim,
                self.model_hidden_dim,
                self.model_p_dropout,
            ),
            eval(self.loss_cls)(**self.loss_kwargs),
            self.optimizer_cls,
            self.optimizer_lr,
            self.optimizer_weight_decay,
        )

        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=self.gpus,
            max_epochs=self.training_max_epochs,
            deterministic=True,
            log_every_n_steps=1,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="train_loss", patience=5, mode="min", strict=True
                )
            ],
            enable_checkpointing=False,
            enable_progress_bar=True,
            logger=False,
            enable_model_summary=False,
        )

        self.trainer.fit(
            model=self.model,
            train_dataloaders=self.train_dataloader(X, y),
            val_dataloaders=None,
        )

    def predict(self, X):
        if self.trainer is None:
            raise RuntimeError("Must call .fit() first!")

        return (
            torch.cat(
                self.trainer.predict(self.model, dataloaders=self.predict_dataloader(X))  # type: ignore
            )
            .squeeze()
            .numpy()
        )

    def score(self, X, y) -> float:
        predictions = self.predict(X)
        return (predictions == y).sum() / len(y)


def init_model(feature_fn: Optional[object]) -> semantic_security.Model:
    if feature_fn:
        model_input_dim = 6
        hidden_dim = 256
    else:
        model_input_dim = 10_000
        hidden_dim = 1_000

    p_dropout = 0.1
    batch_size = 32
    optimizer_cls = torch.optim.AdamW
    optimizer_lr = 3e-4
    optimizer_weight_decay = 0.1

    return sklearn.pipeline.Pipeline(
        steps=[
            ("scaler", sklearn.preprocessing.StandardScaler()),
            (
                "ffnn",
                SemanticSecurityModel(
                    model_input_dim=model_input_dim,
                    model_hidden_dim=hidden_dim,
                    model_p_dropout=p_dropout,
                    training_batch_size=batch_size,
                    optimizer_cls=optimizer_cls,
                    optimizer_lr=optimizer_lr,
                    optimizer_weight_decay=optimizer_weight_decay,
                    training_max_epochs=10000,
                ),
            ),
        ]
    )
