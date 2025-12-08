from pathlib import Path
import typing as tp
import logging
import json
import warnings
from copy import deepcopy
import sys
import itertools

from rich.prompt import Confirm
from rich.console import Console
import torch
from torch import Tensor
import lightning
from torchmetrics import Metric, MeanSquaredError, MeanAbsoluteError, MetricCollection
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, BackboneFinetuning

import torchani
from torchani.arch import _ANI
from torchani.units import hartree2kcalpermol
from torchani.annotations import PyScalar
from torchani.train import losses
from torchani.train._lit_callbacks import (
    SaveConfig,
    ModelCheckpointWithMetrics,
    NoLogLRMonitor,
)
from torchani.datasets import ANIBatchedDataset
from torchani.train.config import TrainConfig


console = Console()


def _get_dotted_name(module: tp.Any, name: str) -> tp.Any:
    parts = name.split(".")
    obj = module
    for part in parts:
        obj = getattr(obj, part)
    return obj


class LitModel(lightning.LightningModule):
    r"""
    ANI-style model, wrapped to enable training with PyTorch Lightning
    """

    def __init__(
        self,
        model: _ANI,
        loss_terms_and_factors: tp.Dict[str, float],
        optimizer_options: tp.Dict[str, PyScalar],
        scheduler_options: tp.Dict[str, PyScalar],
        monitor_label: str = "valid/rmse_default",
        optimizer_cls: str = "AdamW",
        scheduler_cls: str = "ReduceLROnPlateau",
        uncertainty_weighted: bool = False,
        num_head_layers: int = 0,
    ) -> None:
        super().__init__()
        self.optimizer_options = optimizer_options
        self.scheduler_options = scheduler_options
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls

        loss_terms = tuple(
            getattr(losses, name)(factor=factor)
            for name, factor in loss_terms_and_factors.items()
        )
        metrics: tp.Dict[str, tp.Union[Metric, MetricCollection]] = {}
        for term in loss_terms:
            for div in ("valid", "train"):
                # MeanSquaredError(squared=False) is directly the RMSE
                metrics[f"{div}/rmse_{term.label}"] = MeanSquaredError(squared=False)
                metrics[f"{div}/mae_{term.label}"] = MeanAbsoluteError()
        self.metrics = MetricCollection(metrics)

        if len(loss_terms) == 1 and monitor_label == "valid/rmse_default":
            monitor_label = f"valid/rmse_{loss_terms[0].label}"
        elif any(term.label == "forces" for term in loss_terms):
            monitor_label = "valid/rmse_forces"
        elif not any(monitor_label.endswith(term.label) for term in loss_terms):
            raise ValueError("Monitor label must be one of the enabled loss terms")
        self.monitor_label = monitor_label

        self.loss = losses.MultiTaskLoss(loss_terms, uncertainty_weighted)
        self.model = model

        # Hyperparameters
        self.save_hyperparameters(ignore="model")

        # Backbone for finetuning
        module_list = torch.nn.ModuleList()
        if num_head_layers > 0:
            for k in model.symbols:
                layers = model.neural_networks.atomics[k].layers
                last_layer = model.neural_networks.atomics[k].final_layer
                all_layers = itertools.chain(layers, [last_layer])
                module_list.extend(list(all_layers)[:-num_head_layers])
        self.backbone = module_list

    def set_loss(
        self,
        loss_terms_and_factors: tp.Dict[str, float],
        monitor_label: tp.Optional[str] = None,
        uncertainty_weighted: tp.Optional[bool] = None,
    ) -> None:
        loss_terms = tuple(
            getattr(losses, name)(factor=factor)
            for name, factor in loss_terms_and_factors.items()
        )
        metrics: tp.Dict[str, tp.Union[Metric, MetricCollection]] = {}
        for term in loss_terms:
            for div in ("valid", "train"):
                # MeanSquaredError(squared=False) is directly the RMSE
                metrics[f"{div}/rmse_{term.label}"] = MeanSquaredError(squared=False)
                metrics[f"{div}/mae_{term.label}"] = MeanAbsoluteError()
        self.metrics = MetricCollection(metrics)

        if monitor_label is not None:
            if len(loss_terms) == 1 and monitor_label == "valid/rmse_default":
                monitor_label = f"valid/rmse_{loss_terms[0].label}"
            elif any(term.label == "forces" for term in loss_terms):
                monitor_label = "valid/rmse_forces"
            elif not any(monitor_label.endswith(term.label) for term in loss_terms):
                raise ValueError("Monitor label must be one of the enabled loss terms")
            self.monitor_label = monitor_label
        if uncertainty_weighted is not None:
            self.loss = losses.MultiTaskLoss(loss_terms, uncertainty_weighted)
        else:
            self.loss = losses.MultiTaskLoss(loss_terms)

    def on_train_start(self) -> None:
        # Log hyperparameters to tensorboard events file (only a single time)
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                # hparams seems to be of the correct type, but lightning marks it
                # differently
                logger.log_hyperparams(self.hparams)  # type: ignore

    def training_step(
        self,
        batch: tp.Dict[str, Tensor],
        batch_idx: int,
    ) -> Tensor:
        pred = self.batch_eval(batch)
        with torch.no_grad():
            self._update_metrics("train", pred, batch)
        loss_dict = self.loss(pred, batch)
        return loss_dict["loss"]

    def validation_step(
        self,
        batch: tp.Dict[str, Tensor],
        batch_idx: int,
    ) -> None:
        with torch.enable_grad():
            pred = self.batch_eval(batch)
        self._update_metrics("valid", pred, batch)

    def _update_metrics(
        self, div: str, pred: tp.Dict[str, Tensor], batch: tp.Dict[str, Tensor]
    ) -> None:
        for k, v in self.metrics.items():
            if not k.startswith(f"{div}/"):
                continue
            label = "_".join(k.split("_")[1:])
            v.update(pred[label], batch[self.loss.term(label).targ_label])

    # Metrics are logged at the end of each validation epoch only
    # This is only correct if check_val_every_n_epochs=1
    def on_validation_epoch_end(self) -> None:
        results = {}
        for k, c in self.metrics.items():
            if not c.update_called:
                continue
            results[k] = c.compute()
            c.reset()
            if "energies" in k:
                results[f"{k}_kcal|mol"] = hartree2kcalpermol(results[k])
            elif "forces" in k:
                results[f"{k}_kcal|mol|ang"] = hartree2kcalpermol(results[k])

        # I believe callbacks is technically pvt API
        for c in self.trainer.callbacks:  # type: ignore
            if isinstance(c, NoLogLRMonitor):
                results.update(c.extract_stats(self.trainer))
                break

        self.log_dict(results)

    def batch_eval(self, batch: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        for term in self.loss.grad_terms:
            # e.g. batch["coordinates"].requires_grad_(True)
            batch[term.grad_wrt_targ_label].requires_grad_(True)

        # Rename common synonyms
        if "energy" in batch:
            batch["energies"] = batch.pop("energy").view(-1)
        if "force" in batch:
            batch["forces"] = batch.pop("force")
        if "coords" in batch:
            batch["coordinates"] = batch.pop("coords")

        if "cell" in batch:
            # Periodic
            # TODO: Remove float casts
            pred = self.model(
                (batch["species"], batch["coordinates"].float()),
                cell=batch["cell"].view(3, 3).float(),
                pbc=torch.tensor(
                    [True, True, True], dtype=torch.bool, device=batch["species"].device
                ),
            )._asdict()
        else:
            pred = self.model(
                (batch["species"], batch["coordinates"].float())
            )._asdict()
        pred.pop("species")

        for term in self.loss.grad_terms:
            pred[term.label] = (-1 if term.negative_grad else 1) * torch.autograd.grad(
                pred[term.grad_of_label].sum(),
                batch[term.grad_wrt_targ_label],
                retain_graph=True,
                create_graph=True,
            )[0]

        for term in self.loss.grad_terms:
            batch[term.grad_wrt_targ_label].requires_grad_(False)
        return pred

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Optimizer setup
        opt_type = getattr(torch.optim, self.optimizer_cls)
        optimizer = opt_type(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.optimizer_options,
        )
        scheduler_type = getattr(torch.optim.lr_scheduler, self.scheduler_cls)
        # Schedulers setup
        scheduler = scheduler_type(optimizer=optimizer, **self.scheduler_options)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "strict": True,
            "monitor": self.monitor_label,
        }
        return tp.cast(
            OptimizerLRScheduler,
            {"optimizer": optimizer, "lr_scheduler": scheduler_config},
        )


def train_lit_model(
    config: TrainConfig,
    restart: bool = False,
    allow_restart: bool = False,
    verbose: bool = False,
    loss_terms_and_factors: tp.Optional[tp.Dict[str, float]] = None,
    wandb_entity: str = "nnip",
    wandb_project: str = "ani",
    log_wandb: bool = False,
) -> None:
    r"""Train an ANI-style neural network potential using PyTorch Lightning"""

    if not restart and loss_terms_and_factors is not None:
        raise ValueError("Loss terms and factors only valid for restarts")

    if not verbose:
        from lightning_utilities.core.rank_zero import log

        log.setLevel(logging.ERROR)

    if not restart and config.path.is_dir():
        if allow_restart or Confirm.ask("Run exists, do you want to restart it?"):
            # Reload config from the path
            accel = deepcopy(config.accel)
            config = TrainConfig.from_json_file(config.path / "config.json")
            console.print("Overriding accel config")
            config.accel = accel
            restart = True
        else:
            console.print("Exiting without training")
            sys.exit(0)

    if not config.ds.path.exists():
        raise RuntimeError("Dataset does not exist")

    if not config.model.builtin:
        # TODO: Bw compat, this only happens if config is old file that has not lot, or
        # no symbols Remove in the future since it is confusing, and fails for ftune
        if not config.model.lot:
            assert not config.ftune
            assert restart
            warnings.warn("Model LoT not found, assuming equal to ds lot")
            lot = config.ds.lot
        else:
            lot = config.model.lot

        if not config.model.symbols:
            assert not config.ftune
            assert restart
            warnings.warn("Model symbols not found, assuming equal to ds symbols")
            with open(config.ds.path / "creation_log.json", mode="rt") as f:
                symbols = json.load(f)["symbols"]
        else:
            symbols = config.model.symbols

        model = _get_dotted_name(torchani, f"arch.{config.model.arch_fn}")(
            lot=lot,
            symbols=symbols,
            strategy="auto" if config.accel.device in ["cuda", "gpu"] else "pyaev",
            **config.model.options,
        )
    else:
        model = _get_dotted_name(torchani, f"models.{config.model.arch_fn}")(
            strategy="auto" if config.accel.device in ["cuda", "gpu"] else "pyaev",
            **config.model.options,
        )
        model.requires_grad_(True)

    ckpt_path = (config.path / "latest-model") / "latest.ckpt"

    if config.ftune is not None:
        # If a checkpoint path exists this is not needed
        if config.ftune.pretrained_state_dict and not ckpt_path.is_file():
            model.load_state_dict(config.ftune.pretrained_state_dict)

    # Not sure what the problem with mypy is here, it infers LitModel to have
    # type[Never]
    lit_model: tp.Any
    if ckpt_path.is_file():
        # Rewrite ckpt to modify the early-stopping callback, since all callbacks
        # get overwritten on restart
        ckpt = torch.load(ckpt_path)
        callbacks = deepcopy(ckpt["callbacks"])
        for k, v in callbacks.items():
            if k.startswith("EarlyStopping"):
                v["patience"] = config.accel.early_stop_patience
                ckpt["callbacks"][k] = v
        torch.save(ckpt, ckpt_path)

        lit_model = LitModel.load_from_checkpoint(  # type: ignore
            ckpt_path,
            model=model,
        )
        if loss_terms_and_factors:
            lit_model.set_loss(loss_terms_and_factors)
    else:
        no_ftune = config.ftune is None or config.ftune.dummy_ftune
        lit_model = LitModel(  # type: ignore
            model,
            loss_terms_and_factors=config.loss.terms_and_factors,
            monitor_label=config.monitor_label,
            # Loss
            uncertainty_weighted=config.loss.uncertainty_weighted,
            # Optim
            optimizer_cls=config.optim.cls,
            optimizer_options=config.optim.options,
            # Scheduler
            scheduler_cls=config.scheduler.cls,
            scheduler_options=config.scheduler.options,
            # Ftune
            num_head_layers=(
                0 if no_ftune else getattr(config.ftune, "num_head_layers", 0)
            ),
        )

    if restart:
        console.print(f"Restarting run {config.path}")
    else:
        init_model_path = config.path / "init-model"
        init_model_path.mkdir(exist_ok=False, parents=True)
        torch.save(
            {"state_dict": lit_model.state_dict()}, init_model_path / "init.ckpt"
        )

    kwargs = {
        "num_workers": config.accel.num_workers,
        "prefetch_factor": (
            config.accel.prefetch_factor if config.accel.num_workers > 0 else None
        ),
        "pin_memory": True,
    }
    _fold_idx = config.ds.fold_idx if config.ds.fold_idx != "train" else ""
    training = ANIBatchedDataset(
        config.ds.path,
        split=f"training{_fold_idx}",
        limit=config.accel.train_limit or 1.0,
    ).as_dataloader(
        shuffle=True, **kwargs  # type: ignore
    )
    validation = ANIBatchedDataset(
        config.ds.path,
        split=f"validation{_fold_idx}",
        limit=config.accel.validation_limit or 1.0,
    ).as_dataloader(
        shuffle=False, **kwargs  # type: ignore
    )

    # Build all callbacks required for training
    lr_monitor = NoLogLRMonitor()
    early_stopping = EarlyStopping(
        monitor=lit_model.monitor_label,
        strict=True,
        mode="min",
        patience=config.accel.early_stop_patience,
    )
    best_model_ckpt = ModelCheckpointWithMetrics(
        dirpath=config.path / "best-model",
        filename="best",
        save_top_k=1,
        enable_version_counter=False,
        # Specific configuration for saving the best model
        monitor=lit_model.monitor_label,
        mode="min",
        save_weights_only=True,
    )
    latest_model_ckpt = ModelCheckpointWithMetrics(
        dirpath=config.path / "latest-model",
        filename="latest",
        save_top_k=1,
        enable_version_counter=False,
    )
    save_model_config = SaveConfig(config)
    callbacks = [
        lr_monitor,
        early_stopping,
        best_model_ckpt,
        latest_model_ckpt,
        save_model_config,
    ]

    # Finetuning configuration, "dummy ftune" just performs normal training
    if config.ftune is not None and not config.ftune.dummy_ftune:
        if config.ftune.frozen_backbone:
            unfreeze_epoch = config.accel.max_epochs + 1
        else:
            unfreeze_epoch = 0
        ftune_callback = BackboneFinetuning(
            lambda_func=lambda epoch: 1.0,
            backbone_initial_lr=config.ftune.backbone_lr,
            unfreeze_backbone_at_epoch=unfreeze_epoch,
            should_align=False,
            train_bn=False,
            verbose=False,
        )
        callbacks.append(ftune_callback)

    # Build all loggers required for training
    (config.path / "tb-logs").mkdir(exist_ok=True, parents=True)
    tb_logger = TensorBoardLogger(
        save_dir=config.path, name="tb-logs", default_hp_metric=True
    )
    (config.path / "csv-logs").mkdir(exist_ok=True, parents=True)
    csv_logger = CSVLogger(save_dir=config.path, name="csv-logs")
    loggers = [tb_logger, csv_logger]

    if log_wandb:
        wandb_logger = WandbLogger(
            save_dir=Path(config.path) / "wandb-logs",
            id=config.name,
            name=config.name,
            entity=wandb_entity,
            project=wandb_project,
        )
        loggers.append(wandb_logger)

    trainer = lightning.Trainer(
        default_root_dir=config.path,
        devices=1,
        accelerator=config.accel.device.replace("cuda", "gpu"),
        max_epochs=config.accel.max_epochs,
        log_every_n_steps=config.accel.log_interval,
        deterministic=config.accel.deterministic,
        detect_anomaly=config.accel.detect_anomaly,
        profiler=config.accel.profiler,
        check_val_every_n_epoch=1,  # Assumed by TorchANI for logging
        # Callbacks and loggers
        logger=loggers,
        callbacks=callbacks,
    )
    if verbose:
        print(lit_model.model)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", message="Checkpoint directory.*", category=UserWarning
        )
        trainer.fit(
            lit_model,
            train_dataloaders=training,
            val_dataloaders=validation,
            ckpt_path=ckpt_path if ckpt_path.is_file() else None,
        )
