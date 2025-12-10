from pathlib import Path
import typing as tp
import logging
import warnings
from copy import deepcopy
import sys
import itertools
import importlib

from rich.prompt import Confirm
from rich.console import Console
from typer import Abort
import torch
from torch import Tensor
import lightning
from torchmetrics import Metric, MeanSquaredError, MeanAbsoluteError, MetricCollection
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, BackboneFinetuning

import torchani
from torchani.electro import DipoleComputer
from torchani.arch import _ANI
from torchani.units import hartree2kcalpermol
from torchani.annotations import PyScalar
from torchani.train import losses
from torchani.train._lit_callbacks import (
    SaveConfig,
    ModelCheckpointWithMetrics,
    NoLogLRMonitor,
    EMA,
    PhaseChange,
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


def _load_arch_fn_from_file(path: str | Path, arch_fn: str) -> tp.Any:
    path = Path(path).resolve()
    # Create a module spec from the path
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None:
        raise ValueError(f"Could not load spec form file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module

    # Load and execute the module
    loader = spec.loader
    if loader is None:
        raise ValueError(f"Could not construct loader from file {path}")
    loader.exec_module(module)
    return getattr(module, arch_fn)


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
        assume_neutral: bool = True,
    ) -> None:
        super().__init__()
        self.optimizer_options = optimizer_options
        self.scheduler_options = scheduler_options
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls
        # Assume molecules are neutral if total charge is not provided
        self._assume_neutral = assume_neutral

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

        # Auxiliary module to compute dipoles if required
        # mypy complains but this is valid, too lazy to type correctly
        self._dipole_computer = DipoleComputer(
            device=self.device, dtype=self.dtype  # type: ignore
        )

    def set_loss_factors(self, terms_and_factors: tp.Dict[str, float]) -> None:
        for label, factor in terms_and_factors.items():
            if not self.loss.is_enabled(label):
                if factor > 0.0:
                    raise ValueError("Can only modify already enabled terms")
                continue
            self.loss.term(label).factor = factor

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
            if label not in pred:
                console.print(
                    f"Loss has {label} but model doesn't predict it", style="red"
                )
                console.print(f"Predicted labels are: {pred.keys()}", style="red")
                raise Abort()
            targ_label = self.loss.term(label).targ_label
            if targ_label not in batch:
                console.print(
                    f"Loss has {targ_label} but dataset doesn't provide it", style="red"
                )
                console.print(f"Provided labels are: {batch.keys()}", style="red")
                raise Abort()
            v.update(pred[label], batch[targ_label])

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
        # Rename common synonyms
        if "energy" in batch:
            batch["energies"] = batch.pop("energy").view(-1)
        if "force" in batch:
            batch["forces"] = batch.pop("force")
        if "coords" in batch:
            batch["coordinates"] = batch.pop("coords")
        if "total_charges" in batch:
            batch["total_charge"] = batch.pop("total_charges")

        for term in self.loss.grad_terms:
            # e.g. batch["coordinates"].requires_grad_(True)
            batch[term.grad_wrt_targ_label].requires_grad_(True)

        if "cell" in batch:
            # Periodic
            # TODO: Remove float casts?
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

        # Generate dipoles and total charge if needed
        if self.loss.is_enabled("dipoles") and "dipoles" not in pred:
            if "atomic_charges" not in pred:
                console.print(
                    "'atomic_charges' or 'dipoles' required to calculate 'dipoles',"
                    " but model doesn't predict either",
                    style="red",
                )
                console.print(f"Predicted labels are: {pred.keys()}", style="red")
                raise Abort()
            pred["dipoles"] = self._dipole_computer(
                batch["species"], batch["coordinates"], pred["atomic_charges"]
            )
        if self.loss.is_enabled("total_charge") and "total_charge" not in pred:
            if self._assume_neutral and "total_charge" not in batch:
                _coords = batch["coordinates"]
                batch["total_charge"] = _coords.new_zeros(_coords.shape[0])

            if "atomic_charges" not in pred:
                console.print(
                    "'atomic_charges' or 'total_charge' required for 'total_charge',"
                    " but model doesn't predict either",
                    style="red",
                )
                console.print(f"Predicted labels are: {pred.keys()}", style="red")
                raise Abort()
            pred["total_charge"] = pred["atomic_charges"].sum(-1)

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
    print_model: bool = True,
) -> None:
    r"""Train an ANI-style neural network potential using PyTorch Lightning

    Returns the final learning rate
    """

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

    if config.model.builtin:
        # Directly instantiate a builtin TorchANI model
        model = _get_dotted_name(torchani, f"models.{config.model.arch_fn}")(
            strategy="auto" if config.accel.device in ["cuda", "gpu"] else "pyaev",
            **config.model.options,
        )
        model.requires_grad_(True)
    elif config.model.arch_file:
        # Custom architecture located in a user-provided file
        # The resulting model is assumed to conform to the ANI API
        lot = config.model.lot
        symbols = config.model.symbols
        model = _load_arch_fn_from_file(config.model.arch_file, config.model.arch_fn)(
            lot=lot, symbols=symbols, **config.model.options
        )
    else:
        # Build the model from an arch-function
        lot = config.model.lot
        symbols = config.model.symbols
        model = _get_dotted_name(torchani, f"arch.{config.model.arch_fn}")(
            lot=lot,
            symbols=symbols,
            strategy="auto" if config.accel.device in ["cuda", "gpu"] else "pyaev",
            **config.model.options,
        )

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
            lit_model.set_loss_factors(loss_terms_and_factors)
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
        best_model_ckpt,
        latest_model_ckpt,
        save_model_config,
    ]
    if config.accel.early_stop_patience != -1:
        early_stopping = EarlyStopping(
            monitor=lit_model.monitor_label,
            strict=True,
            mode="min",
            patience=config.accel.early_stop_patience,
        )
        callbacks.append(early_stopping)

    if config.do_ema:
        ema = EMA(config.ema_decay, config.ema_mode)
        callbacks.append(ema)

    if config.phase_changes:
        for p in config.phase_changes:
            p = p.copy()
            epoch = int(p.pop("epoch"))
            new_lr = p.pop("lr", None)
            phase_change = PhaseChange(
                epoch=epoch, new_lr=new_lr, new_loss_terms_and_factors=p
            )
            callbacks.append(phase_change)

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
    if verbose and print_model:
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
