import itertools
import typing as tp

import torch
from torch import Tensor
import lightning
from torchmetrics import Metric, MeanSquaredError, MeanAbsoluteError, MetricCollection
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.loggers import TensorBoardLogger

from torchani.arch import _ANI
from torchani.units import hartree2kcalpermol
from torchani.annotations import PyScalar

from torchani.train import losses
from torchani.train._lit_callbacks import NoLogLRMonitor


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
                rev_layers = itertools.chain([last_layer], reversed(layers))
                module_list.extend(list(rev_layers)[:-num_head_layers])
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
