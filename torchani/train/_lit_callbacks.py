r"""
Custom Lightning compatible callbacks
"""

import shutil
from copy import deepcopy
import json
import typing as tp
from pathlib import Path

from rich.console import Console
import torch
from torch import Tensor
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from lightning import Trainer, LightningModule

from torchani.train.config import TrainConfig

console = Console()


class NoLogLRMonitor(LearningRateMonitor):
    r"""
    A learning rate monitor that doesn't automatically log
    """

    def __init__(
        self, log_momentum: bool = False, log_weight_decay: bool = False
    ) -> None:
        super().__init__(
            logging_interval="epoch",
            log_momentum=log_momentum,
            log_weight_decay=log_weight_decay,
        )

    def on_train_epoch_start(
        self, trainer: Trainer, *args: tp.Any, **kwargs: tp.Any
    ) -> None:
        pass

    def on_train_batch_start(
        self, trainer: Trainer, *args: tp.Any, **kwargs: tp.Any
    ) -> None:
        pass

    # TODO: This fn uses a pvt method of lightning, which is not ideal
    def extract_stats(self, trainer: Trainer) -> tp.Dict[str, float]:
        return self._extract_stats(trainer, "epoch")


class ModelCheckpointWithMetrics(ModelCheckpoint):
    r"""
    Checkpoint a model and also save the callback metrics from the trainer
    """

    def check_monitor_top_k(
        self, trainer: Trainer, current: tp.Optional[Tensor] = None
    ) -> bool:
        should_update_and_save = super().check_monitor_top_k(trainer, current)
        if should_update_and_save:
            self._dump_metrics(trainer)
        return should_update_and_save

    def _save_topk_checkpoint(
        self, trainer: Trainer, monitor_candidates: tp.Dict[str, Tensor]
    ) -> None:
        super()._save_topk_checkpoint(trainer, monitor_candidates)
        # In this case check_monitor_top_k is not called, and the
        # metrics should be dump every step
        if self.monitor is not None:
            return
        self._dump_metrics(trainer)

    def _dump_metrics(self, trainer: Trainer) -> None:
        #  names of metrics are (energies|...)_(train|valid)_(rmse|mae)[kcal|mol[|ang]]
        candidates = trainer.callback_metrics
        if self.dirpath is not None:
            dirpath = Path(self.dirpath).resolve()
        else:
            dirpath = Path(trainer.default_root_dir).resolve()
        dirpath.mkdir(exist_ok=True)

        metrics: tp.Dict[str, tp.Union[int, float]] = {"epoch": trainer.current_epoch}
        for k, v in candidates.items():
            if k.startswith("valid/") or k.startswith("train/"):
                metrics[k] = v.item()
        with open(dirpath / "metrics.json", mode="wt", encoding="utf-8") as ft:
            json.dump(metrics, ft, indent=4)

    # Save the EMA weights
    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:

        for callback in trainer.callbacks:  # type: ignore
            if isinstance(callback, EMA):
                callback.make_ema(trainer.lightning_module)

        super()._save_checkpoint(trainer, filepath)

        for callback in trainer.callbacks:  # type: ignore
            if isinstance(callback, EMA):
                callback.make_non_ema(trainer.lightning_module)


class SaveConfig(Callback):
    r"""
    Save the configuration of a training run at the start of the run
    """

    def __init__(
        self,
        config: TrainConfig,
    ) -> None:
        super().__init__()
        self._config = config

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        root = Path(trainer.default_root_dir).resolve()
        self._config.to_json_file(root / "config.json")
        # Copy the contents of the arch file used to instantiate the model, if it exists
        # (file must be self-contained)
        if self._config.model.arch_file:
            shutil.copy(self._config.model.arch_file, root / "arch_file.py")


class EMA(Callback):
    def __init__(self, decay: float = 0.99, mode: str = "batch"):
        assert mode in ("batch", "epoch")
        self._decay = decay
        self._num_updates = 0
        self._mode = mode
        self._pl_module_is_ema = False
        self._ema_model: torch.nn.Module | None = None

    @property
    def decay(self) -> float:
        return self._decay

    @property
    def num_updates(self) -> int:
        return self._num_updates

    @property
    def pl_module_is_ema(self) -> bool:
        return self._pl_module_is_ema

    def on_fit_start(self, trainer, pl_module):
        # Make a frozen copy of the model
        self._ema_model = deepcopy(pl_module.model)
        self._ema_model.to(pl_module.device)
        self._ema_model.requires_grad_(False)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._mode != "batch":
            return
        self.update_params(pl_module)

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module: tp.Any):
        if self._mode != "epoch":
            return
        self.update_params(pl_module)

    @torch.no_grad()
    def swap_params(self, pl_module):
        assert self._ema_model is not None
        for ema_p, p in zip(self._ema_model.parameters(), pl_module.model.parameters()):
            tmp = p.clone()
            p.copy_(ema_p.data)
            ema_p.copy_(tmp)

        self._pl_module_is_ema = not self._pl_module_is_ema

    @torch.no_grad()
    def update_params(self, pl_module):
        assert self._ema_model is not None
        assert not self.pl_module_is_ema
        # The first ~1000 updates use a much smaller decay
        # from https://github.com/fadel/pytorch_ema
        # This prevents initial weights from having too much influence
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        for ema_p, p in zip(self._ema_model.parameters(), pl_module.model.parameters()):
            if p.requires_grad:
                ema_p.mul_(decay).add_(p, alpha=1 - decay)

        self._num_updates += 1

    # Use the EMA weights during validation
    def on_validation_start(self, trainer, pl_module):
        self.make_ema(pl_module)

    def on_validation_end(self, trainer, pl_module):
        self.make_non_ema(pl_module)

    def make_ema(self, pl_module) -> None:
        if not self.pl_module_is_ema:
            self.swap_params(pl_module)

    def make_non_ema(self, pl_module) -> None:
        if self.pl_module_is_ema:
            self.swap_params(pl_module)


class PhaseChange(Callback):
    # Triggers a new lr, and new loss factors
    def __init__(
        self,
        epoch: int,
        new_lr: tp.Optional[float] = None,
        new_loss_terms_and_factors: tp.Optional[tp.Dict[str, float]] = None,
        verbose: bool = True,
    ):
        self._new_terms_and_factors = new_loss_terms_and_factors
        self._trigger_epoch = epoch
        self._new_lr = new_lr
        self._verbose = verbose

    def on_train_epoch_start(self, trainer, pl_module):
        prev_lrs = []
        new_lrs = []
        if trainer.current_epoch == self._trigger_epoch:
            if self._new_lr is not None:
                for optimizer in trainer.optimizers:
                    for param_group in optimizer.param_groups:
                        # Only update the lr if it is greater than the new forced lr
                        prev_lrs.append(param_group["lr"])
                        if param_group["lr"] > self._new_lr:
                            param_group["lr"] = self._new_lr
                        new_lrs.append(param_group["lr"])

                # Reset ReduceLROnPlateau internal counters
                for config in trainer.lr_scheduler_configs:
                    if config.reduce_on_plateau:
                        config.scheduler.cooldown_counter = 0
                        config.scheduler.num_bad_epochs = 0

            if self._new_terms_and_factors:
                trainer.lightning_module.set_loss_factors(self._new_terms_and_factors)

            if self._verbose:
                console.print("Starting new phase of training")
                if prev_lrs:
                    console.print(f"Final lrs of previous phase: {prev_lrs}")
                    console.print(f"Initial lrs of this phase: {new_lrs}")
                if self._new_terms_and_factors:
                    console.print(
                        f"New loss factors for phase: {self._new_terms_and_factors}"
                    )
