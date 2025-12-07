import typing_extensions as tpx
import json
import itertools
import hashlib

import typing as tp
from dataclasses import dataclass, asdict, field
from pathlib import Path

from torchani.annotations import PyScalar
from torchani.train.paths import BATCH_PATH, FTUNE_PATH, TRAIN_PATH


def load_state_dict(path: Path) -> tp.Dict[str, tp.Any]:
    r"""
    Load a model's state dict either from a torch .pt file, or a lightning
    .ckpt file (it is assumed that the model is located in a .model attribute
    in the LightningModule in the latter case).
    """
    import torch

    _state_dict = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in _state_dict:
        _state_dict = _state_dict["state_dict"]
        return {
            k.replace("model.", ""): v
            for k, v in _state_dict.items()
            if k.startswith("model.")
        }
    return _state_dict


class ConfigError(RuntimeError):
    pass


@dataclass
class Config:

    def to_json_file(self, path: Path) -> None:
        with open(path, mode="wt", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4)

    def to_json_str(self) -> str:
        return json.dumps(asdict(self), indent=4)

    @classmethod
    def from_json_file(cls, path: Path) -> tpx.Self:
        if not path.is_file():
            raise ValueError(f"{path} is not a config file")
        with open(path, mode="rt", encoding="utf-8") as f:
            dict_ = json.load(f)
            # Needed for bw compat
            if "options" in dict_:
                dict_["options"].pop("use_cuda_ops", "")
        obj = cls()
        for k, v in dict_.copy().items():
            if isinstance(getattr(obj, k), Config):
                setattr(obj, k, getattr(obj, k).from_json_str(json.dumps(v)))
            elif k == "ftune" and v is not None:
                # Special case, since this can be None
                setattr(obj, k, FinetuneConfig.from_json_str(json.dumps(v)))
            else:
                setattr(obj, k, v)
        return obj

    @classmethod
    def from_json_str(cls, json_str: str) -> tpx.Self:
        dict_ = json.loads(json_str)
        # Needed for bw compat
        if "options" in dict_:
            dict_["options"].pop("use_cuda_ops", "")
        obj = cls()
        for k, v in dict_.copy().items():
            if isinstance(getattr(obj, k), Config):
                setattr(obj, k, getattr(obj, k).from_json(json.dumps(v)))
            elif k == "ftune" and v is not None:
                # Special case, since this can be None
                setattr(obj, k, FinetuneConfig.from_json_str(json.dumps(v)))
            else:
                setattr(obj, k, v)
        return obj


@dataclass
class FinetuneConfig(Config):
    r"""
    ftune-specific configurations
    """

    pretrained_name: str = ""
    raw_state_dict_path: str = ""
    num_head_layers: int = 1
    backbone_lr: float = 0.0
    dummy_ftune: bool = False

    @property
    def state_dict_path(self) -> tp.Optional[Path]:
        if not self.raw_state_dict_path:
            return None
        return Path(self.raw_state_dict_path).resolve()

    @property
    def frozen_backbone(self) -> bool:
        return self.backbone_lr == 0.0

    @property
    def pretrained_state_dict(self) -> tp.Dict[str, tp.Any]:
        if self.state_dict_path is None:
            return {}
        return load_state_dict(self.state_dict_path)


# Dataset parameters
@dataclass
class DatasetConfig(Config):
    r"""
    dataset-specific configurations
    """

    fold_idx: tp.Union[int, str] = "train"
    folds: tp.Optional[int] = None
    train_frac: float = 0.8
    validation_frac: float = 0.2
    properties: tp.List[str] = field(default_factory=list)
    data_names: tp.List[str] = field(default_factory=list)
    raw_src_paths: tp.List[str] = field(default_factory=list)
    lot: str = "wb97x-631gd"
    batch_size: int = 2560
    divs_seed: int = 1234
    batch_seed: int = 1234
    label: str = ""

    @property
    def src_paths(self) -> tp.Tuple[Path, ...]:
        return tuple(map(Path, self.raw_src_paths))

    @property
    def split_dict(self) -> tp.Dict[str, float]:
        return {
            "training": self.train_frac,
            "validation": self.validation_frac,
        }

    @property
    def split_spec(self) -> tp.Dict[str, tp.Union[int, tp.Dict[str, float]]]:
        # Get specification for folds or splits
        if self.folds is not None:
            if self.train_frac != 0.8 or self.validation_frac != 0.2:
                raise ConfigError("train|val frac can't be set if training to folds")
            if not isinstance(self.fold_idx, int):
                raise ConfigError("A fold idx must be present when training to folds")
            return {"folds": self.folds}
        return {"splits": self.split_dict}

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        return "_".join(
            itertools.chain(
                (
                    p.stem.replace("-", "_")
                    for p in sorted(map(Path, self.raw_src_paths))
                ),
                sorted(self.data_names),
            )
        )

    @property
    def path(self) -> Path:
        dict_ = asdict(self)
        dict_.pop("fold_idx")
        dict_.pop("label")
        state = sorted((k, v) for k, v in dict_.items())
        hasher = hashlib.shake_128()
        hasher.update(str(state).encode())
        _path = BATCH_PATH / f"{self.name}-{hasher.hexdigest(4)}"
        return _path


@dataclass
class FnConfig(Config):
    options: tp.Dict[str, PyScalar] = field(default_factory=dict)


@dataclass
class SrcConfig(Config):
    train_src: tp.List[str] = field(default_factory=list)
    ftune_src: tp.List[str] = field(default_factory=list)

    @property
    def num(self) -> int:
        return len(self.train_src) + len(self.ftune_src)


@dataclass
class ModelConfig(FnConfig):
    r"""
    model-specific configurations
    """

    lot: str = ""
    arch_fn: str = ""
    builtin: bool = False
    symbols: tp.List[str] = field(default_factory=list)
    options: tp.Dict[str, PyScalar] = field(default_factory=dict)

    @classmethod
    def from_builtin(cls, name_or_idx: str) -> tpx.Self:
        name, idx = name_or_idx.split(":")
        symbols = ["H", "C", "N", "O"]
        lot = {
            "ani1x": "wb97x-631gd",
            "ani2x": "wb97x-631gd",
            "aniala": "wb97x-631gd",
            "anidr": "b973c-def2mtzvp",
            "ani1ccx": "ccsd(t)star-cbs",
            "animbis": "wb97x-631gd",
        }
        if not (("1x" in name) or ("1ccx" in name)):
            symbols.extend(["S", "F", "Cl"])
        return cls(
            builtin=True,
            arch_fn=name,
            options={"model_index": int(idx)},
            symbols=symbols,
            lot=lot[name],
        )


@dataclass
class OptimizerConfig(FnConfig):
    r"""
    Optimizer configuration
    """

    cls: str = "AdamW"

    @property
    def lr(self) -> float:
        return tp.cast(float, self.options["lr"])

    @property
    def weight_decay(self) -> float:
        return tp.cast(float, self.options["weight_decay"])


@dataclass
class SchedulerConfig(FnConfig):
    r"""
    lr-Scheduler configuration
    """

    cls: str = "ReduceLROnPlateau"


@dataclass
class LossConfig(Config):
    r"""
    loss-specific configurations
    """

    terms_and_factors: tp.Dict[str, float] = field(
        default_factory=lambda: {"Energies": 1.0}
    )
    uncertainty_weighted: bool = False


@dataclass
class AccelConfig(Config):
    r"""
    Acceleration specific configuration. Does not affect reproducibility.
    """

    device: str = "gpu"
    num_workers: int = 2
    prefetch_factor: int = 2
    max_epochs: int = 200
    limit: tp.Optional[int] = None
    deterministic: bool = False
    detect_anomaly: bool = False
    profiler: tp.Optional[str] = None
    early_stop_patience: int = 50

    @property
    def log_interval(self) -> tp.Optional[int]:
        return max(1, self.limit) if self.limit is not None else 50

    @property
    def train_limit(self) -> tp.Optional[int]:
        return self.limit if self.limit is None else max(1, int(self.limit))

    @property
    def validation_limit(self) -> tp.Optional[int]:
        return self.limit if self.limit is None else max(1, int(self.limit / 10))


@dataclass
class TrainConfig(Config):
    r"""
    Configuration for all the training.
    """

    name: str = "run"
    monitor_label: str = "energies"
    ds: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    ftune: tp.Optional[FinetuneConfig] = None
    debug: bool = False
    accel: AccelConfig = field(default_factory=AccelConfig)

    @property
    def path(self) -> Path:
        dict_ = asdict(self)
        root = FTUNE_PATH if self.ftune is not None else TRAIN_PATH

        dict_.pop("debug")
        dict_.pop("accel")
        dict_.pop("name")
        keys = tuple(dict_.keys())
        for k in keys:
            if isinstance(dict_[k], dict):
                dict_[k] = sorted((k, v) for k, v in dict_[k].items())
        state = sorted((k, v) for k, v in dict_.items())
        hasher = hashlib.shake_128()
        hasher.update(str(state).encode())
        return root / f"{self.name}-{hasher.hexdigest(4)}"
