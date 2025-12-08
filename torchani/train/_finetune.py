from pathlib import Path

from typer import Abort
from rich.console import Console

from torchani.paths import select_subdirs, DataKind
from torchani.train.config import FinetuneConfig, TrainConfig, ModelConfig

console = Console()


def validate_ftune_options(backbone_lr: float, lr: float, num_head_layers: int) -> None:
    # Validation
    if backbone_lr < 0.0:
        console.print("backbone lr must be >= 0", style="red")
        raise Abort()
    if backbone_lr > lr:
        console.print("Backbone lr must be <= head lr", style="red")
        raise Abort()
    if num_head_layers < 1:
        console.print("There must be at least one head layer", style="red")
        raise Abort()


def setup_finetune_and_model_config(
    ftune_from: str,
    backbone_lr: float = 0.0,
    num_head_layers: int = 1,
    dummy_ftune: bool = False,
) -> tuple[FinetuneConfig, ModelConfig]:
    if ftune_from.split(":")[0] in (
        "ani1x",
        "ani2x",
        "ani2xr",
        "ani2dr",
        "ani1ccx",
        "anidr",
        "aniala",
    ):
        ptrain_name = ftune_from
        model_config = ModelConfig.from_builtin(ftune_from)
        raw_ptrain_state_dict_path = ""
    else:
        try:
            _path = select_subdirs((ftune_from,), kind=DataKind.TRAIN)[0]
        except Exception:
            _path = select_subdirs((ftune_from,), kind=DataKind.FTUNE)[0]

        ptrain_name = _path.name
        model_config = TrainConfig.from_json_file(_path / "config.json").model
        raw_ptrain_state_dict_path = str(Path(_path, "best-model", "best.ckpt"))

        if not Path(raw_ptrain_state_dict_path).is_file():
            console.print(
                f"{raw_ptrain_state_dict_path} is not a valid ckpt", style="red"
            )
            raise Abort()
    ftune_config = FinetuneConfig(
        pretrained_name=ptrain_name,
        raw_state_dict_path=raw_ptrain_state_dict_path,
        num_head_layers=num_head_layers,
        backbone_lr=backbone_lr,
        dummy_ftune=dummy_ftune,
    )
    return ftune_config, model_config
