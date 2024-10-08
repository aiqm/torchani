import typing as tp
from typer import Argument
from pathlib import Path
import typing_extensions as tpx
from typer import Option, Typer

from torchani.datasets import (
    DatasetId,
    LotId,
    datapull as _datapull,
    datainfo as _datainfo,
    datapack as _datapack,
)

REPO_BASE_URL = "https://github.com/roitberg-group/torchani_sandbox"

main = Typer(
    rich_markup_mode="markdown",
    help=r"""
    ## TorchANI

    A PyTorch library for training, development and research of
    ANI-style neural networks, maintained by the *Roitberg Group*.

    Datasets and Models are saved in `$TORCHANI_DATA_DIR/Datasets` and
    `$TORCHANI_DATA_DIR/StateDicts` respectively. By default
    `TORCHANI_DATA_DIR=~/.local/share/torchani`.
    """,
)


# Data manipulation utilites
@main.command(help="Download a built-in dataset")
def datapull(
    name: tpx.Annotated[DatasetId, Argument()],
    lot: tpx.Annotated[
        tp.Optional[LotId],
        Option("-l", "--lot"),
    ] = None,
    skip_check: tpx.Annotated[
        bool,
        Option("-s/-S", "--skip-check/--no-skip-check"),
    ] = False,
) -> None:
    _datapull(name, lot=lot, verbose=True, skip_check=skip_check)


@main.command(help="Display info regarding built-in datasets")
def datainfo(
    name: tpx.Annotated[DatasetId, Argument()],
    lot: tpx.Annotated[
        tp.Optional[LotId],
        Option("-l", "--lot"),
    ] = None,
    skip_check: tpx.Annotated[
        bool,
        Option("-s/-S", "--skip-check/--no-skip-check"),
    ] = False,
) -> None:
    _datainfo(name, lot=lot, skip_check=skip_check)


@main.command(help="Create .tar.gz, .yaml, and .json files from a dir with .h5 files")
def datapack(
    path: tpx.Annotated[Path, Argument()],
    dest: tpx.Annotated[tp.Optional[Path], Option("-o")] = None,
    name: tpx.Annotated[str, Option("-n", "--name")] = "",
    lot: tpx.Annotated[str, Option("-l", "--lot")] = "",
) -> None:
    dest_dir = dest if dest is not None else Path.cwd()
    _datapack(src_dir=path, dest_dir=dest_dir, name=name, lot=lot)
