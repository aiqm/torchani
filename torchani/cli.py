import typing as tp
import json
import tarfile
import csv
from typer import Argument
from pathlib import Path
import typing_extensions as tpx
from typer import Option, Typer
import re

from torchani.paths import datasets_dir
import torchani.datasets
from torchani.datasets.utils import _calc_file_md5
from torchani.datasets.builtin import DatasetId, LotId

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


@main.command(help="Download a built-in dataset")
def datapull(
    name: tpx.Annotated[DatasetId, Argument()],
    lot: tpx.Annotated[
        tp.Optional[LotId],
        Option("-l", "--lot"),
    ] = None,
    verbose: tpx.Annotated[bool, Option("-v/-V", "--verbose/--no-verbose"),] = True,
    skip_check: tpx.Annotated[
        bool,
        Option("-s/-S", "--skip-check/--no-skip-check"),
    ] = False,
) -> None:
    r"""Download a built-in dataset to the default location in disk"""
    location = (datasets_dir() / f"{name.value}-{lot}").resolve()
    if location.exists() and verbose:
        if skip_check:
            print("Dataset found locally, skipping integrity check")
            return
        print("Dataset found locally, starting files integrity check ...")
    else:
        print("Dataset not found locally, starting download...")

    getter = getattr(torchani.datasets, name.value)
    if lot is None:
        getter(download=True)
        return
    getter(
        download=True, lot=lot.value, skip_check=skip_check
    )


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
    getter = getattr(torchani.datasets, name.value)
    if lot is None:
        ds = getter(
            download=False, skip_check=skip_check
        )
    else:
        ds = getter(
            download=False, lot=lot, skip_check=skip_check
        )
    groups = list(ds.keys())
    conformer = ds.get_numpy_conformers(groups[0], 0)
    key_max_len = max([len(k) for k in conformer.keys()]) + 3
    shapes = [str(list(conformer[k].shape)) for k in conformer.keys()]
    shape_max_len = max([len(s) for s in shapes]) + 3
    print("\nFirst Conformer Properties (non-batched): ")
    for i, k in enumerate(conformer.keys()):
        key = k.ljust(key_max_len)
        shape = shapes[i].ljust(shape_max_len)
        dtype = conformer[k].dtype
        print(f"  {key} shape: {shape} dtype: {dtype}")


@main.command(help="Create .tar.gz, .yaml, and .json files from a dir with .h5 files")
def datapack(
    src_dir: tpx.Annotated[Path, Argument()],
    dest: tpx.Annotated[tp.Optional[Path], Option("-o")] = None,
    name: tpx.Annotated[str, Option("-n", "--name")] = "",
    lot: tpx.Annotated[str, Option("-l", "--lot")] = "",
    suffix: tpx.Annotated[str, Option("-s", "--suffix"),] = ".h5",
) -> None:
    dest_dir = dest if dest is not None else Path.cwd()

    def _validate_label(label: str, label_name: str, lower: bool = False) -> str:
        while not re.match(r"[0-9A-Za-z_]+", label):
            print(f"{label} invalid for {label_name}, it should match r'[0-9A-Za-z_]+'")
            label = input(f"Input {label_name}: ")
        if lower:
            return label.lower()
        return label

    files = sorted(src_dir.glob(f"*{suffix}"))

    print(
        "Packaging ANI Dataset\n"
        "When prompted write the requested names\n"
        "**Only alphanumeric characters or '_' are supported**"
    )
    method, basis = lot.split("-")
    name = _validate_label(name, label_name="data")
    # lot is case insensitive
    method = _validate_label(method, label_name="method", lower=True)
    basis = _validate_label(basis, label_name="basis", lower=True)

    archive_path = dest_dir / f"{'-'.join((name, method, basis))}.tar.gz"
    csv_path = dest_dir / f"{name}.md5s.csv"
    json_path = dest_dir / f"{name}.json"

    data_dict: tp.Dict[str, tp.Any] = {
        name: {
            "lot": {
                lot: {
                    "archive": archive_path.name,
                    "files": [],
                }
            },
            "default-lot": lot,
        },
    }

    # Write csv and tarfile
    with tarfile.open(archive_path, "w:gz") as archive:
        with open(csv_path, "w", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["filename", "md5_hash"])

            for f in files:
                part = input(f"Specific label for file {f.name}?: ")
                part = _validate_label(part, label_name="file-specific label")

                stem = "-".join((name, part, method, basis))
                arcname = f"{stem}{f.suffix}"
                archive.add(f, arcname=arcname)
                md5 = _calc_file_md5(f)
                data_dict[name]["lot"][lot]["files"].append(arcname)
                writer.writerow([arcname, md5])

    # Write json
    with open(json_path, "wt", encoding="utf-8") as fj:
        json.dump(data_dict, fj)
