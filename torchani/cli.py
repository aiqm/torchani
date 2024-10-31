r"""Contains the TorchANI CLI entrypoints.

The actual implementation of the functions is considered internal. Please don't rely on
calling functions inside :mod:`torchani.cli` directly.
"""

import shutil
import typer
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
from torchani.datasets._utils import (
    DatasetIntegrityError,
    _DATASETS_SPEC,
    _calc_file_md5,
    _fetch_and_create_builtin_dataset,
    _available_dataset_lots,
    _available_archives,
    _default_dataset_lot,
)
from torchani.datasets.builtin import _DatasetId, _LotId


REPO_BASE_URL = "https://github.com/roitberg-group/torchani_sandbox"

main = Typer(
    rich_markup_mode="markdown",
    help=r"""
    ## TorchANI

    A PyTorch library for training, development and research of
    ANI-style neural networks, maintained by the *Roitberg Group*.

    Datasets and Models are saved in ``$TORCHANI_DATA_DIR/Datasets`` and
    ``$TORCHANI_DATA_DIR/StateDicts`` respectively. By default
    ``TORCHANI_DATA_DIR=~/.local/share/Torchani``.
    """,
)

data_app = typer.Typer()
main.add_typer(data_app, name="data", help="Manage TorchANI datasets")


@data_app.command("pull", help="Download one or more built-in datasets.")
def data_pull(
    names: tpx.Annotated[
        tp.Optional[tp.List[_DatasetId]],
        Argument(
            help="Dataset(s) to download. If unspecified all datasets are downloaded"
        ),
    ] = None,
    lots: tpx.Annotated[
        tp.Optional[tp.List[_LotId]],
        Option(
            "-l",
            "--lot",
            help="LoT for the specified dataset(s)."
            "'default' (a default dataset-dependent LoT)"
            " and 'all' (all available LoT for the dataset) are also supported options."
            " Note that not all datasets support all LoT. To check which LoT"
            " are available for a given dataset run ``ani data info <dataset-name>``",
        ),
    ] = None,
    ds_dir: tpx.Annotated[
        tp.Optional[Path],
        Option(
            "-d",
            "--datasets-dir",
            show_default=False,
            help="Datasets are downloaded to <datasets-dir>/<dataset-name>",
        ),
    ] = None,
    verbose: tpx.Annotated[
        bool,
        Option("-v/-V", "--verbose/--no-verbose"),
    ] = True,
    check: tpx.Annotated[
        bool,
        Option("-s/-S", "--check/--no-check"),
    ] = True,
) -> None:
    r"""
    Download a built-in dataset to the default location in disk, or to a
    custom location
    """
    names = names or list(_DatasetId)
    lots = lots or [_LotId.DEFAULT]

    if len(lots) == 1:
        lots = lots * len(names)

    if len(lots) != len(names):
        raise ValueError(
            "Incorrect --lot specification"
            " When downloading more than one dataset, possible options for --lot are:"
            " - Unspecified (selects a default LoT depending on the dataset)"
            " - Specified a single time (applies to all datasets)"
            " - One LoT specified per dataset (order is the same as dataset order)"
        )

    processed_lots = []
    processed_names = []
    for name, lot in zip(names, lots):
        if lot is _LotId.ALL:
            all_lots = [_LotId(_lot) for _lot in _available_dataset_lots(name.value)]
            processed_lots.extend(all_lots)
            processed_names.extend([name] * len(all_lots))
        else:
            if lot is _LotId.DEFAULT:
                lot = _LotId(_default_dataset_lot(name.value))
            processed_lots.append(lot)
            processed_names.append(name)

    root = ds_dir or datasets_dir()
    for name, lot in zip(processed_names, processed_lots):
        dest_dir = (root / f"{name.value}-{lot.value}").resolve()
        if dest_dir.exists() and verbose:
            if not check:
                print("Dataset found locally, skipping integrity check")
                continue
            print("Dataset found locally, running integrity check...")
        else:
            print("Dataset not found locally, downloading...")

        _fetch_and_create_builtin_dataset(
            ds_name=name.value,
            root=root,
            download=True,
            lot=lot.value,
            skip_check=not check,
        )


@data_app.command("clean", help="Remove datasets with data integrity issues")
def data_clean() -> None:
    archives = _available_archives()
    deleted = 0
    for d in sorted(datasets_dir().iterdir()):
        if d.name not in archives:
            continue
        name, lot = archives[d.name]
        try:
            getattr(torchani.datasets, name)(lot=lot, download=False, verbose=False)
        except DatasetIntegrityError:
            data_rm(_DatasetId(name), _LotId(lot))
    if deleted == 0:
        print("No integrity issues found, no datasets deleted")


@data_app.command("rm", help="Remove a downloaded dataset")
def data_rm(
    name: tpx.Annotated[_DatasetId, Argument()],
    lot: tpx.Annotated[
        tp.Optional[_LotId],
        Option("-l", "--lot"),
    ] = None,
) -> None:
    if lot is None:
        dirname = _DATASETS_SPEC[name.value]["default-lot"]["archive"].split(".")[0]
    else:
        dirname = _DATASETS_SPEC[name.value]["lot"][lot.value]["archive"].split(".")[0]
    ds_dir = datasets_dir() / dirname
    if ds_dir.exists():
        print(f"Deleting dataset {dirname} ...")
        shutil.rmtree(ds_dir)
        print("Done!")
    else:
        print(f"Dataset {dirname} not found")


@data_app.command("ls", help="List downloaded built-in datasets")
def data_ls(
    check: tpx.Annotated[
        bool,
        Option("-s/-S", "--check/--no-check"),
    ] = False,
) -> None:
    archives = _available_archives()
    for d in sorted(datasets_dir().iterdir()):
        if d.name not in archives:
            continue
        name, lot = archives[d.name]
        if check:
            try:
                getattr(torchani.datasets, name)(lot=lot, download=False, verbose=False)
                print(f"{d.name}, status: OK")
            except DatasetIntegrityError:
                print(f"{d.name}, status: Error!")
        else:
            print(f"{d.name}, status: Not checked")


@data_app.command("info", help="Display info regarding downloaded built-in datasets")
def data_info(
    name: tpx.Annotated[_DatasetId, Argument()],
    lot: tpx.Annotated[
        tp.Optional[_LotId],
        Option("-l", "--lot"),
    ] = None,
    check: tpx.Annotated[
        bool,
        Option("-s/-S", "--check/--no-check"),
    ] = True,
) -> None:
    getter = getattr(torchani.datasets, name.value)
    if lot is None:
        ds = getter(download=False, skip_check=not check)
    else:
        ds = getter(download=False, lot=lot.value, skip_check=not check)
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


@data_app.command(
    "pack", help="Create .tar.gz, .yaml, and .json files from a dir with .h5 files"
)
def data_pack(
    src_dir: tpx.Annotated[Path, Argument()],
    dest: tpx.Annotated[tp.Optional[Path], Option("-o")] = None,
    name: tpx.Annotated[str, Option("-n", "--name")] = "",
    lot: tpx.Annotated[str, Option("-l", "--lot")] = "",
    suffix: tpx.Annotated[
        str,
        Option("-s", "--suffix"),
    ] = ".h5",
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
