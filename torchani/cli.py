r"""Contains the TorchANI CLI entrypoints.

The actual implementation of the functions is considered internal. Please don't rely on
calling functions inside :mod:`torchani.cli` directly.
"""

import sys
from copy import deepcopy
import warnings
import os
import uuid
from enum import Enum
import shutil
import typing as tp
from typing import Annotated
import json
import tarfile
import csv
import subprocess
from typer import Argument, Option, Typer, Abort
from pathlib import Path
import re

import jinja2
from rich.console import Console

from ._builtin_dataset_ids import _DatasetId, _LotId


console = Console()

REPO_BASE_URL = "https://github.com/roitberg-group/torchani_sandbox"

main = Typer(
    rich_markup_mode="markdown",
    help=r"""
    ## TorchANI

    A PyTorch library for training, development and research of
    ANI-style neural networks, maintained by the *Roitberg Group*.

    If you find this work useful please cite the following articles:
    - *TorchANI 2.0: An extensible, high performance library for the design, training, and use of NN-IPs*

        https://pubs.acs.org/doi/10.1021/acs.jcim.5c01853
    - *TorchANI: A Free and Open Source PyTorch-Based Deep Learning Implementation of the ANI Neural Network Potentials*

        https://pubs.acs.org/doi/10.1021/acs.jcim.0c00451

    To execute single point calculations run `ani sp <path-to-xyz-file> -m <model>`
    For example `ani sp methane.xyz -m ani2x`.

    To download a dataset run `ani data pull <dataset-name> --lot <lot>`, where `<lot>`
    is the level of theory.
    For example, `ani data pull ANI1x --lot wb97x-631gd`.
    To display available datasets `ani data ls`.
    To remove a downloaded dataset `ani data rm <dataset-name> --lot <lot>`.

    Datasets and Models are saved in ``$TORCHANI_DATA_DIR/Datasets`` and
    ``$TORCHANI_DATA_DIR/Models`` respectively. By default
    ``TORCHANI_DATA_DIR=~/.local/share/Torchani``.
    """,  # noqa
)

data_app = Typer()
main.add_typer(data_app, name="data", help="Manage TorchANI datasets")


class DTypeKind(Enum):
    F32 = "f32"
    F64 = "f64"


class DeviceKind(Enum):
    CUDA = "cuda"
    CPU = "cpu"


@main.command("build-extensions")
def _build_extensions(
    sms: Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "-s", "--sm", show_default=False, help="SMs to build for. (e.g. 8.9 10 12)"
        ),
    ] = None,
) -> None:
    r"""Build CUDA and C++ extensions"""
    import torch
    import torchani.paths
    from torch.utils.cpp_extension import load

    if sms is not None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ",".join(sms)

    nvcc_args = ["--expt-extended-lambda"]
    nvcc_args.extend(
        [
            "-DCUB_NS_QUALIFIER=::cuaev::cub",
            "-DCUB_NS_PREFIX='namespace cuaev {'",
            "-DCUB_NS_POSTFIX=}",
        ]
    )
    nvcc_args.extend(["-DTORCHANI_OPT", "-use_fast_math"])
    this_dir = str(Path(__file__).parent.resolve())
    include_dirs = torch.utils.cpp_extension.include_paths(device_type="cuda")
    include_dirs.append(f"{this_dir}/csrc/")
    if os.getenv("CONDA_PREFIX") and not os.getenv("CUDA_HOME"):
        # Help load() to detect cuda inside conda environments
        if Path(
            f"{os.environ['CONDA_PREFIX']}/targets/x86_64-linux/include/cuda_runtime_api.h"  # noqa:E501
        ).is_file():
            include_dirs.append(
                f"{os.environ['CONDA_PREFIX']}/targets/x86_64-linux/include"
            )
        os.environ["CUDA_HOME"] = f"{os.environ['CONDA_PREFIX']}/targets/x86_64-linux/"
    print("Building cuAEV extension...")
    build_dir = torchani.paths.data_dir().parent.parent / "lib" / "Torchani"
    build_dir.mkdir(exist_ok=True, parents=True)
    _ = load(
        name="cuaev",
        sources=[f"{this_dir}/csrc/cuaev.cpp", f"{this_dir}/csrc/aev.cu"],
        extra_include_paths=include_dirs,
        build_directory=str(build_dir),
        extra_cuda_cflags=nvcc_args,
        extra_cflags=["-std=c++17"],
        with_cuda=True,
        is_python_module=False,
    )
    print("Done!")
    print()
    print("Building MNP extension...")
    _ = load(
        name="mnp",
        sources=[f"{this_dir}/csrc/mnp.cpp"],
        extra_include_paths=include_dirs,
        build_directory=str(build_dir),
        extra_cflags=["-std=c++17", "-fopenmp"],
        with_cuda=True,
        is_python_module=False,
    )
    print("Done!")
    print()
    print("Building FastCellList extension (experimental)...")
    _ = load(
        name="cell_list",
        sources=[f"{this_dir}/csrc/cell_list.cpp"],
        extra_include_paths=include_dirs,
        build_directory=str(build_dir),
        extra_cflags=["-std=c++17"],
        with_cuda=True,
        is_python_module=False,
    )
    print("Done!")
    print()

    # Cleanup
    for f in build_dir.iterdir():
        if f.suffix != ".so":
            f.unlink()


@main.command(hidden=True)
def opt(
    paths: Annotated[
        tp.List[Path],
        Argument(),
    ],
    output_path: Annotated[
        tp.Optional[Path],
        Option("-o", "--output", show_default=False),
    ] = None,
    model_key: Annotated[
        str,
        Option("-m", "--model"),
    ] = "ANI2x",
    device: Annotated[
        tp.Optional[DeviceKind],
        Option("-d", "--device"),
    ] = None,
    dtype: Annotated[
        tp.Optional[DTypeKind],
        Option("-t", "--dtype"),
    ] = None,
    forces: Annotated[
        bool,
        Option("-f/-F", "--forces/--no-forces"),
    ] = False,
    hessians: Annotated[
        bool,
        Option("-s/-S", "--hessians/--no-hessians"),
    ] = False,
) -> None:
    r"""Execute a cartesian coords geom opt, using L-BFGS, with a TorchANI model"""
    raise NotImplementedError()
    import torchani
    from torchani.utils import _parse_device_and_dtype

    model_key = model_key.lower().replace("ani", "ANI")
    _device, _dtype = _parse_device_and_dtype(
        getattr(device, "value", None), getattr(dtype, "value", None)
    )
    model = getattr(torchani.models, model_key)(device=_device, dtype=_dtype)
    output: tp.Dict[str, tp.Any] = {"energies": []}
    if hessians:
        forces = True  # It is free to get the forces if you ask for the hessians
        output["hessians"] = []
    if forces:
        output["forces"] = []
    print("Sorry. Not implemented yet!")
    raise Abort()
    for p in paths:
        znums, coords, cell, pbc = torchani.io.read_xyz(p, device=_device, dtype=_dtype)
        for _znums, _coords in zip(znums, coords):
            unpadded = torchani.utils.strip_redundant_padding(
                {"species": _znums.unsqueeze(0), "coordinates": _coords.unsqueeze(0)}
            )
            _znums = unpadded["species"]
            _coords = unpadded["coordinates"]
            result = torchani.single_point(
                model, _znums, _coords, cell, pbc, forces=forces, hessians=hessians
            )
            # Optimization should be performed here
            output["energies"].extend(result["energies"].tolist())
            if forces:
                output["forces"].extend(result["forces"].tolist())
            if hessians:
                output["hessians"].extend(result["hessians"].tolist())
    if output_path is not None:
        output_path.write_text(json.dumps(output, indent=4))
    else:
        print(json.dumps(output))


@main.command()
def sp(
    paths: Annotated[
        tp.List[Path],
        Argument(
            help="Paths to input files. Any format supported by ASE is accepted, such as .xyz or .pdb"  # noqa:E501
        ),
    ],
    output_path: Annotated[
        tp.Optional[Path],
        Option("-o", "--output", show_default=False),
    ] = None,
    model_key: Annotated[
        str,
        Option("-m", "--model"),
    ] = "ANI2x",
    device: Annotated[
        tp.Optional[DeviceKind],
        Option("-d", "--device"),
    ] = None,
    dtype: Annotated[
        tp.Optional[DTypeKind],
        Option("-t", "--dtype"),
    ] = None,
    atomic_charges: Annotated[
        bool,
        Option("-q/-Q", "--charges/--no-charges"),
    ] = False,
    forces: Annotated[
        bool,
        Option("-f/-F", "--forces/--no-forces"),
    ] = False,
    hessians: Annotated[
        bool,
        Option("-s/-S", "--hessians/--no-hessians"),
    ] = False,
) -> None:
    r"""Execute a single point calculation using a TorchANI model"""
    import torchani
    import torch
    from ase import Atoms
    from ase.io import read as ase_read
    from torchani.utils import _parse_device_and_dtype

    model_key = model_key.lower().replace("ani", "ANI")
    _device, _dtype = _parse_device_and_dtype(
        getattr(device, "value", None), getattr(dtype, "value", None)
    )
    model = getattr(torchani.models, model_key)(device=_device, dtype=_dtype)
    output: tp.Dict[str, tp.Any] = {"energies": []}
    if hessians:
        forces = True  # It is free to get the forces if you ask for the hessians
        output["hessians"] = []
    if forces:
        output["forces"] = []
    if atomic_charges:
        output["atomic_charges"] = []
    for p in paths:
        if p.suffix == ".xyz":
            znums, coords, cell, pbc = torchani.io.read_xyz(
                p, device=_device, dtype=_dtype
            )
        else:
            # Single molecule supported only
            atoms = tp.cast(Atoms, ase_read(p))
            if isinstance(atoms, list):
                raise ValueError("Batch eval only supported for single molecules")
            coords = (
                torch.from_numpy(atoms.positions)
                .to(dtype=_dtype, device=_device)
                .unsqueeze(0)
            )
            cell = torch.from_numpy(atoms.get_cell()[:]).to(
                dtype=_dtype, device=_device
            )
            znums = torch.from_numpy(atoms.numbers).to(device=_device).unsqueeze(0)
            pbc = torch.tensor(atoms.pbc, dtype=torch.bool, device=_device)
            if not pbc.any():
                pbc = None
                cell = None
        result = torchani.single_point(
            model,
            znums,
            coords,
            cell,
            pbc,
            forces=forces,
            hessians=hessians,
            atomic_charges=atomic_charges,
        )
        output["energies"].extend(result["energies"].tolist())
        if forces:
            output["forces"].extend(result["forces"].tolist())
        if hessians:
            output["hessians"].extend(result["hessians"].tolist())
        if atomic_charges:
            output["atomic_charges"].extend(result["atomic_charges"].tolist())

    if output_path is not None:
        output_path.write_text(json.dumps(output, indent=4))
    else:
        print(json.dumps(output))


@data_app.command("batch")
def data_batch(
    src_ds: Annotated[
        tp.List[str],
        Argument(
            help="Built-in ANI ds name(s) to src from (format is 'name:lot')"
            " or paths to on-disk datasets",
        ),
    ],
    out_name: Annotated[
        str,
        Option("-n", "--out-name", help="Name of output batched dataset"),
    ] = "",
    out_lot: Annotated[
        str,
        Option(
            "-l",
            "--out-lot",
            help="LoT of the output batched dataset."
            " By default it is set to the lot of the builtin datasets."
            " If there is a mismatch, or if no built-in datasets are specified, "
            " it must be explicitly passed",
        ),
    ] = "",
    properties: Annotated[
        tp.Optional[tp.List[str]],
        Option("-p", "--property", help="Properties to batch. All by default"),
    ] = None,
    batch_size: Annotated[
        int,
        Option("--batch-size", help="Batch size"),
    ] = 2560,
    folds: Annotated[
        tp.Optional[int],
        Option("--folds", help="Num. of folds. Useful for training ensembles"),
    ] = None,
    train_frac: Annotated[
        float, Option("--tf", "--train-frac", help="Training set fraction")
    ] = 0.9,
    divs_seed: Annotated[
        int, Option("--divs-seed", help="Seed for divisions (train, validation, etc)")
    ] = 1234,
    batch_seed: Annotated[
        int,
        Option("--shuffle-seed", help="Seed for shuffling divisions before batching"),
    ] = 1234,
    allow_lot_mismatch: Annotated[
        bool,
        Option(
            "--allow-ds-lot-mismatch/ ",
            help="Allow built-in ds with different LoT",
            hidden=True,
        ),
    ] = False,
    max_batches_per_packet: Annotated[
        int,
        Option("--max-batches-per-packet"),
    ] = 100,
) -> None:
    r"""Generate a pre-batched dataset from one or more ANI datasets"""
    from torchani import datasets
    from torchani.train.config import DatasetConfig

    src_paths = []
    src_builtins = []
    for ds in src_ds:
        if Path(ds).exists():
            src_paths.append(ds)
        else:
            src_builtins.append(ds)

    # Make order-independent
    src_paths = sorted(src_paths)
    src_builtins = sorted(src_builtins)

    # Concatenate the source paths of all datasets into a list, since they will be
    # loaded as a *single* ANIDataset
    all_src_paths = deepcopy(src_paths)
    in_lots = set()
    for i, builtin in enumerate(src_builtins):
        if ":" in builtin:
            ds_name, ds_lot = builtin.split(":")
            ds = getattr(datasets, ds_name)(
                skip_check=True,
                lot=ds_lot,
            )
        else:
            ds_name = builtin
            ds = getattr(datasets, ds_name)(skip_check=True)
        in_lots.add(ds.lot)
        if not out_name and i == 0:
            out_name = f"{ds_name}:{ds.lot}"
        all_src_paths.extend(ds.store_locations)
    all_src_paths = sorted(set(all_src_paths))

    if len(in_lots) > 1:
        if not allow_lot_mismatch or not out_lot:
            console.print(
                "One or more of the specified built-in ds have different LoT"
                "If intended use --allow-lot-mismatch and --out-lot",
                style="red",
            )
            raise Abort()
    elif len(in_lots) == 0:
        if not out_lot:
            warnings.warn("Output LoT unspecified")
        out_lot = "unspecified"
    else:
        if out_lot:
            warnings.warn(
                "Specified custom output LoT different from LoT"
                f" present in datasets, which is {list(in_lots)[0]}"
            )
        else:
            out_lot = list(in_lots)[0]

    # TODO: Allow concatenating datasets with different properties
    ani_ds = datasets.ANIDataset(locations=all_src_paths)

    if properties is None:
        properties = sorted(ani_ds.tensor_properties)

    config = DatasetConfig(
        label=out_name,
        lot=out_lot,
        data_names=src_builtins,
        properties=sorted(properties),
        raw_src_paths=sorted(src_paths),
        batch_size=batch_size,
        fold_idx=-1,
        folds=folds,
        validation_frac=round(1.0 - train_frac, 5),
        train_frac=train_frac,
        batch_seed=batch_seed,
        divs_seed=divs_seed,
    )

    datasets.create_batched_dataset(
        src=ani_ds,
        max_batches_per_packet=max_batches_per_packet,
        dest_path=config.path,
        batch_size=config.batch_size,
        divs_seed=config.divs_seed,
        batch_seed=config.batch_seed,
        properties=set(config.properties) | {"species", "coordinates"},
        **config.split_spec,  # type: ignore
    )
    config.to_json_file(config.path / "ds_config.json")


@data_app.command("pull", help="Download one or more built-in datasets.")
def data_pull(
    names: Annotated[
        tp.Optional[tp.List[_DatasetId]],
        Argument(
            help="Dataset(s) to download. If unspecified all datasets are downloaded"
        ),
    ] = None,
    lots: Annotated[
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
    ds_dir: Annotated[
        tp.Optional[Path],
        Option(
            "-d",
            "--datasets-dir",
            show_default=False,
            help="Datasets are downloaded to <datasets-dir>/<dataset-name>",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        Option("-v/-V", "--verbose/--no-verbose"),
    ] = True,
    check: Annotated[
        bool,
        Option("-s/-S", "--check/--no-check"),
    ] = True,
) -> None:
    r"""
    Download a built-in dataset to the default location in disk, or to a
    custom location
    """
    import torchani.paths
    from torchani.datasets._utils import (
        _available_dataset_lots,
        _default_dataset_lot,
        _fetch_and_create_builtin_dataset,
    )

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

    root = ds_dir or torchani.paths.datasets_dir()
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
    import torchani.paths
    from torchani.datasets._utils import DatasetIntegrityError, _available_archives

    archives = _available_archives()
    deleted = 0
    for d in sorted(torchani.paths.datasets_dir().iterdir()):
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
    name: Annotated[_DatasetId, Argument()],
    lot: Annotated[
        tp.Optional[_LotId],
        Option("-l", "--lot"),
    ] = None,
) -> None:
    import torchani.paths
    from torchani.datasets._utils import _DATASETS_SPEC

    if lot is None:
        dirname = _DATASETS_SPEC[name.value]["default-lot"]["archive"].split(".")[0]
    else:
        dirname = _DATASETS_SPEC[name.value]["lot"][lot.value]["archive"].split(".")[0]
    ds_dir = torchani.paths.datasets_dir() / dirname
    if ds_dir.exists():
        print(f"Deleting dataset {dirname} ...")
        shutil.rmtree(ds_dir)
        print("Done!")
    else:
        print(f"Dataset {dirname} not found")


@data_app.command("ls", help="List downloaded built-in datasets")
def data_ls(
    check: Annotated[
        bool,
        Option("-s/-S", "--check/--no-check"),
    ] = False,
) -> None:
    import torchani
    import torchani.paths
    from torchani.datasets._utils import _available_archives, DatasetIntegrityError

    archives = _available_archives()
    for d in sorted(torchani.paths.datasets_dir().iterdir()):
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
            print(d.name)


@data_app.command("info", help="Display info regarding downloaded built-in datasets")
def data_info(
    name: Annotated[_DatasetId, Argument()],
    lot: Annotated[
        tp.Optional[_LotId],
        Option("-l", "--lot"),
    ] = None,
    check: Annotated[
        bool,
        Option("-s/-S", "--check/--no-check"),
    ] = True,
) -> None:
    import torchani

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
    "pack",
    help="Create .tar.gz, .yaml, and .json files from a dir with .h5 files",
    hidden=True,
)
def data_pack(
    src_dir: Annotated[Path, Argument()],
    dest: Annotated[tp.Optional[Path], Option("-o")] = None,
    name: Annotated[str, Option("-n", "--name")] = "",
    lot: Annotated[str, Option("-l", "--lot")] = "",
    suffix: Annotated[
        str,
        Option("-s", "--suffix"),
    ] = ".h5",
) -> None:
    from torchani.datasets._utils import _calc_file_md5

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


# HUGE training function
@main.command(help="Train from scratch or finetune an ANI-style model")
def train(
    batch_id: Annotated[str, Argument(help="Name|idx of the batched dataset")],
    fold_idx: Annotated[
        tp.Optional[int],
        Option(
            "-i",
            "--fold-idx",
            help="Idx to use if training from folds",
            show_default=False,
        ),
    ] = None,
    name: Annotated[str, Option("-n", "--run-name", help="Name of run")] = "run",
    slurm: Annotated[
        str,
        Option("--slurm"),
    ] = "",
    slurm_gpu: Annotated[
        str,
        Option("--slurm-gpu"),
    ] = "",
    num_workers: Annotated[
        int,
        Option("--num-workers", help="Num workers for Dataloader"),
    ] = 1,
    allow_lot_mismatch: Annotated[
        bool,
        Option(
            "--allow-ds-model-lot-mismatch/ ",
            help="Allow model lot to differ from ds lot. Useful for transfer learning.",
            hidden=True,
        ),
    ] = False,
    auto_restart: Annotated[
        bool,
        Option("--auto-restart/ ", help="Auto restart runs that match a prev run"),
    ] = False,
    max_epochs: Annotated[
        int, Option("--max-epochs", help="Max epochs to train")
    ] = 1000,
    early_stop_patience: Annotated[
        int,
        Option(
            "--early-stop-patience",
            help="Max epochs without improving monitor metric before early stopping",
        ),
    ] = 50,
    # From-scratch specific config
    symbols: Annotated[
        str,
        Option(
            "--symbols",
            help="Chemical symbols the model will support."
            " The default is 'all present in the dataset'."
            " If specified, it should be a single string"
            " with symbols separated by commas. e.g. '--symbols H,C,N,O,F,S'",
            show_default=False,
            rich_help_panel="Arch",
        ),
    ] = "",
    lot: Annotated[
        str,
        Option(
            "--lot",
            help="LoT of the model. Default is 'dataset lot'.",
            show_default=False,
            rich_help_panel="Arch",
        ),
    ] = "",
    device: Annotated[
        tp.Optional[DeviceKind],
        Option("-d", "--device", case_sensitive=False),
    ] = None,
    arch_fn: Annotated[
        str,
        Option(
            "-a",
            "--arch",
            help="Callable that creates the model",
            rich_help_panel="Arch",
        ),
    ] = "simple_ani",
    arch_options: Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "--ao",
            "--arch-opt",
            help="Options for arch fn, key=val fmt",
            rich_help_panel="Arch",
            show_default=False,
        ),
    ] = None,
    # LrSched config
    lrsched: Annotated[
        str,
        Option(
            "-s", "--sched", help="Type of lr-scheduler", rich_help_panel="LR scheduler"
        ),
    ] = "Plateau",
    lrsched_opts: Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "--so",
            "--sched-opt",
            help="Options for lr-scheduler, key=val fmt",
            rich_help_panel="LR scheduler",
            show_default=False,
        ),
    ] = None,
    # Optimizer config
    optim: Annotated[
        str,
        Option("-o", "--optim", help="Type of optimizer", rich_help_panel="Optimizer"),
    ] = "AdamW",
    optim_opts: Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "--oo",
            "--optim-opt",
            rich_help_panel="Optimizer",
            help="Options for optim, key=val fmt (lr, wd are separate)",
            show_default=False,
        ),
    ] = None,
    wd: Annotated[
        float,
        Option("--wd", help="Weight decay for optim", rich_help_panel="Optimizer"),
    ] = 1e-7,
    lr: Annotated[
        float,
        Option(
            "--lr",
            help="Initial lr. If ftune, used for the 'head'",
            rich_help_panel="Optimizer",
        ),
    ] = 5e-4,
    # Loss config
    xc: Annotated[
        bool, Option("--xc/ ", help="Train to XC energies", rich_help_panel="Loss")
    ] = False,
    no_sqrt_atoms: Annotated[
        bool,
        Option(
            "--no-sqrt-atoms/ ",
            help="Divide energy loss by atoms instead of sqrt(atoms)",
            rich_help_panel="Loss",
        ),
    ] = False,
    energies: Annotated[
        float, Option("-e", "--energies", help="Energy factor", rich_help_panel="Loss")
    ] = 1.0,
    forces: Annotated[
        float, Option("-f", "--forces", help="Force factor", rich_help_panel="Loss")
    ] = 0.0,
    dipoles: Annotated[
        float, Option("-m", "--dipoles", help="Dipole factor", rich_help_panel="Loss")
    ] = 0.0,
    atomic_volumes: Annotated[
        float,
        Option(
            "-V",
            "--atomic-volumes",
            help="Atomic volumes factor",
            rich_help_panel="Loss",
        ),
    ] = 0.0,
    atomic_charges: Annotated[
        float,
        Option(
            "-q",
            "--atomic-charges",
            help="Atomic charges factor",
            rich_help_panel="Loss",
        ),
    ] = 0.0,
    total_charge: Annotated[
        float,
        Option("--total-q", help="Total charge factor", rich_help_panel="Loss"),
    ] = 0.0,
    monitor: Annotated[
        str,
        Option(
            "--monitor",
            help="Loss label to monitor during training."
            " Format is 'valid/rmse_energies', 'train/rmse_forces', etc."
            " If a single loss term is present, it is the valid/rmse_'loss-term'."
            " Otherwise, if 'forces' is a loss term, it is valid/rmse_forces."
            " Otherwise it must be explicitly specified.",
            rich_help_panel="Loss",
            show_default=False,
        ),
    ] = "valid/rmse_default",
    # Finetuning specific config
    dummy_ftune: Annotated[
        bool,
        Option(
            "--dummy-ftune/--no-dummy-ftune", rich_help_panel="Finetuning", hidden=True
        ),
    ] = False,
    ftune_from: Annotated[
        str,
        Option(
            "--ftune-from",
            help="Name|idx of pretrain run. ani1x:idx, ... also supported",
            rich_help_panel="Finetuning",
            show_default=False,
        ),
    ] = "",
    num_head_layers: Annotated[
        tp.Optional[int],
        Option(
            "--num-head",
            help="If fine-tuning, num. of head layers. Defaults to 1",
            rich_help_panel="Finetuning",
            show_default=False,
        ),
    ] = None,
    backbone_lr: Annotated[
        tp.Optional[float],
        Option(
            "--backbone-lr",
            help="If fine-tuning, lr for backbone. Defaults to 0",
            rich_help_panel="Finetuning",
            show_default=False,
        ),
    ] = None,
    # Debug and profiling specific config
    debug: Annotated[
        bool,
        Option(
            "-g/ ", "--debug/ ", help="Run in debug config", rich_help_panel="Debug"
        ),
    ] = False,
    profiler: Annotated[
        tp.Optional[str],
        Option(
            "--prof",
            help="Profiler, 'simple', 'advanced', or 'pytorch'",
            rich_help_panel="Debug",
            show_default=False,
        ),
    ] = None,
    limit: Annotated[
        tp.Optional[int],
        Option(
            "--lim",
            help="Limit num batches or percent",
            rich_help_panel="Debug",
            show_default=False,
        ),
    ] = None,
    deterministic: Annotated[
        bool,
        Option(
            "--deterministic/ ",
            help="Deterministic training",
            rich_help_panel="Debug",
        ),
    ] = False,
    detect_anomaly: Annotated[
        bool,
        Option(
            "--detect-anomaly/ ",
            help="Detect anomalies during training",
            rich_help_panel="Debug",
        ),
    ] = False,
    verbose: Annotated[
        bool, Option("-v/ ", "--verbose/ ", rich_help_panel="Debug")
    ] = False,
    log_wandb: Annotated[
        bool,
        Option("--wandb/--no-wandb", rich_help_panel="Wandb"),
    ] = False,
    wandb_entity: Annotated[
        str,
        Option("--wandb-entity", rich_help_panel="Wandb"),
    ] = "nnip",
    wandb_project: Annotated[
        str,
        Option("--wandb-project", rich_help_panel="Wandb"),
    ] = "ani",
) -> None:

    import torch
    from torchani.paths import DataKind, select_subdirs
    from torchani.train._lit_training import train_lit_model
    from torchani.train.config import (
        FinetuneConfig,
        TrainConfig,
        DatasetConfig,
        AccelConfig,
        ModelConfig,
        LossConfig,
        OptimizerConfig,
        SchedulerConfig,
    )
    from torchani.train.defaults import (
        resolve_options,
        parse_scheduler_str,
        parse_optimizer_str,
    )

    if device is None:
        device = DeviceKind.CUDA if torch.cuda.is_available() else DeviceKind.CPU

    batched_dataset_path = select_subdirs((batch_id,), kind=DataKind.BATCH)[0]
    ds_config_path = batched_dataset_path / "ds_config.json"
    ds_config = DatasetConfig.from_json_file(ds_config_path)
    ds_config.fold_idx = "train" if fold_idx is None else fold_idx

    if fold_idx is not None:
        if not name:
            name = "train" if not ftune_from else "ftune"
        name = f"{str(fold_idx).zfill(2)}-{name}"

    with open(ds_config.path / "creation_log.json", mode="rt") as f:
        ds_symbols = json.load(f)["symbols"]

    if debug:
        console.print("Debugging enabled:")
        if name == "train":
            _uuid = uuid.uuid4().hex[:8]
            console.print(f"    - Name set to 'debug-{_uuid}'")
            name = f"debug-{_uuid}"
        if max_epochs == 1000:
            max_epochs = 3
            console.print(f"    - Max epochs set to {max_epochs}")
        if limit is None:
            limit = 3
            console.print(f"    - Batch limit set to {limit}")
        console.print("    - Verbosity increased")
        verbose = True
        console.print("    - Deterministic mode set")
        deterministic = True
        console.print("    - Anomaly detection mode set")
        detect_anomaly = True

    terms_and_factors: tp.Dict[str, float] = {}
    if energies > 0.0:
        label = "EnergiesXC" if xc else "Energies"
        terms_and_factors[label if no_sqrt_atoms else f"{label}SqrtAtoms"] = energies
    if forces > 0.0:
        terms_and_factors["Forces"] = forces
    if dipoles > 0.0:
        terms_and_factors["Dipoles"] = dipoles
    if atomic_charges > 0.0:
        terms_and_factors["AtomicCharges"] = atomic_charges
    if atomic_volumes > 0.0:
        terms_and_factors["AtomicVolumes"] = atomic_volumes
    if total_charge > 0.0:
        terms_and_factors["TotalCharge"] = total_charge

    lrsched = parse_scheduler_str(lrsched)
    optim = parse_optimizer_str(optim)
    lrsched_opts = lrsched_opts or []
    optim_opts = (optim_opts or []) + [f"lr={lr}", f"weight_decay={wd}"]

    if lr <= 0.0:
        console.print("lr must be strictly positive", style="red")
        raise Abort()

    # Finetune config
    if ftune_from:
        if arch_fn != "simple_ani" or arch_options or symbols:
            console.print(
                "Don't specify 'arch', 'arch-opts' or 'symbols' for ftune", style="red"
            )
            raise Abort()
        backbone_lr = backbone_lr or 0.0
        num_head_layers = num_head_layers or 1
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
        # Create finetune and model configs
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
    else:
        ftune_config = None
        model_config = ModelConfig(
            lot=lot or ds_config.lot,
            symbols=symbols.split(",") if symbols else ds_symbols,
            arch_fn=arch_fn,
            options=resolve_options(arch_options or (), arch_fn),
        )
    if not allow_lot_mismatch and model_config.lot != ds_config.lot:
        console.print(
            "Model LoT must match dataset LoT unless --allow-ds-model-lot-mismatch",
            style="red",
        )
        raise Abort()

    if not set(ds_symbols).issubset(model_config.symbols):
        console.print(
            f"Not all ds symbols {ds_symbols} are supported by the model."
            f"Model supports {model_config.symbols}",
            style="red",
        )
        raise Abort()

    config = TrainConfig(
        name=name,
        debug=debug,
        ds=ds_config,
        monitor_label=monitor,
        ftune=ftune_config,
        model=model_config,
        loss=LossConfig(terms_and_factors=terms_and_factors),
        optim=OptimizerConfig(resolve_options(optim_opts, optim), optim),
        scheduler=SchedulerConfig(resolve_options(lrsched_opts, lrsched), lrsched),
        accel=AccelConfig(
            device=device.value,
            limit=limit,
            deterministic=deterministic,
            detect_anomaly=detect_anomaly,
            max_epochs=max_epochs,
            early_stop_patience=early_stop_patience,
            profiler=profiler,
            num_workers=num_workers,
        ),
    )

    # Re-run everything after the train config has been set up, to prevent potential
    # issues
    if slurm:
        if slurm == "moria":
            assert slurm_gpu in ["v100", "gp100", "titanv", "gtx1080ti", ""]
        elif slurm == "hpg":
            assert slurm_gpu in ["b200", "l4", ""]
        else:
            console.print(f"Unknown cluster {slurm}", style="red")
            raise Abort()
        slurm_gpu = f"{slurm_gpu}:1" if slurm_gpu else "1"

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates/"),
            undefined=jinja2.StrictUndefined,
            autoescape=jinja2.select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        arg_list = sys.argv[1:]
        for j, arg in enumerate(deepcopy(arg_list)):
            # re-introduce quotes in strings
            if arg in ["--prof", "--ftune-from", "--monitor", "--lot"]:
                arg_list[j + 1] = f"'{arg_list[j + 1]}'"
            if arg == "--slurm":
                arg_list[j] = ""
                arg_list[j + 1] = ""
            if arg == "--slurm-gpu":
                arg_list[j] = ""
                arg_list[j + 1] = ""
        args = " ".join(arg_list)
        tmpl = env.get_template(f"{slurm}.slurm.sh.jinja").render(
            num_workers=num_workers,
            name=str(config.path.name),
            gpu=slurm_gpu,
            args=args,
        )
        unique_id = config.path.name.split("-")[-1]
        j = 0
        input_dir = Path(Path.home(), "IO", "ani", f"{unique_id}_v{j}")
        while input_dir.is_dir():
            j += 1
            input_dir = Path(Path.home(), "IO", "ani", f"{unique_id}_v{j}")
        input_dir.mkdir(exist_ok=False, parents=True)
        input_fpath = input_dir / f"{slurm}.slurm.sh"
        input_fpath.write_text(tmpl)
        console.print("Launching slurm script ...")
        subprocess.run(["sbatch", str(input_fpath)], cwd=input_dir, check=True)
        sys.exit(0)
    train_lit_model(
        config,
        allow_restart=auto_restart,
        verbose=verbose,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        log_wandb=log_wandb,
    )
