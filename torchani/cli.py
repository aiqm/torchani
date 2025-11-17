r"""Contains the TorchANI CLI entrypoints.

The actual implementation of the functions is considered internal. Please don't rely on
calling functions inside :mod:`torchani.cli` directly.
"""

import torch
import os
from enum import Enum
import shutil
import typing as tp
import json
import tarfile
import csv
from typer import Argument, Option, Typer, Abort
from pathlib import Path
import typing_extensions as tpx
import re

from ._builtin_dataset_ids import _DatasetId, _LotId


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
    sms: tpx.Annotated[
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
    paths: tpx.Annotated[
        tp.List[Path],
        Argument(),
    ],
    output_path: tpx.Annotated[
        tp.Optional[Path],
        Option("-o", "--output", show_default=False),
    ] = None,
    model_key: tpx.Annotated[
        str,
        Option("-m", "--model"),
    ] = "ANI2x",
    device: tpx.Annotated[
        tp.Optional[DeviceKind],
        Option("-d", "--device"),
    ] = None,
    dtype: tpx.Annotated[
        tp.Optional[DTypeKind],
        Option("-t", "--dtype"),
    ] = None,
    forces: tpx.Annotated[
        bool,
        Option("-f/-F", "--forces/--no-forces"),
    ] = False,
    hessians: tpx.Annotated[
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
    paths: tpx.Annotated[
        tp.List[Path],
        Argument(
            help="Paths to input files. Any format supported by ASE is accepted, such as .xyz or .pdb"  # noqa:E501
        ),
    ],
    output_path: tpx.Annotated[
        tp.Optional[Path],
        Option("-o", "--output", show_default=False),
    ] = None,
    model_key: tpx.Annotated[
        str,
        Option("-m", "--model"),
    ] = "ANI2x",
    device: tpx.Annotated[
        tp.Optional[DeviceKind],
        Option("-d", "--device"),
    ] = None,
    dtype: tpx.Annotated[
        tp.Optional[DTypeKind],
        Option("-t", "--dtype"),
    ] = None,
    atomic_charges: tpx.Annotated[
        bool,
        Option("-q/-Q", "--charges/--no-charges"),
    ] = False,
    forces: tpx.Annotated[
        bool,
        Option("-f/-F", "--forces/--no-forces"),
    ] = False,
    hessians: tpx.Annotated[
        bool,
        Option("-s/-S", "--hessians/--no-hessians"),
    ] = False,
) -> None:
    r"""Execute a single point calculation using a TorchANI model"""
    import torchani
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
    name: tpx.Annotated[_DatasetId, Argument()],
    lot: tpx.Annotated[
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
    check: tpx.Annotated[
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
    src_dir: tpx.Annotated[Path, Argument()],
    dest: tpx.Annotated[tp.Optional[Path], Option("-o")] = None,
    name: tpx.Annotated[str, Option("-n", "--name")] = "",
    lot: tpx.Annotated[str, Option("-l", "--lot")] = "",
    suffix: tpx.Annotated[
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
