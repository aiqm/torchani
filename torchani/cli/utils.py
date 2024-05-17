import tarfile
import hashlib
import re
import csv
import typing as tp
import json
from pathlib import Path

from torchani.datasets import ANIDataset
from torchani.datasets.builtin import _DATASETS_JSON_PATH
from torchani.datasets.download import _CHUNK_SIZE


def h5info(path: tp.Union[str, Path]) -> None:
    path_ = Path(path)
    assert path_.exists(), f"{str(path_)} does not exist"
    if path_.is_file():
        files = [path_]
    else:
        files = list(Path(path_).glob("*.h5"))
        assert len(files) > 0, f"no h5 files found at {path_}"
    names = [f.stem for f in files]
    ds = ANIDataset(locations=files, names=names)
    print(ds)
    groups = list(ds.keys())
    conformer = ds.get_numpy_conformers(groups[0], 0)
    key_max_len = max([len(k) for k in conformer.keys()]) + 3
    shapes = [str(list(conformer[k].shape)) for k in conformer.keys()]
    shape_max_len = max([len(s) for s in shapes]) + 3
    print('\nFirst Conformer Properties (Non-batched): ')
    for i, k in enumerate(conformer.keys()):
        key = k.ljust(key_max_len)
        shape = shapes[i].ljust(shape_max_len)
        dtype = conformer[k].dtype
        print(f'  {key} shape: {shape} dtype: {dtype}')


def h5pack(
    paths: tp.Union[Path, tp.Sequence[Path]],
    internal: bool = False,
    dest_dir: Path = Path.cwd(),
    name: str = "",
    functional: str = "",
    basis_set: str = "",
    interactive: bool = True,
    force_renaming: bool = True,
) -> None:
    r"""
    If passed a directory with h5 files, generates a corresponding
    dataset.json, md5.csv, and an archive dataset.tar.gz for the given dataset.
    internal: bool
        Controls whether to append to the internal files the dataset has, or to
        create external files. Default is to create external files.
    name: str
        name of the dataset
    functional: str
        Functional for the dataset
    basis_set: str
        Basis set for the dataset
    force_renaming: bool
        Forces renaming all the files inside the archive with "data_part_name"
    interactive: bool
        Pass all arguments from interactive prompts, useful for cli
    paths: Path | Sequence[Path]
        Path to a directory with .h5 files, or sequence of paths to .h5 files
    dest_dir: Path
        Destination directory for the archive, md5.csv and dataset.json
    """

    if force_renaming and not interactive:
        raise ValueError("Force renaming only supported for interactive calls")
    if isinstance(paths, (Path, str)):
        paths = Path(paths)
        file_paths = sorted(paths.rglob("*.h5"))
    else:
        file_paths = sorted(Path(p) for p in paths)
    if interactive:
        if internal:
            print("WARNING: Modifying internal repository files")
        print(
            "Packaging ANI Dataset\n"
            "When prompted write the requested names\n"
            "**Only alphanumeric characters or '_' are supported**"
        )
    parts = {"Data": name, "Functional": functional, "Basis-set": basis_set}
    for kind in parts.copy().keys():
        while not re.match(r"\w+$", parts[kind]):
            if interactive:
                if parts[kind]:
                    print(f"Invalid name {parts[kind]}")
                    print("**Only alphanumeric characters or '_' are supported**")
                parts[kind] = input(
                    f"{kind} name?: "
                )
            else:
                raise ValueError(
                    f"Invalid name {parts[kind]}."
                    " Dataset name should only use alphanumeric characters or _"
                )
    # lot is case-insensitive
    parts["Functional"] = parts["Functional"].lower()
    parts["Basis-set"] = parts["Basis-set"].lower()

    archive_stem = "-".join(
        (
            parts["Data"],
            parts["Functional"],
            parts["Basis-set"],
        )
    )
    ds_name = parts["Data"]
    archive_name = f"{archive_stem}.tar.gz"
    lot = f'{parts["Functional"]}-{parts["Basis-set"]}'

    data_dict: tp.Dict[str, tp.Any] = {
        ds_name: {
            "lot": {
                lot: {
                    "archive": archive_name,
                    "files": [],
                }
            },
            "default-lot": lot,
        },
    }
    if internal:
        csv_path = (Path(__file__).parent.parent / "datasets") / "md5s.csv"
    else:
        csv_path = dest_dir / f"{ds_name}.md5s.csv"
        if csv_path.exists():
            raise ValueError(
                f"CSV path {str(csv_path)} should not exist"
            )
        csv_path.touch()

    registered_md5s: tp.Set[tp.Tuple[str, str]] = set()
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")

        for row in reader:
            if row[0] == "filename":
                continue
            registered_md5s.add((row[0], row[1]))

    with tarfile.open(dest_dir / archive_name, "w:gz") as archive:
        with open(csv_path, "a", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            if not internal:
                writer.writerow(["filename", "md5_hash"])

            for f in file_paths:
                if force_renaming:
                    data_part_name = input(
                        f"Data part name for file {f.name}?: "
                    )
                else:
                    data_part_name = f.stem
                while not re.match(r"\w+$", data_part_name):
                    if interactive:
                        if data_part_name:
                            print(f"Invalid name {data_part_name}")
                            print("Only [a-zA-Z0-9_] chars are supported")

                        data_part_name = input(
                            f"Data part name for file {f.name}?: "
                        )
                    else:
                        raise ValueError(
                            "File names should only use alphanumeric characters or _"
                        )
                if force_renaming:
                    main_part_name = input(
                        f"Main part name for file {f.name}?: "
                    )
                else:
                    main_part_name = ds_name
                while not re.match(r"\w+$", main_part_name):
                    if interactive:
                        if data_part_name:
                            print(f"Invalid name {main_part_name}")
                            print("Only [a-zA-Z0-9_] chars are supported")

                        data_part_name = input(
                            f"Main part name for file {f.name}?: "
                        )
                    else:
                        raise ValueError(
                            "File names should only use alphanumeric characters or _"
                        )
                    main_part_name = input(
                        f"Main part name for file {f.name}?: "
                    )
                new_stem = "-".join(
                    (
                        main_part_name,
                        data_part_name,
                        parts["Functional"],
                        parts["Basis-set"]
                    )
                )
                arcname = f"{new_stem}{f.suffix}"
                archive.add(f, arcname=arcname)
                hasher = hashlib.md5()
                with open(f, "rb") as h5file:
                    for chunk in iter(lambda: h5file.read(_CHUNK_SIZE), b""):
                        hasher.update(chunk)

                data_dict[ds_name]["lot"][lot]["files"].append(arcname)
                md5 = hasher.hexdigest()
                if (arcname, md5) not in registered_md5s:
                    writer.writerow([arcname, md5])
                    registered_md5s.add((arcname, md5))
                else:
                    print(
                        f"NOTE: File {arcname} is already present in the MD5 registry"
                    )

        if internal:
            json_path = _DATASETS_JSON_PATH
            with open(_DATASETS_JSON_PATH, mode="rt", encoding="utf-8") as fj:
                internal_data_dict = json.load(fj)
            if ds_name in internal_data_dict:
                if lot in internal_data_dict[ds_name]["lot"]:
                    raise ValueError(
                        f"Dataset with name {ds_name} and lot {lot} should be unique"
                    )
                else:
                    data_dict[ds_name].pop("default-lot")
                    internal_data_dict[ds_name]["lot"].update(
                        data_dict["ds_name"]["lot"]
                    )
            else:
                internal_data_dict.update(data_dict)
                data_dict = internal_data_dict.copy()
        else:
            json_path = dest_dir / f"{parts['Data']}.json"

        with open(json_path, "wt", encoding="utf-8") as fj:
            json.dump(data_dict, fj)
