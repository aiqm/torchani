from pathlib import Path
import json
import tarfile
import csv
import re
import typing as tp

import torch
from torch import Tensor
from tqdm import tqdm

from torchani.units import hartree2kcalpermol
from torchani.models import ANI
from torchani.nn import Ensemble
from torchani.annotations import Conformers, StrPath, Backend
from torchani.datasets.anidataset import ANIDataset
from torchani.datasets.builtin import _calc_file_md5


__all__ = [
    "filter_by_high_force",
    "filter_by_high_energy_error",
    "concatenate",
    "datapack",
]


def datapack(
    src_dir: Path,
    dest_dir: Path,
    name: str,
    lot: str,
    suffix: str = ".h5",
) -> None:
    r"""
    If passed a directory with .h5 files, generates a corresponding
    dataset.json, md5.csv, and dataset.tar.gz.

    That this function is interactive by design, it always expects input from a
    user.

    name: str
        name of the dataset
    lot: str
        Level of Theory of the dataset, in format <method>-<basis>
    paths: Path | Sequence[Path]
        Path to a directory with .h5 files, or sequence of paths to .h5 files
    dest_dir: Path
        Destination directory for the archive, md5.csv and dataset.json
    """

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


def concatenate(
    source: ANIDataset,
    dest_location: StrPath,
    verbose: bool = True,
    backend: Backend = "hdf5",
    delete_originals: bool = False,
) -> ANIDataset:
    r"""Combine all the backing stores in a given ANIDataset into one"""
    dest_location = Path(dest_location).resolve()
    if source.grouping not in ["by_formula", "by_num_atoms"]:
        raise ValueError("Please regroup your dataset before concatenating")

    dest = ANIDataset("tmp", backend=backend, grouping=source.grouping, verbose=False)
    try:
        for k, v in tqdm(
            source.numpy_items(),
            desc="Concatenating datasets",
            total=source.num_conformer_groups,
            disable=not verbose,
        ):
            dest.append_conformers(k.split("/")[-1], v)
    except Exception:
        dest._first_subds._store.location.clear()

    dest._first_subds._store.location.root = dest_location
    if delete_originals:
        for subds in tqdm(
            source._datasets.values(),
            desc="Deleting original stores",
            total=source.num_stores,
            disable=not verbose,
        ):
            subds._store.location.clear()
    return dest


def filter_by_high_force(
    dataset: ANIDataset,
    criteria: str = "magnitude",
    threshold: float = 2.0,  # Ha / Angstrom
    max_split: int = 2560,
    delete_inplace: bool = False,
    device: str = "cpu",
    verbose: bool = True,
) -> tp.Optional[tp.Tuple[tp.List[Conformers], tp.Dict[str, Tensor]]]:
    r"""
    Filter outlier conformations in a dataset, either by force components or
    force magnitude
    """
    if dataset.grouping == "legacy":
        raise ValueError("Legacy grouping not supported in filters")
    if criteria not in {"magnitude", "components"}:
        raise ValueError('Criteria must be one of "magnitude" or "components"')

    desc = f"Filtering where any atomic force {criteria} > {threshold} Ha / Angstrom"
    _bad_keys_and_idxs: tp.Dict[str, tp.List[Tensor]] = dict()
    with torch.no_grad():
        for key, cumul_idx, group in tqdm(
            dataset.chunked_items(max_size=max_split, properties=("forces",)),
            total=dataset.num_conformers // max_split,
            desc=desc,
            disable=not verbose,
        ):
            f = tp.cast(Tensor, group["forces"]).to(device)
            if criteria == "components":
                bad_idxs = (
                    (f.abs() > threshold).any(dim=-1).any(dim=-1).nonzero().squeeze()
                )
            elif criteria == "magnitude":
                bad_idxs = (f.norm(dim=-1) > threshold).any(dim=-1).nonzero().squeeze()

            if bad_idxs.numel() > 0:
                # unsqueeze scalar tensors
                if not bad_idxs.shape:
                    bad_idxs = bad_idxs.unsqueeze(0)
                if key not in _bad_keys_and_idxs:
                    _bad_keys_and_idxs[key] = [bad_idxs + cumul_idx]
                else:
                    _bad_keys_and_idxs[key].append(bad_idxs + cumul_idx)
            del f
    if _bad_keys_and_idxs:
        bad_keys_and_idxs = {k: torch.cat(v) for k, v in _bad_keys_and_idxs.items()}
        return _fetch_and_delete_conformations(
            dataset, bad_keys_and_idxs, device, delete_inplace, verbose
        )
    return (list(), dict())


def filter_by_high_energy_error(
    dataset: ANIDataset,
    model: ANI,
    threshold: float = 100.0,
    max_split: int = 2560,
    device: str = "cpu",
    delete_inplace: bool = False,
    verbose: bool = True,
) -> tp.Tuple[tp.List[Conformers], tp.Dict[str, Tensor]]:
    r"""
    Filter conformations for which a model has an excessively high absolute
    error w.r.t. a given ANI model
    """
    if dataset.grouping == "legacy":
        raise ValueError("Legacy grouping not supported in filters")

    _bad_keys_and_idxs: tp.Dict[str, tp.List[Tensor]] = dict()
    model = model.to(device)
    if not model.periodic_table_index:
        raise ValueError(
            "Periodic table index must be True to filter high energy error"
        )

    desc = f"Filtering where any |energy error| > {threshold} kcal / mol"
    with torch.no_grad():
        for key, cumul_idx, group in tqdm(
            dataset.chunked_items(
                max_size=max_split, properties=("species", "coordinates", "energies")
            ),
            total=dataset.num_conformers // max_split,
            desc=desc,
            disable=not verbose,
        ):
            s = tp.cast(Tensor, group["species"]).to(device)
            c = tp.cast(Tensor, group["coordinates"]).to(device)
            ta = tp.cast(Tensor, group["energies"]).to(device)

            if isinstance(model.neural_networks, Ensemble):
                member_energies = model.members_energies((s, c)).energies
            else:
                member_energies = model((s, c)).energies.unsqueeze(0)
            errors = hartree2kcalpermol((member_energies - ta).abs())
            # any over individual models of the ensemble
            bad_idxs = (errors > threshold).any(dim=0).nonzero().squeeze()

            if bad_idxs.numel() > 0:
                # unsqueeze scalar tensors
                if not bad_idxs.shape:
                    bad_idxs = bad_idxs.unsqueeze(0)
                if key not in _bad_keys_and_idxs:
                    _bad_keys_and_idxs[key] = [bad_idxs + cumul_idx]
                else:
                    _bad_keys_and_idxs[key].append(bad_idxs + cumul_idx)
            del s, c, ta
    if _bad_keys_and_idxs:
        bad_keys_and_idxs = {k: torch.cat(v) for k, v in _bad_keys_and_idxs.items()}
        return _fetch_and_delete_conformations(
            dataset, bad_keys_and_idxs, device, delete_inplace, verbose
        )
    return (list(), dict())


def _fetch_and_delete_conformations(
    dataset: ANIDataset,
    bad_keys_and_idxs: tp.Dict[str, Tensor],
    device: str,
    delete_inplace: bool,
    verbose: bool,
) -> tp.Tuple[tp.List[Conformers], tp.Dict[str, Tensor]]:
    bad_conformations: tp.List[Conformers] = []
    for k, idxs in bad_keys_and_idxs.items():
        if idxs.ndim == 0:
            idxs = idxs.unsqueeze(0)
        bad_conformations.append(
            {k: v.to(device) for k, v in dataset.get_conformers(k, idxs).items()}
        )
    if delete_inplace:
        for key, idx in tqdm(
            bad_keys_and_idxs.items(),
            total=len(bad_keys_and_idxs),
            desc="Deleting filtered conformers",
            disable=not verbose,
        ):
            dataset.delete_conformers(key, idx)
    if verbose:
        total_filtered = sum(v.numel() for v in bad_keys_and_idxs.values())
        if delete_inplace:
            print(f"Deleted {total_filtered} bad conformations")
        else:
            print(f"Found {total_filtered} bad conformations")
    return bad_conformations, bad_keys_and_idxs
