r"""Utilities for working with ANI Datasets"""
import typing as tp
import warnings

import torch
from torch import Tensor

from torchani.units import hartree2kcalmol
from torchani.models import BuiltinModel
from torchani.utils import tqdm
from torchani.nn import Ensemble
from torchani.datasets._annotations import Conformers, StrPath
from torchani.datasets.datasets import ANIDataset
from torchani.datasets._backends import TemporaryLocation


__all__ = ['filter_by_high_force', 'filter_by_high_energy_error', 'concatenate']


def concatenate(source: ANIDataset,
                dest_location: StrPath,
                verbose: bool = True,
                delete_originals: bool = False) -> ANIDataset:
    r"""Combine all the backing stores in a given ANIDataset into one"""
    if source.grouping not in ['by_formula', 'by_num_atoms']:
        raise ValueError("Please regroup your dataset before concatenating")

    if source.metadata:
        warnings.warn("Source dataset has metadata which will not be copied.")

    with TemporaryLocation(source._first_subds._backend) as tmp_location:
        dest = ANIDataset(tmp_location,
                          create=True,
                          grouping=source.grouping,
                          verbose=False)
        for k, v in tqdm(source.numpy_items(),
                      desc='Concatenating datasets',
                      total=source.num_conformer_groups,
                      disable=not verbose):
            dest.append_conformers(k.split('/')[-1], v)
        dest._first_subds._store.location.root = dest_location
    # TODO this depends on the original stores being files, it should be
    # changed for generality
    if delete_originals:
        for subds in tqdm(source._datasets.values(),
                      desc='Deleting original store',
                      total=source.num_stores,
                      disable=not verbose):
            del subds._store.location.root
    return dest


def filter_by_high_force(
    dataset: ANIDataset,
    criteria: str = 'magnitude',
    threshold: float = 2.0,  # Ha / Angstrom
    max_split: int = 2560,
    delete_inplace: bool = False,
    device: str = 'cpu',
    verbose: bool = True
) -> tp.Optional[tp.Tuple[tp.List[Conformers], tp.Dict[str, Tensor]]]:
    r"""
    Filter outlier conformations in a dataset, either by force components or
    force magnitude
    """
    if dataset.grouping == 'legacy':
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
            if criteria == 'components':
                bad_idxs = (f.abs() > threshold).any(dim=-1).any(dim=-1).nonzero().squeeze()
            elif criteria == 'magnitude':
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
        return _fetch_and_delete_conformations(dataset, bad_keys_and_idxs, device, delete_inplace, verbose)
    return (list(), dict())


def filter_by_high_energy_error(
    dataset: ANIDataset,
    model: BuiltinModel,
    threshold: float = 100.0,
    max_split: int = 2560,
    device: str = 'cpu',
    delete_inplace: bool = False,
    verbose: bool = True,
) -> tp.Tuple[tp.List[Conformers], tp.Dict[str, Tensor]]:
    r"""
    Filter conformations for which a model has an excessively high absolute
    error w.r.t. a given ANI model
    """
    if dataset.grouping == 'legacy':
        raise ValueError("Legacy grouping not supported in filters")

    _bad_keys_and_idxs: tp.Dict[str, tp.List[Tensor]] = dict()
    model = model.to(device)
    if not model.periodic_table_index:
        raise ValueError("Periodic table index must be True to filter high energy error")

    desc = f"Filtering where any |energy error| > {threshold} kcal / mol"
    with torch.no_grad():
        for key, cumul_idx, group in tqdm(
            dataset.chunked_items(max_size=max_split, properties=("species", "coordinates", "energies")),
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
            errors = hartree2kcalmol((member_energies - ta).abs())
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
        return _fetch_and_delete_conformations(dataset, bad_keys_and_idxs, device, delete_inplace, verbose)
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
        bad_conformations.append({k: v.to(device)
                                  for k, v in dataset.get_conformers(k, idxs).items()})
    if delete_inplace:
        for key, idx in tqdm(bad_keys_and_idxs.items(),
                             total=len(bad_keys_and_idxs),
                             desc='Deleting filtered conformers',
                             disable=not verbose):
            dataset.delete_conformers(key, idx)
    if verbose:
        total_filtered = sum(v.numel() for v in bad_keys_and_idxs.values())
        if delete_inplace:
            print(f"Deleted {total_filtered} bad conformations")
        else:
            print(f"Found {total_filtered} bad conformations")
    return bad_conformations, bad_keys_and_idxs
