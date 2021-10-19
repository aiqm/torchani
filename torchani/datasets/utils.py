r"""Utilities for working with ANI Datasets"""
import warnings
from typing import List, Tuple, Optional, Dict

import torch
from torch import Tensor

from ..units import hartree2kcalmol
from ..models import BuiltinModel
from ..utils import tqdm
from ..nn import Ensemble
from ._annotations import Conformers, StrPath
from .datasets import ANIDataset
from ._backends import TemporaryLocation


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


def filter_by_high_force(dataset: ANIDataset,
                         criteria: str = 'magnitude',
                         threshold: float = 2.0,
                         max_split: int = 2560,
                         delete_inplace: bool = False,
                         device: str = 'cpu',
                         verbose: bool = True) -> Optional[Tuple[List[Conformers], Dict[str, Tensor]]]:
    r"""Filter outlier conformations either by force components or force magnitude"""
    if criteria == 'magnitude':
        desc = f"Filtering where force magnitude > {threshold} Ha / Angstrom"
    elif criteria == 'components':
        desc = f"Filtering where any force component > {threshold} Ha / Angstrom"
    else:
        raise ValueError('Criteria must be one of "magnitude" or "components"')
    if dataset.grouping == 'legacy':
        raise ValueError("Legacy grouping not supported in filters")
    # Threshold is by default 2 Ha / Angstrom
    bad_keys_and_idxs: Dict[str, Tensor] = dict()
    with torch.no_grad():
        for key, g in tqdm(dataset.items(),
                           total=dataset.num_conformer_groups,
                           desc=desc,
                           disable=not verbose):
            # conformers are split into pieces of up to max_split to avoid
            # loading large groups into GPU memory at the same time and
            # calculating over them
            species, coordinates, forces = _fetch_splitted(g, ('species', 'coordinates', 'forces'), max_split)
            for split_idx, (s, c, f) in enumerate(zip(species, coordinates, forces)):
                s, c, f = s.to(device), c.to(device), f.to(device)
                if criteria == 'components':
                    # any over atoms and over x y z
                    bad_idxs = (f.abs() > threshold).any(dim=-1).any(dim=-1).nonzero().squeeze()
                elif criteria == 'magnitude':
                    # any over atoms
                    bad_idxs = (f.norm(dim=-1) > threshold).any(dim=-1).nonzero().squeeze()
                if bad_idxs.numel() > 0:
                    bad_keys_and_idxs.update({key: bad_idxs + split_idx * max_split})
                del s, c, f
    if bad_keys_and_idxs:
        return _fetch_and_delete_conformations(dataset, bad_keys_and_idxs, device, delete_inplace, verbose)
    return None


def filter_by_high_energy_error(dataset: ANIDataset,
                                model: BuiltinModel,
                                threshold: int = 100,
                                max_split: int = 2560,
                                device: str = 'cpu',
                                delete_inplace: bool = False,
                                verbose: bool = True) -> Optional[Tuple[List[Conformers], Dict[str, Tensor]]]:
    r"""Filter conformations for which a model has an excessively high absolute error"""
    if dataset.grouping == 'legacy':
        raise ValueError("Legacy grouping not supported in filters")
    bad_keys_and_idxs: Dict[str, Tensor] = dict()
    model = model.to(device)
    if not model.periodic_table_index:
        raise ValueError("Periodic table index must be True to filter high energy error")
    is_ensemble = isinstance(model.neural_networks, Ensemble)
    with torch.no_grad():
        for key, g in tqdm(dataset.items(),
                           total=dataset.num_conformer_groups,
                           desc=f"Filtering where any |energy error| > {threshold} kcal / mol"):
            species, coordinates, target_energies = _fetch_splitted(g, ('species', 'coordinates', 'energies'), max_split)
            for split_idx, (s, c, ta) in enumerate(zip(species, coordinates, target_energies)):
                s, c, ta = s.to(device), c.to(device), ta.to(device)
                if is_ensemble:
                    member_energies = model.members_energies((s, c)).energies
                else:
                    member_energies = model((s, c)).energies.unsqueeze(0)
                errors = hartree2kcalmol((member_energies - ta).abs())
                # any over individual models of the ensemble
                bad_idxs = (errors > threshold).any(dim=0).nonzero().squeeze()
                if bad_idxs.numel() > 0:
                    bad_keys_and_idxs.update({key: bad_idxs + split_idx * max_split})
                del s, c, ta
    if bad_keys_and_idxs:
        return _fetch_and_delete_conformations(dataset, bad_keys_and_idxs, device, delete_inplace, verbose)
    return None


def _fetch_and_delete_conformations(dataset: ANIDataset,
                                    bad_keys_and_idxs: Dict[str, Tensor],
                                    device: str,
                                    delete_inplace: bool,
                                    verbose: bool) -> Tuple[List[Conformers], Dict[str, Tensor]]:
    bad_conformations: List[Conformers] = []
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
        print(f"Deleted {total_filtered} conformations")
    return bad_conformations, bad_keys_and_idxs


# Input is a tuple of keys, output is a tuple of splitted tuples of tensors,
# corresponding to the input keys, each of which has at most "max_split" len
def _fetch_splitted(conformers: Conformers,
                    keys_to_split: Tuple[str, ...],
                    max_split: int) -> Tuple[Tuple[Tensor, ...], ...]:
    return tuple(torch.split(conformers[k], max_split) for k in keys_to_split)
