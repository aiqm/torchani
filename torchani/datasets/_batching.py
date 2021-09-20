r"""Functions for creating batched datasets"""
import warnings
import math
import json
import pickle
import datetime
from pathlib import Path
from typing import Tuple, Dict, Optional, Sequence, List, Union, Collection
from collections import OrderedDict

import h5py
import torch
from torch import Tensor
import numpy as np

from ..utils import pad_atomic_properties, cumsum_from_zero, PADDING, tqdm
from .datasets import ANIDataset, ANIBatchedDataset
from ._annotations import Conformers, StrPath, Transform


# TODO a batcher class would make this code much more clear
def create_batched_dataset(locations: Union[Collection[StrPath], StrPath, ANIDataset],
                           dest_path: Optional[StrPath] = None,
                           shuffle: bool = True,
                           shuffle_seed: Optional[int] = None,
                           file_format: str = 'hdf5',
                           include_properties: Optional[Sequence[str]] = None,
                           batch_size: int = 2560,
                           max_batches_per_packet: int = 350,
                           padding: Optional[Dict[str, float]] = None,
                           splits: Optional[Dict[str, float]] = None,
                           folds: Optional[int] = None,
                           inplace_transform: Optional[Transform] = None,
                           direct_cache: bool = False,
                           verbose: bool = True) -> Dict[str, ANIBatchedDataset]:

    if file_format != 'hdf5' and include_properties is None:
        include_properties = ('species', 'coordinates', 'energies')
        warnings.warn('Only species, coordinates and energies are included by default if format is not hdf5')

    if folds is not None and splits is not None:
        raise ValueError('Only one of ["folds", "splits"] should be specified')

    if direct_cache and dest_path is not None:
        raise ValueError("Destination path not needed for direct cache")

    # NOTE: All the tensor manipulation in this function is handled in CPU
    if dest_path is None:
        dest_path = Path.cwd() / f'batched_dataset_{file_format}'
    else:
        dest_path = Path(dest_path).resolve()

    if isinstance(locations, ANIDataset):
        dataset = locations
    else:
        dataset = ANIDataset(locations)

    # (1) Get all indices and shuffle them if needed
    #
    # These are pairs of indices that index first the group and then the
    # specific conformer, it is possible to just use one index for
    # everything but this is simpler at the cost of slightly more memory.
    # First we get all group sizes for all datasets concatenated in a tensor, in the same
    # order as h5_map
    group_sizes_values = torch.tensor(tuple(dataset.group_sizes.values()), dtype=torch.long)
    conformer_indices = torch.cat([torch.stack((torch.full(size=(s.item(),), fill_value=j, dtype=torch.long),
                                               (torch.arange(0, s.item(), dtype=torch.long))), dim=-1)
                                                for j, s in enumerate(group_sizes_values)])
    rng = _get_random_generator(shuffle, shuffle_seed)
    conformer_indices = _maybe_shuffle_indices(conformer_indices, rng)

    # (2) Split shuffled indices according to requested dataset splits or folds
    # by defaults we use splits, if folds or splits is specified we
    # do the specified operation
    if folds is not None:
        conformer_splits, split_paths = _divide_into_folds(conformer_indices, dest_path, folds, rng, direct_cache)
    else:
        if splits is None:
            splits = {'training': 0.8, 'validation': 0.2}

        if not math.isclose(sum(list(splits.values())), 1.0):
            raise ValueError("The sum of the split fractions has to add up to one")

        conformer_splits, split_paths = _divide_into_splits(conformer_indices, dest_path, splits, direct_cache)

    # (3) Compute the batch indices for each split and save the conformers to disk
    batched_datasets = _save_splits_into_batches(split_paths,
                                                 conformer_splits,
                                                 inplace_transform,
                                                 file_format,
                                                 include_properties,
                                                 dataset,
                                                 padding,
                                                 batch_size,
                                                 max_batches_per_packet,
                                                 direct_cache,
                                                 verbose)
    # log creation data
    if not direct_cache:
        creation_log = {'datetime_created': str(datetime.datetime.now()),
                        'source_store_locations': dataset.store_locations,
                        'splits': splits,
                        'folds': folds,
                        'padding': PADDING if padding is None else padding,
                        'shuffle': shuffle,
                        'shuffle_seed': shuffle_seed,
                        'include_properties': sorted(include_properties) if include_properties is not None else 'all',
                        'batch_size': batch_size,
                        'total_num_conformers': dataset.num_conformers,
                        'total_conformer_groups': dataset.num_conformer_groups}
        with open(dest_path.joinpath('creation_log.json'), 'w') as logfile:
            json.dump(creation_log, logfile, indent=1)
    return batched_datasets


def _get_random_generator(shuffle: bool = False, shuffle_seed: Optional[int] = None) -> Optional[torch.Generator]:

    if shuffle_seed is not None:
        assert shuffle
        seed = shuffle_seed
    else:
        seed = torch.random.seed()

    if shuffle:
        return torch.random.manual_seed(seed)
    return None


def _maybe_shuffle_indices(conformer_indices: Tensor,
                           rng: Optional[torch.Generator] = None) -> Tensor:
    total_num_conformers = len(conformer_indices)
    if rng is not None:
        shuffle_indices = torch.randperm(total_num_conformers, generator=rng)
        conformer_indices = conformer_indices[shuffle_indices]
    else:
        warnings.warn("Dataset will not be shuffled, this should only be used for debugging")
    return conformer_indices


def _divide_into_folds(conformer_indices: Tensor,
                        dest_path: Path,
                        folds: int,
                        rng: Optional[torch.Generator] = None,
                        direct_cache: bool = False) -> Tuple[Tuple[Tensor, ...], 'OrderedDict[str, Path]']:

    # the idea here is to work with "blocks" of size num_conformers / folds
    # cast to list for mypy
    conformer_blocks = list(torch.chunk(conformer_indices, folds))
    conformer_splits: List[Tensor] = []
    split_paths_list: List[Tuple[str, Path]] = []

    print(f"Generating {folds} folds for cross validation or ensemble training")
    for f in range(folds):
        # the first shuffle is necessary so that validation splits are shuffled
        validation_split = conformer_blocks[f]

        training_split = torch.cat(conformer_blocks[:f] + conformer_blocks[f + 1:])
        # afterwards all training folds are reshuffled to get different
        # batching for different models in the ensemble / cross validation
        # process (it is technically redundant to reshuffle the first one but
        # it is done for simplicity)
        training_split = _maybe_shuffle_indices(training_split, rng)
        conformer_splits.extend([training_split, validation_split])
        split_paths_list.extend([(f'training{f}', dest_path.joinpath(f'training{f}')),
                                 (f'validation{f}', dest_path.joinpath(f'validation{f}'))])
    split_paths = OrderedDict(split_paths_list)
    if not direct_cache:
        _create_split_paths(split_paths)

    return tuple(conformer_splits), split_paths


def _divide_into_splits(conformer_indices: Tensor,
                        dest_path: Path,
                        splits: Dict[str, float],
                        direct_cache: bool = False) -> Tuple[Tuple[Tensor, ...], 'OrderedDict[str, Path]']:
    total_num_conformers = len(conformer_indices)
    split_sizes = OrderedDict([(k, int(total_num_conformers * v)) for k, v in splits.items()])
    split_paths = OrderedDict([(k, dest_path.joinpath(k)) for k in split_sizes.keys()])
    if not direct_cache:
        _create_split_paths(split_paths)

    leftover = total_num_conformers - sum(split_sizes.values())
    if leftover != 0:
        # We slightly modify a random section if the fractions don't split
        # the dataset perfectly. This also automatically takes care of the
        # cases leftover > 0 and leftover < 0
        any_key = list(split_sizes.keys())[0]
        split_sizes[any_key] += leftover
        assert sum(split_sizes.values()) == total_num_conformers
    conformer_splits = torch.split(conformer_indices, list(split_sizes.values()))
    assert len(conformer_splits) == len(split_sizes.values())
    print(f'Splits have number of conformers: {dict(split_sizes)}.'
          f' The requested percentages were: {splits}')
    return conformer_splits, split_paths


def _create_split_paths(split_paths: 'OrderedDict[str, Path]') -> None:
    for p in split_paths.values():
        if p.is_dir():
            subdirs = [d for d in p.iterdir()]
            if subdirs:
                raise ValueError('The dest_path provided already has files'
                                 ' or directories, please provide'
                                 ' a different path')
        else:
            if p.is_file():
                raise ValueError('The dest_path is a file, it should be a directory')
            p.mkdir(parents=True)


def _save_splits_into_batches(split_paths: 'OrderedDict[str, Path]',
                              conformer_splits: Tuple[Tensor, ...],
                              inplace_transform: Optional[Transform],
                              file_format: str,
                              include_properties: Optional[Sequence[str]],
                              dataset: ANIDataset,
                              padding: Optional[Dict[str, float]],
                              batch_size: int,
                              max_batches_per_packet: int,
                              direct_cache: bool,
                              verbose: bool) -> Dict[str, ANIBatchedDataset]:
    # NOTE: Explanation for following logic, please read
    #
    # This sets up a given number of batches (packet) to keep in memory and
    # then scans the dataset and find the conformers needed for the packet. It
    # then saves the batches and fetches the next packet.
    #
    # A "packet" is a list that has tensors, each of which
    # has batch indices, for instance [tensor([[0, 0, 1, 1, 2], [1, 2, 3, 5]]),
    #                                  tensor([[3, 5, 5, 5], [1, 2, 3, 3]])]
    # would be a "packet" of 2 batch_indices, each of which has in the first
    # row the index for the group, and in the second row the index for the
    # conformer
    #
    # It is important to do this with a packet and not only 1 batch.  The
    # number of reads to the h5 file is batches x conformer_groups x 3 for 1x
    # (factor of 3 from energies, species, coordinates), which means ~ 2000 x
    # 3000 x 3 = 9M reads, this is a bad bottleneck and very slow, even if we
    # fetch all necessary molecules from each conformer group simultaneously.
    #
    # Doing it for all batches at the same time is (reasonably) fast, ~ 9000
    # reads, but in this case it means we will have to put all, or almost all
    # the dataset into memory at some point, which is not feasible for larger
    # datasets.
    if inplace_transform is None:
        inplace_transform = lambda x: x  # noqa: E731

    # get all group keys concatenated in a list, with the associated file indexes
    key_list = list(dataset.keys())

    # Important: to prevent possible bugs / errors, that may happen
    # due to incorrect conversion to indices, species is **always*
    # converted to atomic numbers when saving the batched dataset.
    batched_datasets: Dict[str, ANIBatchedDataset] = dict()
    with dataset.keep_open() as ro_dataset:
        for (split_name, split_path), indices_of_split in zip(split_paths.items(), conformer_splits):
            all_batch_indices = torch.split(indices_of_split, batch_size)

            all_batch_indices_packets = [all_batch_indices[j:j + max_batches_per_packet]
                                        for j in range(0, len(all_batch_indices), max_batches_per_packet)]
            num_batch_indices_packets = len(all_batch_indices_packets)

            overall_batch_idx = 0
            if direct_cache:
                in_memory_batches: List[Conformers] = []
            for j, batch_indices_packet in enumerate(all_batch_indices_packets):
                num_batches_in_packet = len(batch_indices_packet)
                # Now first we cat and sort according to the first index in order to
                # fetch all conformers of the same group simultaneously
                batch_indices_cat = torch.cat(batch_indices_packet, 0)
                indices_to_sort_batch_indices_cat = torch.argsort(batch_indices_cat[:, 0])
                sorted_batch_indices_cat = batch_indices_cat[indices_to_sort_batch_indices_cat]
                uniqued_idxs_cat, counts_cat = torch.unique_consecutive(sorted_batch_indices_cat[:, 0],
                                                                        return_counts=True)
                cumcounts_cat = cumsum_from_zero(counts_cat)

                # batch_sizes and indices_to_unsort are needed for the
                # reverse operation once the conformers have been
                # extracted
                batch_sizes = [len(batch_indices) for batch_indices in batch_indices_packet]
                indices_to_unsort_batch_cat = torch.argsort(indices_to_sort_batch_indices_cat)
                assert len(batch_sizes) <= max_batches_per_packet

                all_conformers: List[Conformers] = []
                end_idxs = counts_cat + cumcounts_cat
                groups_slices = zip(uniqued_idxs_cat, cumcounts_cat, end_idxs)
                if direct_cache:
                    desc = (f'Saving batch packet {j + 1} of {num_batch_indices_packets} '
                            f'of split {split_path.name} into memory')
                else:
                    desc = (f'Saving batch packet {j + 1} of {num_batch_indices_packets} '
                            f'of split {split_path.name} in format {file_format}')
                for step, group_slice in tqdm(enumerate(groups_slices),
                                              total=len(counts_cat),
                                              desc=desc,
                                              disable=not verbose):
                    group_idx, start, end = group_slice
                    # select the specific group from the whole list of files
                    # and get a slice with the indices to extract the necessary
                    # conformers from the group for all batches in pack.
                    selected_indices = sorted_batch_indices_cat[start:end, 1]
                    assert selected_indices.dim() == 1
                    conformers = ro_dataset.get_conformers(key_list[group_idx.item()],
                                                                      selected_indices,
                                                                      properties=include_properties)
                    all_conformers.append(conformers)
                batches_cat = pad_atomic_properties(all_conformers, padding)
                # Now we need to reassign the conformers to the specified
                # batches. Since to get here we cat'ed and sorted, to
                # reassign we need to unsort and split.
                # The format of this is {'species': (batch1, batch2, ...), 'coordinates': (batch1, batch2, ...)}
                batch_packet_dict = {k: torch.split(t[indices_to_unsort_batch_cat], batch_sizes)
                                     for k, t in batches_cat.items()}
                for packet_batch_idx in range(num_batches_in_packet):
                    batch = {k: v[packet_batch_idx] for k, v in batch_packet_dict.items()}
                    batch = inplace_transform(batch)
                    if direct_cache:
                        in_memory_batches.append(batch)
                    else:
                        _save_batch(split_path, overall_batch_idx, batch, file_format, len(all_batch_indices))
                    overall_batch_idx += 1
            if direct_cache:
                split_ds = ANIBatchedDataset(batches=in_memory_batches, split=split_name)
                split_ds = split_ds.cache(verbose=False, pin_memory=torch.cuda.is_available())
                batched_datasets[split_name] = split_ds
            else:
                batched_datasets[split_name] = ANIBatchedDataset(store_dir=split_path.parent, split=split_name)
    return batched_datasets


# Saves the batch to disk; we use pickle, numpy or hdf5 since saving in pytorch
# format is extremely slow
def _save_batch(path: Path, idx: int, batch: Conformers, file_format: str, total_batches: int) -> None:
    batch = {k: v.numpy() for k, v in batch.items()}
    # The batch names are e.g. 00034_batch.h5
    batch_path = path / f'{str(idx).zfill(len(str(total_batches)))}_batch'
    if file_format == 'pickle':
        with open(batch_path.with_suffix('.pkl'), 'wb') as batch_file:
            pickle.dump(batch, batch_file)
    elif file_format == 'numpy':
        np.savez(batch_path, **batch)
    elif file_format == 'hdf5':
        with h5py.File(batch_path.with_suffix('.h5'), 'w-') as f:
            for k, v in batch.items():
                f.create_dataset(k, data=v)
    else:
        raise ValueError("Unknown file format")
