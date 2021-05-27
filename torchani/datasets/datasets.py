from pathlib import Path
from functools import partial
import json
import datetime
import math
import pickle
import warnings
import importlib
from typing import Union, Optional, Dict, Sequence, Iterator, Tuple, List, Set, Callable
from collections import OrderedDict, Counter
from collections.abc import Mapping

import h5py
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray

from torchani import utils

PKBAR_INSTALLED = importlib.util.find_spec('pkbar') is not None
if PKBAR_INSTALLED:
    import pkbar

# type alias for transform
Transform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]


class AniBatchedDataset(torch.utils.data.Dataset):

    SUPPORTED_FILE_FORMATS = ('numpy', 'hdf5', 'single_hdf5', 'pickle')
    batch_size: Optional[int]

    def __init__(self, store_dir: Union[str, Path],
                       file_format: Optional[str] = None,
                       split: str = 'training',
                       transform: Transform = lambda x: x,
                       flag_property: Optional[str] = None,
                       drop_last: bool = False):

        self.split = split
        self.store_dir = Path(store_dir).resolve().joinpath(self.split)
        if not self.store_dir.is_dir():
            raise ValueError(f'The directory {self.store_dir.as_posix()} exists, '
                             f'but the split {split} could not be found')

        self.batch_paths = [f for f in self.store_dir.iterdir()]
        if not self.batch_paths:
            raise RuntimeError("The path provided has no files")
        if not all([f.is_file() for f in self.batch_paths]):
            raise RuntimeError("Subdirectories were found in path, this is not supported")

        suffix = self.batch_paths[0].suffix
        if not all([f.suffix == suffix for f in self.batch_paths]):
            raise RuntimeError("Different file extensions were found in path, not supported")

        self.transform = transform

        def numpy_extractor(idx: int, paths: List[Path]) -> Dict[str, Tensor]:
            return {k: torch.as_tensor(v) for k, v in np.load(paths[idx]).items()}

        def pickle_extractor(idx: int, paths: List[Path]) -> Dict[str, Tensor]:
            with open(paths[idx], 'rb') as f:
                return {k: torch.as_tensor(v) for k, v in pickle.load(f).items()}

        def hdf5_extractor(idx: int, paths: List[Path]) -> Dict[str, Tensor]:
            with h5py.File(paths[idx], 'r') as f:
                return {k: torch.as_tensor(v[()]) for k, v in f['/'].items()}

        def single_hdf5_extractor(idx: int, group_keys: List[str], path: Path) -> Dict[str, Tensor]:
            k = group_keys[idx]
            with h5py.File(path, 'r') as f:
                return {k: torch.as_tensor(v[()]) for k, v in f[k].items()}

        # We use pickle or numpy or hdf5 since saving in
        # pytorch format is extremely slow
        if file_format is None:
            format_suffix_map = {'.npz': 'numpy', '.pkl': 'pickle', '.h5': 'hdf5'}
            file_format = format_suffix_map[suffix]
            if file_format == 'hdf5' and ('single' in self.batch_paths[0].name):
                file_format = 'single_hdf5'

        if file_format not in self.SUPPORTED_FILE_FORMATS:
            raise ValueError(f"The file format {file_format} is not in the"
                             f"supported formats {self.SUPPORTED_FILE_FORMATS}")

        if file_format == 'numpy':
            self.extractor = partial(numpy_extractor, paths=self.batch_paths)
        elif file_format == 'pickle':
            self.extractor = partial(pickle_extractor, paths=self.batch_paths)
        elif file_format == 'hdf5':
            self.extractor = partial(hdf5_extractor, paths=self.batch_paths)
        elif file_format == 'single_hdf5':
            warnings.warn('Depending on the implementation, a single HDF5 file '
                          'may not support parallel reads, so using num_workers > 1 '
                          'may have a detrimental effect on performance')
            with h5py.File(self.batch_paths[0], 'r') as f:
                keys = list(f.keys())
                self._len = len(keys)
                self.extractor = partial(single_hdf5_extractor, group_keys=keys, path=self.batch_paths[0])
        else:
            raise RuntimeError(f'Format for file with extension {suffix} '
                                'could not be inferred, please specify explicitly')
        try:
            with open(self.store_dir.parent.joinpath('creation_log.json'), 'r') as logfile:
                creation_log = json.load(logfile)
            self.is_inplace_transformed = creation_log['is_inplace_transformed']
            self.batch_size = creation_log['batch_size']
        except Exception:
            warnings.warn("No creation log found, is_inplace_transformed assumed False, and batch_size is set to None")
            self.is_inplace_transformed = False
            self.batch_size = None

        self._flag_property = flag_property
        if drop_last:
            self._recalculate_batch_size_and_drop_last()

        self._len = len(self.batch_paths)

    def _recalculate_batch_size_and_drop_last(self) -> None:
        warnings.warn('Recalculating batch size is necessary for drop_last and it may take considerable time if your disk is an HDD')
        batch_sizes = {path: _get_properties_size(b, self._flag_property, set(b.keys()))
                           for b, path in zip(self, self.batch_paths)}
        batch_size_counts = Counter(batch_sizes.values())
        # in case that there are more than one batch sizes, self.batch_size
        # holds the most common one
        self.batch_size = batch_size_counts.most_common(1)[0][0]
        # we drop the batch with the smallest size, if there is only one of
        # them, otherwise this errors out
        assert len(batch_size_counts) in [1, 2], "More than two different batch lengths found"
        if len(batch_size_counts) == 2:
            smallest = min(batch_sizes.items(), key=lambda x: x[1])
            assert batch_size_counts[smallest[1]] == 1, "There is more than one small batch"
            self.batch_paths.remove(smallest[0])

    def cache(self, pin_memory: bool = True,
                    verbose: bool = True,
                    apply_transform: bool = True) -> 'AniBatchedDataset':
        if verbose:
            print(f"Cacheing split {self.split} of dataset, this may take some time...")
            print("Important: Cacheing the dataset may use a lot of memory, be careful!")

        def memory_extractor(idx: int, ds: AniBatchedDataset) -> Dict[str, Tensor]:
            return ds._data[idx]

        self._data = [self.extractor(idx) for idx in range(len(self))]

        if apply_transform:
            if verbose:
                print("Applying transforms if they are present...")
                print("Important: Transformations, if there are any present,"
                      " will be applied once during cacheing and then discarded.")
                print("If you want a different behavior pass apply_transform=False")
            with torch.no_grad():
                self._data = [self.transform(properties) for properties in self._data]
            # discard transform after aplication
            self.transform = lambda x: x

        # When the dataset is cached memory pinning is done here. When the
        # dataset is not cached memory pinning is done by the torch DataLoader.
        if pin_memory:
            if verbose:
                print("Pinning memory...")
                print("Important: Cacheing pins memory automatically.")
                print("Do **not** use pin_memory=True in torch.utils.data.DataLoader")
            self._data = [{k: v.pin_memory() for k, v in properties.items()} for properties in self._data]

        self.extractor = partial(memory_extractor, ds=self)
        return self

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # integral indices must be provided for compatibility with pytorch
        # DataLoader API
        properties = self.extractor(idx)
        with torch.no_grad():
            properties = self.transform(properties)
        return properties

    def __iter__(self) -> Iterator[Dict[str, Tensor]]:
        j = 0
        try:
            while True:
                yield self[j]
                j += 1
        except IndexError:
            return

    def __len__(self) -> int:
        return self._len


class AniH5Dataset(Mapping):

    def __init__(self,
                 store_file: Union[str, Path],
                 flag_property: Optional[str] = None,
                 element_keys: Sequence[str] = ('species', 'numbers', 'atomic_numbers', 'smiles')):
        store_file = Path(store_file).resolve()
        if not store_file.is_file():
            raise FileNotFoundError(f"The h5 file in {store_file.as_posix()} could not be found")

        self._store_file = store_file

        # flag key is used to infer size of molecule groups
        # when iterating over the dataset
        self._flag_property = flag_property

        group_sizes, supported_properties = self._cache_group_sizes_and_properties()
        self.group_sizes = OrderedDict(group_sizes)
        self.supported_properties = supported_properties

        # element keys are treated differently because they don't have a batch dimension
        self._supported_element_keys = tuple((k for k in self.supported_properties if k in element_keys))
        self._supported_non_element_keys = tuple((k for k in self.supported_properties if k not in element_keys))

        self.num_conformers = sum(self.group_sizes.values())
        self.num_conformer_groups = len(self.group_sizes.keys())

        self.symbols_to_atomic_numbers = utils.ChemicalSymbolsToAtomicNumbers()

    def __getitem__(self, key: str) -> Dict[str, ndarray]:
        # this is a simple extraction that just fetches everything
        return self._get_group(key, self._supported_non_element_keys, self._supported_element_keys)

    def __len__(self) -> int:
        return self.num_conformer_groups

    def __iter__(self) -> Iterator[str]:
        # Iterating over groups and yield the associated molecule groups as
        # dictionaries of numpy arrays (except for species, which is a list of
        # strings)
        return iter(self.group_sizes.keys())

    def get_conformers(self,
                       key: str,
                       idx: Optional[Union[int, ndarray]] = None,
                       include_properties: Optional[Sequence[str]] = None,
                       strict: bool = False,
                       raw_output: bool = True) -> Dict[str, ndarray]:
        element_keys, non_element_keys = self._properties_into_keys(include_properties)
        # fetching a conformer actually copies all the group into memory first,
        # because indexing directly into hdf5 is much slower.
        conformers = self._get_group(key, non_element_keys, element_keys, idx, strict)
        if raw_output:
            return conformers
        else:
            # here we convert species to atomic numbers and repeat along the
            # batch dimension all element_keys
            if 'species' in element_keys:
                tensor_species = self.symbols_to_atomic_numbers(conformers['species'].tolist())
                conformers['species'] = tensor_species.cpu().numpy()

            if isinstance(idx, ndarray):
                for k in element_keys:
                    conformers[k] = np.tile(conformers[k].reshape((1, -1)), (len(idx), 1))
            elif idx is None:
                any_key = non_element_keys[0]
                conformers[k] = np.tile(conformers[k].reshape((1, -1)), (conformers[any_key].shape[0], 1))

            return conformers

    def iter_conformers(self,
                        include_properties: Optional[Sequence[str]] = None,
                        strict: bool = False) -> Iterator[Dict[str, ndarray]]:
        for _, _, c in self.iter_key_idx_conformers(include_properties, strict):
            yield c

    def iter_key_idx_conformers(self,
                                include_properties: Optional[Sequence[str]] = None,
                                strict: bool = False) -> Iterator[Tuple[str, int, Dict[str, ndarray]]]:

        element_keys, non_element_keys = self._properties_into_keys(include_properties)

        # Iterate sequentially over conformers also copies all the group
        # into memory first, so it is also fast
        for k, size in self.group_sizes.items():
            conformer_group = self._get_group(k, non_element_keys, element_keys, None, strict)
            for idx in range(size):
                out_conformer_group = {k: conformer_group[k] for k in element_keys}
                out_conformer_group.update({k: conformer_group[k][idx] for k in non_element_keys})
                yield k, idx, out_conformer_group

    def _properties_into_keys(self,
                              properties: Optional[Sequence[str]] = None) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        if properties is None:
            element_keys = self._supported_element_keys
            non_element_keys = self._supported_non_element_keys
        elif set(properties).issubset(self.supported_properties):
            element_keys = tuple((k for k in properties if k in self._supported_element_keys))
            non_element_keys = tuple((k for k in properties if k not in self._supported_element_keys))
        else:
            raise ValueError(f"Some of the properties demanded {properties} are not "
                             f"in the dataset, which has properties {self.supported_properties}")
        return element_keys, non_element_keys

    def _cache_group_sizes_and_properties(self) -> Tuple[List[Tuple[str, int]], Set[str]]:
        # cache paths of all molecule groups into a list
        # and all supported properties into a set
        def visitor_fn(name: str,
                       object_: Union[h5py.Dataset, h5py.Group],
                       group_sizes: List[Tuple[str, int]],
                       supported_properties: Set[str],
                       flag_property: Optional[str] = None) -> None:

            if isinstance(object_, h5py.Dataset):
                molecule_group = object_.parent
                # Check if we already visited this group via one of its
                # children or not
                if molecule_group.name not in [tup[0] for tup in group_sizes]:
                    # Collect properties and check that all the datasets have
                    # the same properties
                    if not supported_properties:
                        supported_properties.update({k for k in molecule_group.keys()})
                        if flag_property is not None and flag_property not in supported_properties:
                            raise RuntimeError(f"Flag property {flag_property} "
                                               f"not found in {supported_properties}")
                    else:
                        if not {k for k in molecule_group.keys()} == supported_properties:
                            raise RuntimeError(f"group {molecule_group.name} has incompatible keys, "
                                               f"which should be {supported_properties}, inferred from other groups")
                    # Check for format correctness
                    for v in molecule_group.values():
                        if not isinstance(v, h5py.Dataset):
                            raise RuntimeError("Invalid dataset format, there "
                                               "shouldn't be Groups inside Groups "
                                               "that have Datasets")
                    group_sizes.append((molecule_group.name,
                                         _get_properties_size(molecule_group,
                                                              flag_property,
                                                              supported_properties)))

        group_sizes: List[Tuple[str, int]] = []
        supported_properties: Set[str] = set()

        with h5py.File(self._store_file, 'r') as f:
            f.visititems(partial(visitor_fn,
                                 group_sizes=group_sizes,
                                 supported_properties=supported_properties,
                                 flag_property=self._flag_property))

        return group_sizes, supported_properties

    def _get_group(self,
                   key: str,
                   non_element_keys: Tuple[str, ...],
                   element_keys: Tuple[str, ...],
                   idx: Optional[Union[int, ndarray]] = None,
                   strict: bool = False) -> Dict[str, ndarray]:

        # NOTE: If some keys are not found then
        # this returns a partial result with the keys that are found, (maybe
        # even empty) unless strict is passed.
        with h5py.File(self._store_file, 'r') as f:
            group = f[key]
            if strict and not all([p in group.keys() for p in element_keys + non_element_keys]):
                raise RuntimeError('Some of the requested properties could not '
                                  f'be found in group {key}')

            molecules = {k: np.copy(group[k]) for k in element_keys}
            if idx is None:
                molecules.update({k: np.copy(group[k]) for k in non_element_keys})
            else:
                molecules.update({k: np.copy(group[k])[idx] for k in non_element_keys})

            if 'species' in element_keys:
                molecules['species'] = molecules['species'].astype(str)
        return molecules


def _save_batch(path: Path, idx: int, batch: Dict[str, Tensor], file_format: str) -> None:
    # We use pickle, numpy or hdf5 since saving in
    # pytorch format is extremely slow
    batch = {k: v.numpy() for k, v in batch.items()}
    if file_format == 'pickle':
        with open(path.joinpath(f'batch{idx}.pkl'), 'wb') as batch_file:
            pickle.dump(batch, batch_file)
    elif file_format == 'numpy':
        np.savez(path.joinpath(f'batch{idx}'), **batch)
    elif file_format == 'hdf5':
        with h5py.File(path.joinpath(f'batch{idx}.h5'), 'w-') as f:
            for k, v in batch.items():
                f.create_dataset(k, data=v)
    elif file_format == 'single_hdf5':
        with h5py.File(path.joinpath(f'{path.name}_single.h5'), 'a') as f:
            f.create_group(f'batch{idx}')
            g = f[f'batch{idx}']
            for k, v in batch.items():
                g.create_dataset(k, data=v)


def create_batched_dataset(h5_path: Union[str, Path],
                           dest_path: Optional[Union[str, Path]] = None,
                           shuffle: bool = True,
                           shuffle_seed: Optional[int] = None,
                           file_format: str = 'hdf5',
                           include_properties: Optional[Sequence[str]] = ('species', 'coordinates', 'energies'),
                           batch_size: int = 2560,
                           max_batches_per_packet: int = 350,
                           padding: Optional[Dict[str, float]] = None,
                           splits: Optional[Dict[str, float]] = None,
                           folds: Optional[int] = None,
                           inplace_transform: Optional[Transform] = None,
                           verbose: bool = True) -> None:

    if folds is not None and splits is not None:
        raise ValueError('Only one of ["folds", "splits"] should be specified')

    # NOTE: All the tensor manipulation in this function is handled in CPU
    if file_format == 'single_hdf5':
        warnings.warn('Depending on the implementation, a single HDF5 file may'
                      'not support parallel reads, so using num_workers > 1 may'
                      'have a detrimental effect on performance, its probably better'
                      'to save in many hdf5 files with file_format=hdf5')
    if dest_path is None:
        dest_path = Path(f'./batched_dataset_{file_format}').resolve()
    dest_path = Path(dest_path).resolve()

    h5_path = Path(h5_path).resolve()
    if h5_path.is_dir():
        h5_datasets = [AniH5Dataset(p) for p in h5_path.iterdir() if p.suffix == '.h5']
    elif h5_path.is_file():
        h5_datasets = [AniH5Dataset(h5_path)]

    # (1) Get all indices and shuffle them if needed
    #
    # These are pairs of indices that index first the group and then the
    # specific conformer, it is possible to just use one index for
    # everything but this is simpler at the cost of slightly more memory.
    # First we get all group sizes for all datasets concatenated in a tensor, in the same
    # order as h5_datasets
    group_sizes_values = torch.cat([torch.tensor(list(h5ds.group_sizes.values()), dtype=torch.long) for h5ds in h5_datasets])
    conformer_indices = torch.cat([torch.stack((torch.full(size=(s.item(),), fill_value=j, dtype=torch.long),
                                     (torch.arange(0, s.item(), dtype=torch.long))), dim=-1)
                                     for j, s in enumerate(group_sizes_values)])

    rng = _get_random_generator(shuffle, shuffle_seed)

    conformer_indices = _maybe_shuffle_indices(conformer_indices, rng)

    # (2) Split shuffled indices according to requested dataset splits or folds
    # by defaults we use splits, if folds or splits is specified we
    # do the specified operation
    if folds is not None:
        conformer_splits, split_paths = _divide_into_folds(conformer_indices, dest_path, folds, rng)
    else:
        if splits is None:
            splits = {'training': 0.8, 'validation': 0.2}

        if not math.isclose(sum(list(splits.values())), 1.0):
            raise ValueError("The sum of the split fractions has to add up to one")

        conformer_splits, split_paths = _divide_into_splits(conformer_indices, dest_path, splits)

    # (3) Compute the batch indices for each split and save the conformers to disk
    _save_splits_into_batches(split_paths,
                              conformer_splits,
                              inplace_transform,
                              file_format,
                              include_properties,
                              h5_datasets,
                              padding,
                              batch_size,
                              max_batches_per_packet,
                              verbose)

    # log creation data
    creation_log = {'datetime_created': str(datetime.datetime.now()),
                    'source_path': h5_path.as_posix(),
                    'splits': splits,
                    'folds': folds,
                    'padding': utils.PADDING if padding is None else padding,
                    'is_inplace_transformed': inplace_transform is not None,
                    'shuffle': shuffle,
                    'include_properties': include_properties if include_properties is not None else 'all',
                    'batch_size': batch_size,
                    'total_num_conformers': len(conformer_indices),
                    'total_conformer_groups': len(group_sizes_values)}

    with open(dest_path.joinpath('creation_log.json'), 'w') as logfile:
        json.dump(creation_log, logfile, indent=1)


def _get_random_generator(shuffle: bool = False, shuffle_seed: Optional[int] = None) -> Optional[torch.Generator]:

    if shuffle_seed is not None:
        assert shuffle
        seed = shuffle_seed
    else:
        # non deterministic seed
        seed = torch.random.seed()

    rng: Optional[torch.Generator]

    if shuffle:
        rng = torch.random.manual_seed(seed)
    else:
        rng = None

    return rng


def _get_properties_size(molecule_group: Union[h5py.Group, Dict[str, ndarray], Dict[str, Tensor]],
                        flag_property: Optional[str] = None,
                        supported_properties: Optional[Set[str]] = None) -> int:
    if flag_property is not None:
        size = len(molecule_group[flag_property])
    else:
        assert supported_properties is not None
        if 'coordinates' in supported_properties:
            size = len(molecule_group['coordinates'])
        elif 'energies' in supported_properties:
            size = len(molecule_group['energies'])
        elif 'forces' in supported_properties:
            size = len(molecule_group['forces'])
        else:
            raise RuntimeError('Could not infer number of molecules in properties'
                               ' since "coordinates", "forces" and "energies" dont'
                               ' exist, please provide a key that holds an array/tensor with the'
                               ' molecule size as its first axis/dim')
    return size


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
                        rng: Optional[torch.Generator] = None) -> Tuple[Tuple[Tensor, ...], 'OrderedDict[str, Path]']:

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

    _create_split_paths(split_paths)

    return tuple(conformer_splits), split_paths


def _divide_into_splits(conformer_indices: Tensor,
                        dest_path: Path,
                        splits: Dict[str, float]) -> Tuple[Tuple[Tensor, ...], 'OrderedDict[str, Path]']:
    total_num_conformers = len(conformer_indices)
    split_sizes = OrderedDict([(k, int(total_num_conformers * v)) for k, v in splits.items()])
    split_paths = OrderedDict([(k, dest_path.joinpath(k)) for k in split_sizes.keys()])

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
                              h5_datasets: Sequence[AniH5Dataset],
                              padding: Optional[Dict[str, float]],
                              batch_size: int,
                              max_batches_per_packet: int,
                              verbose: bool) -> None:
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
    file_idxs_and_group_keys = [(j, k)
                  for j, h5ds in enumerate(h5_datasets)
                  for k in h5ds.group_sizes.keys()]

    use_pbar = PKBAR_INSTALLED and verbose
    for split_path, indices_of_split in zip(split_paths.values(), conformer_splits):
        all_batch_indices = torch.split(indices_of_split, batch_size)

        all_batch_indices_packets = [all_batch_indices[j:j + max_batches_per_packet]
                                    for j in range(0, len(all_batch_indices), max_batches_per_packet)]
        num_batch_indices_packets = len(all_batch_indices_packets)

        overall_batch_idx = 0
        for j, batch_indices_packet in enumerate(all_batch_indices_packets):
            num_batches_in_packet = len(batch_indices_packet)
            # Now first we cat and sort according to the first index in order to
            # fetch all conformers of the same group simultaneously
            batch_indices_cat = torch.cat(batch_indices_packet, 0)
            indices_to_sort_batch_indices_cat = torch.argsort(batch_indices_cat[:, 0])
            sorted_batch_indices_cat = batch_indices_cat[indices_to_sort_batch_indices_cat]
            uniqued_idxs_cat, counts_cat = torch.unique_consecutive(sorted_batch_indices_cat[:, 0],
                                                                    return_counts=True)
            cumcounts_cat = utils.cumsum_from_zero(counts_cat)

            # batch_sizes and indices_to_unsort are needed for the
            # reverse operation once the conformers have been
            # extracted
            batch_sizes = [len(batch_indices) for batch_indices in batch_indices_packet]
            indices_to_unsort_batch_cat = torch.argsort(indices_to_sort_batch_indices_cat)
            assert len(batch_sizes) <= max_batches_per_packet

            if use_pbar:
                pbar = pkbar.Pbar(f'=> Saving batch packet {j + 1} of {num_batch_indices_packets}'
                                  f' of split {split_path.name},'
                                  f' in format {file_format}', len(counts_cat))

            all_conformers: List[Dict[str, Tensor]] = []
            for step, (group_idx, count, start_index) in enumerate(zip(uniqued_idxs_cat, counts_cat, cumcounts_cat)):
                # select the specific group from the whole list of files
                file_idx, group_key = file_idxs_and_group_keys[group_idx.item()]

                # get a slice with the indices to extract the necessary
                # conformers from the group for all batches in pack.
                end_index = start_index + count
                selected_indices = sorted_batch_indices_cat[start_index:end_index, 1]

                # Important: to prevent possible bugs / errors, that may happen
                # due to incorrect conversion to indices, species is **always*
                # converted to atomic numbers when saving the batched dataset.
                numpy_conformers = h5_datasets[file_idx].get_conformers(group_key,
                                                                  selected_indices.cpu().numpy(),
                                                                  include_properties, raw_output=False)
                all_conformers.append({k: torch.as_tensor(v) for k, v in numpy_conformers.items()})
                if use_pbar:
                    pbar.update(step)

            batches_cat = utils.pad_atomic_properties(all_conformers, padding)
            # Now we need to reassign the conformers to the specified
            # batches. Since to get here we cat'ed and sorted, to
            # reassign we need to unsort and split.
            # The format of this is {'species': (batch1, batch2, ...), 'coordinates': (batch1, batch2, ...)}
            batch_packet_dict = {k: torch.split(t[indices_to_unsort_batch_cat], batch_sizes)
                                 for k, t in batches_cat.items()}

            for packet_batch_idx in range(num_batches_in_packet):
                batch = {k: v[packet_batch_idx] for k, v in batch_packet_dict.items()}
                batch = inplace_transform(batch)
                _save_batch(split_path, overall_batch_idx, batch, file_format)
                overall_batch_idx += 1
