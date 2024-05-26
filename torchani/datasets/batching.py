r"""Functions for creating batched datasets"""
import typing as tp
import warnings
import math
import json
import datetime
from pathlib import Path
from collections import OrderedDict

import h5py
import torch
from torch import Tensor
from tqdm import tqdm

from torchani.utils import pad_atomic_properties, cumsum_from_zero, PADDING
from torchani.datasets.datasets import ANIDataset
from torchani.transforms import Transform, Identity
from torchani.datasets._annotations import Conformers, StrPath


class ANIBatchedDataset(torch.utils.data.Dataset[Conformers]):
    _batch_paths: tp.Optional[tp.List[Path]]

    def __init__(
        self,
        store_dir: tp.Optional[StrPath] = None,
        batches: tp.Optional[tp.List[Conformers]] = None,
        file_format: tp.Optional[str] = None,
        split: str = "training",
        transform: tp.Optional[Transform] = None,
        properties: tp.Optional[tp.Sequence[str]] = None,
        drop_last: bool = False,
    ):
        # (store_dir or file_format or transform) and batches are mutually
        # exclusive options, batches is passed if the dataset directly lives in
        # memory and has no backing store, otherwise there should be a backing
        # store in store_dir/split
        if batches is not None and any(
            v is not None for v in (file_format, store_dir, transform)
        ):
            raise ValueError(
                "Batches is mutually exclusive with file_format/store_dir/transform"
            )
        self.split = split
        self.properties = properties
        self.transform = Identity() if transform is None else transform
        container: tp.Union[tp.List[Path], tp.List[Conformers]]
        if not batches:
            if store_dir is None:
                raise ValueError("One of batches or store_dir must be specified")
            store_dir = Path(store_dir).resolve()
            self._batch_paths = self._get_batch_paths(store_dir / split)
            self._extractor = self._hdf5_extractor
            container = self._batch_paths
        else:
            self._data = batches
            self._batch_paths = None
            self._extractor = self._memory_extractor
            container = self._data
        # Drops last batch only if requested and if its smaller than the rest
        if drop_last and self.batch_size(-1) < self.batch_size(0):
            container.pop()
        self._len = len(container)

    def _memory_extractor(self, idx: int) -> Conformers:
        return self._data[idx]

    def batch_size(self, idx: int) -> int:
        batch = self[idx]
        return batch[next(iter(batch.keys()))].shape[0]

    def _get_batch_paths(self, batches_dir: Path) -> tp.List[Path]:
        # We assume batch names are prefixed by a zero-filled number so that
        # sorting alphabetically sorts batch numbers
        try:
            batch_paths = sorted(batches_dir.iterdir())
            first_batch = batch_paths[0]
            # notadirectory error is handled
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The dir {batches_dir.parent.as_posix()} exists,"
                f" but the split {batches_dir.as_posix()} does not"
            ) from None
        except IndexError:
            raise FileNotFoundError(
                f"The dir {batches_dir.as_posix()} has no files"
            ) from None

        if any(f.suffix != first_batch.suffix for f in batch_paths):
            raise RuntimeError(
                f"Files with different extensions found in {batches_dir.as_posix()}"
            )

        if any(f.is_dir() for f in batch_paths):
            raise RuntimeError(f"Subdirectories found in {batches_dir.as_posix()}")
        return batch_paths

    def _hdf5_extractor(self, idx: int) -> Conformers:
        with h5py.File(self._batch_paths[idx], "r") as f:  # type: ignore
            return {
                k: torch.as_tensor(v[()])
                for k, v in f["/"].items()
                if self.properties is None or k in self.properties
            }

    def cache(
        self,
        pin_memory: bool = True,
        verbose: bool = True,
    ) -> "ANIBatchedDataset":
        r"""Saves the full dataset into RAM"""
        desc = f"Cacheing {self.split}, Warning: this may use a lot of RAM!"
        self._data = [
            self._extractor(idx)
            for idx in tqdm(
                range(len(self)),
                total=len(self),
                disable=not verbose,
                desc=desc,
                leave=False,
            )
        ]
        desc = "Applying transforms once and discarding"
        with torch.no_grad():
            self._data = [
                self.transform(p)
                for p in tqdm(
                    self._data,
                    total=len(self),
                    disable=not verbose,
                    desc=desc,
                    leave=False,
                )
            ]
            self.transform = Identity()
        if pin_memory:
            desc = "Pinning memory; don't pin memory in torch DataLoader!"
            self._data = [
                {k: v.pin_memory() for k, v in batch.items()}
                for batch in tqdm(
                    self._data,
                    total=len(self),
                    disable=not verbose,
                    desc=desc,
                    leave=False,
                )
            ]
        self._extractor = self._memory_extractor
        return self

    def __getitem__(self, idx: int) -> Conformers:
        # integral indices must be provided for compatibility with pytorch
        # DataLoader API
        batch = self._extractor(idx)
        with torch.no_grad():
            batch = self.transform(batch)
        return batch

    def __len__(self) -> int:
        return self._len


# TODO a batcher class would make this code much more clear
def create_batched_dataset(
    locations: tp.Union[tp.Collection[StrPath], StrPath, ANIDataset],
    dest_path: tp.Optional[StrPath] = None,
    shuffle: bool = True,
    shuffle_seed: tp.Optional[int] = None,
    properties: tp.Iterable[str] = (),
    batch_size: int = 2560,
    max_batches_per_packet: int = 350,
    padding: tp.Optional[tp.Dict[str, float]] = None,
    splits: tp.Optional[tp.Dict[str, float]] = None,
    folds: tp.Optional[int] = None,
    transform: tp.Optional[Transform] = None,
    direct_cache: bool = False,
    verbose: bool = True,
) -> tp.Dict[str, ANIBatchedDataset]:
    if folds is not None and splits is not None:
        raise ValueError('Only one of ["folds", "splits"] should be specified')

    if direct_cache and dest_path is not None:
        raise ValueError("Destination path not needed for direct cache")
    transform = Identity() if transform is None else transform

    # NOTE: All the tensor manipulation in this function is handled in CPU
    if dest_path is None:
        dest_path = Path.cwd() / "batched_dataset_hdf5"
    else:
        dest_path = Path(dest_path).resolve()

    if isinstance(locations, ANIDataset):
        dataset = locations
    else:
        dataset = ANIDataset(locations)

    if isinstance(properties, str):
        properties = (properties,)
    if not properties:
        properties = tuple(dataset.tensor_properties)
    properties = tuple(sorted(properties))
    padding = PADDING if padding is None else padding
    assert padding is not None  # mypy

    # (1) Get all indices and shuffle them if needed
    #
    # These are pairs of indices that index first the group and then the
    # specific conformer, it is possible to just use one index for
    # everything but this is simpler at the cost of slightly more memory.
    # First we get all group sizes for all datasets concatenated in a tensor,
    # in the same
    # order as h5_map
    group_sizes_values = torch.tensor(
        tuple(dataset.group_sizes.values()), dtype=torch.long
    )
    conformer_indices = torch.cat(
        [
            torch.stack(
                (
                    torch.full(size=(s.item(),), fill_value=j, dtype=torch.long),
                    (torch.arange(0, s.item(), dtype=torch.long)),
                ),
                dim=-1,
            )
            for j, s in enumerate(group_sizes_values)
        ]
    )
    rng = _get_random_generator(shuffle, shuffle_seed)
    conformer_indices = _maybe_shuffle_indices(conformer_indices, rng)

    # (2) Split shuffled indices according to requested dataset splits or folds
    # by defaults we use splits, if folds or splits is specified we
    # do the specified operation
    if folds is not None:
        conformer_splits, split_paths = _divide_into_folds(
            conformer_indices,
            dest_path,
            folds,
            rng,
            direct_cache,
            verbose,
        )
    else:
        if splits is None:
            splits = {"training": 0.8, "validation": 0.2}

        if not math.isclose(sum(list(splits.values())), 1.0):
            raise ValueError("The sum of the split fractions has to add up to one")

        conformer_splits, split_paths = _divide_into_splits(
            conformer_indices,
            dest_path,
            splits,
            direct_cache,
            verbose,
        )

    # (3) Compute the batch indices for each split and save the conformers to disk
    batched_datasets = _save_splits_into_batches(
        split_paths,
        conformer_splits,
        transform,
        properties,
        dataset,
        padding,
        batch_size,
        max_batches_per_packet,
        direct_cache,
        verbose,
    )
    # log creation data
    if not direct_cache:
        try:
            symbols = dataset.symbols
        except ValueError:
            symbols = ("?",)  # legacy grouping, symbols can't be determined
        creation_log = {
            "datetime_created": str(datetime.datetime.now()),
            "splits": splits,
            "folds": folds,
            "padding": padding,
            "shuffle": shuffle,
            "shuffle_seed": shuffle_seed,
            "properties": properties,
            "batch_size": batch_size,
            "store_locations": dataset.store_locations,
            "symbols": symbols,
            "num_conformers": dataset.num_conformers,
        }
        with open(dest_path.joinpath("creation_log.json"), "w") as logfile:
            json.dump(creation_log, logfile, indent=1)
    return batched_datasets


def _get_random_generator(
    shuffle: bool = False, shuffle_seed: tp.Optional[int] = None
) -> tp.Optional[torch.Generator]:
    if shuffle_seed is not None:
        assert shuffle
        seed = shuffle_seed
    else:
        seed = torch.random.seed()

    if shuffle:
        return torch.random.manual_seed(seed)
    return None


def _maybe_shuffle_indices(
    conformer_indices: Tensor, rng: tp.Optional[torch.Generator] = None
) -> Tensor:
    total_num_conformers = len(conformer_indices)
    if rng is not None:
        shuffle_indices = torch.randperm(total_num_conformers, generator=rng)
        conformer_indices = conformer_indices[shuffle_indices]
    else:
        warnings.warn(
            "Dataset will not be shuffled, this should only be used for debugging"
        )
    return conformer_indices


def _divide_into_folds(
    conformer_indices: Tensor,
    dest_path: Path,
    folds: int,
    rng: tp.Optional[torch.Generator] = None,
    direct_cache: bool = False,
    verbose: bool = True,
) -> tp.Tuple[tp.Tuple[Tensor, ...], tp.OrderedDict[str, Path]]:
    # the idea here is to work with "blocks" of size num_conformers / folds
    # cast to list for mypy
    conformer_blocks = list(torch.chunk(conformer_indices, folds))
    conformer_splits: tp.List[Tensor] = []
    split_paths_list: tp.List[tp.Tuple[str, Path]] = []
    if verbose:
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
        split_paths_list.extend(
            [
                (f"training{f}", dest_path.joinpath(f"training{f}")),
                (f"validation{f}", dest_path.joinpath(f"validation{f}")),
            ]
        )
    split_paths = OrderedDict(split_paths_list)
    if not direct_cache:
        _create_split_paths(split_paths)

    return tuple(conformer_splits), split_paths


def _divide_into_splits(
    conformer_indices: Tensor,
    dest_path: Path,
    splits: tp.Dict[str, float],
    direct_cache: bool = False,
    verbose: bool = True,
) -> tp.Tuple[tp.Tuple[Tensor, ...], tp.OrderedDict[str, Path]]:
    total_num_conformers = len(conformer_indices)
    split_sizes = OrderedDict(
        [(k, int(total_num_conformers * v)) for k, v in splits.items()]
    )
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
    # TODO: Unnecessary cast in current pytorch
    conformer_splits = tuple(torch.split(conformer_indices, list(split_sizes.values())))
    assert len(conformer_splits) == len(split_sizes.values())
    if verbose:
        print(
            f"Splits have number of conformers: {dict(split_sizes)}."
            f" The requested percentages were: {splits}"
        )
    return conformer_splits, split_paths


def _create_split_paths(split_paths: tp.OrderedDict[str, Path]) -> None:
    for p in split_paths.values():
        if p.is_dir():
            subdirs = [d for d in p.iterdir()]
            if subdirs:
                raise ValueError(
                    "The dest_path provided already has files"
                    " or directories, please provide"
                    " a different path"
                )
        else:
            if p.is_file():
                raise ValueError("The dest_path is a file, it should be a directory")
            p.mkdir(parents=True)


def _save_splits_into_batches(
    split_paths: tp.OrderedDict[str, Path],
    conformer_splits: tp.Tuple[Tensor, ...],
    transform: Transform,
    properties: tp.Sequence[str],
    dataset: ANIDataset,
    padding: tp.Dict[str, float],
    batch_size: int,
    max_batches_per_packet: int,
    direct_cache: bool,
    verbose: bool,
) -> tp.Dict[str, ANIBatchedDataset]:
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

    # get all group keys concatenated in a list, with the associated file indexes
    key_list = list(dataset.keys())

    # Important: to prevent possible bugs / errors, that may happen
    # due to incorrect conversion to indices, species is **always*
    # converted to atomic numbers when saving the batched dataset.
    batched_datasets: tp.Dict[str, ANIBatchedDataset] = dict()
    with dataset.keep_open() as ro_dataset:
        for (split_name, split_path), indices_of_split in zip(
            split_paths.items(), conformer_splits
        ):
            all_batch_indices = torch.split(indices_of_split, batch_size)

            all_batch_indices_packets = [
                all_batch_indices[j:j + max_batches_per_packet]
                for j in range(0, len(all_batch_indices), max_batches_per_packet)
            ]
            num_batch_indices_packets = len(all_batch_indices_packets)

            overall_batch_idx = 0
            if direct_cache:
                in_memory_batches: tp.List[Conformers] = []
            for j, batch_indices_packet in enumerate(all_batch_indices_packets):
                num_batches_in_packet = len(batch_indices_packet)
                # Now first we cat and sort according to the first index in order to
                # fetch all conformers of the same group simultaneously
                batch_indices_cat = torch.cat(batch_indices_packet, 0)
                indices_to_sort_batch_indices_cat = torch.argsort(
                    batch_indices_cat[:, 0]
                )
                sorted_batch_indices_cat = batch_indices_cat[
                    indices_to_sort_batch_indices_cat
                ]
                uniqued_idxs_cat, counts_cat = torch.unique_consecutive(
                    sorted_batch_indices_cat[:, 0], return_counts=True
                )
                cumcounts_cat = cumsum_from_zero(counts_cat)

                # batch_sizes and indices_to_unsort are needed for the
                # reverse operation once the conformers have been
                # extracted
                batch_sizes = [
                    len(batch_indices) for batch_indices in batch_indices_packet
                ]
                indices_to_unsort_batch_cat = torch.argsort(
                    indices_to_sort_batch_indices_cat
                )
                assert len(batch_sizes) <= max_batches_per_packet

                all_conformers: tp.List[Conformers] = []
                end_idxs = counts_cat + cumcounts_cat
                groups_slices = zip(uniqued_idxs_cat, cumcounts_cat, end_idxs)
                if direct_cache:
                    desc = (
                        f"Saving batch packet {j + 1} of {num_batch_indices_packets} "
                        f"of split {split_path.name} into memory"
                    )
                else:
                    desc = (
                        f"Saving batch packet {j + 1} of {num_batch_indices_packets} "
                        f"of split {split_path.name}"
                    )
                for step, group_slice in tqdm(
                    enumerate(groups_slices),
                    total=len(counts_cat),
                    desc=desc,
                    disable=not verbose,
                    leave=False,
                ):
                    group_idx, start, end = group_slice
                    # select the specific group from the whole list of files
                    # and get a slice with the indices to extract the necessary
                    # conformers from the group for all batches in pack.
                    selected_indices = sorted_batch_indices_cat[start:end, 1]
                    assert selected_indices.dim() == 1
                    conformers = ro_dataset.get_conformers(
                        key_list[group_idx.item()],
                        selected_indices,
                        properties=properties,
                    )
                    all_conformers.append(conformers)
                batches_cat = pad_atomic_properties(all_conformers, padding)
                # Now we need to reassign the conformers to the specified
                # batches. Since to get here we cat'ed and sorted, to
                # reassign we need to unsort and split.
                # The format of this is {'species': (batch1, batch2, ...),
                # 'coordinates': (batch1, batch2, ...)}
                batch_packet_dict = {
                    k: torch.split(t[indices_to_unsort_batch_cat], batch_sizes)
                    for k, t in batches_cat.items()
                }
                for packet_batch_idx in range(num_batches_in_packet):
                    batch = {
                        k: v[packet_batch_idx] for k, v in batch_packet_dict.items()
                    }
                    batch = transform(batch)
                    if direct_cache:
                        in_memory_batches.append(batch)
                    else:
                        _save_batch(
                            split_path,
                            overall_batch_idx,
                            batch,
                            len(all_batch_indices),
                        )
                    overall_batch_idx += 1
            if direct_cache:
                split_ds = ANIBatchedDataset(
                    batches=in_memory_batches, split=split_name
                )
                split_ds = split_ds.cache(
                    verbose=False, pin_memory=torch.cuda.is_available()
                )
                batched_datasets[split_name] = split_ds
            else:
                batched_datasets[split_name] = ANIBatchedDataset(
                    store_dir=split_path.parent, split=split_name
                )
    return batched_datasets


def _save_batch(path: Path, idx: int, batch: Conformers, total_batches: int) -> None:
    batch = {k: v.numpy() for k, v in batch.items()}
    # The batch names are e.g. 00034_batch.h5
    batch_path = path / f"{str(idx).zfill(len(str(total_batches)))}_batch"
    with h5py.File(batch_path.with_suffix(".h5"), "w-") as f:
        for k, v in batch.items():
            f.create_dataset(k, data=v)
