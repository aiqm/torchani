import os
from dataclasses import dataclass
import typing_extensions as tpx
import typing as tp
import math
import json
import datetime
from pathlib import Path

import h5py
import torch
from torch import Tensor
from tqdm import tqdm

from torchani.paths import datasets_dir
from torchani.annotations import Conformers, StrPath
from torchani.utils import pad_atomic_properties, strip_redundant_padding, PADDING
from torchani.transforms import Transform, identity
from torchani.datasets.anidataset import ANIDataset


_T = tp.TypeVar("_T")


class BatchedDataset(torch.utils.data.Dataset[Conformers]):
    split: str
    transform: Transform

    # Explicit implementation of __iter__ to avoid relying on python's legacy behavior
    def __iter__(self) -> tp.Iterator[Conformers]:
        j = 0
        while True:
            try:
                yield self[j]
                j += 1
            except IndexError:
                break

    @staticmethod
    def _batch_size(batch: tp.Mapping[str, Tensor]) -> int:
        return batch[next(iter(batch.keys()))].shape[0]

    def cache(self, verbose: bool = True, pin_memory: bool = False) -> "BatchedDataset":
        return self

    def as_dataloader(
        self,
        num_workers: tp.Optional[int] = None,
        pin_memory: tp.Optional[bool] = None,
        prefetch_factor: tp.Optional[int] = None,
        shuffle: bool = True,
    ) -> torch.utils.data.DataLoader:
        if prefetch_factor is None:
            prefetch_factor = 2
        if num_workers is None:
            num_workers = len(os.sched_getaffinity(0)) - 1
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        return torch.utils.data.DataLoader(
            self,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            shuffle=shuffle,
            batch_size=None,
        )

    def __len__(self) -> int:
        return 0

    def _limit_batches(
        self, batches: tp.List[_T], limit: tp.Union[int, float]
    ) -> tp.List[_T]:
        if isinstance(limit, float):
            if not (0.0 <= limit <= 1.0):
                raise ValueError("limit must lie in (0.0, 1.0)")
            batches = batches[: int(limit * len(batches))]
        elif isinstance(limit, int):
            if not (0 <= limit <= len(batches)):
                raise ValueError("limit must lie in (0, num_batches)")
            batches = batches[: limit]
        return batches


class ANIBatchedInMemoryDataset(BatchedDataset):
    r"""
    This dataset does not support multiprocessing or pin_memory=True in
    dataloader (num_workers>0)
    """

    def __init__(
        self,
        batches: tp.Sequence[Conformers],
        transform: Transform = identity,
        limit: tp.Union[int, float] = 1.0,
        split: str = "division",
        drop_last: bool = False,
    ) -> None:
        self.div = split
        batches = list(batches)
        self.transform = transform
        if drop_last and self._batch_size(batches[0]) > self._batch_size(batches[-1]):
            batches.pop()
        self._batches = self._limit_batches(batches, limit)

    def pin_memory(self, verbose: bool = True) -> None:
        if verbose:
            print("Pinning memory ...")
        self._batches = [
            {k: v.pin_memory() for k, v in batch.items()} for batch in self._batches
        ]

    def as_dataloader(
        self,
        num_workers: tp.Optional[int] = None,
        pin_memory: tp.Optional[bool] = None,
        prefetch_factor: tp.Optional[int] = None,
        shuffle: bool = True,
    ) -> torch.utils.data.DataLoader:
        if num_workers not in (None, 0) or prefetch_factor is not None:
            raise ValueError("multiprocessing not supported for in-memory datasets")
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        if pin_memory:
            self.pin_memory(verbose=False)
        return torch.utils.data.DataLoader(
            self,
            num_workers=0,
            pin_memory=False,
            shuffle=shuffle,
            batch_size=None,
        )

    def __getitem__(self, idx: int) -> Conformers:
        with torch.no_grad():
            batch = self.transform(self._batches[idx])
        return batch

    def __len__(self) -> int:
        return len(self._batches)


# NOTE: This is suceptible to allocating a lot of memory if multiporcessing,
# since it stores batch patchs in a list, but as long as the number of batches
# is not huge it should not be an issue
class ANIBatchedDataset(BatchedDataset):
    def __init__(
        self,
        store_dir: StrPath,
        split: str = "division",
        transform: Transform = identity,
        limit: tp.Union[int, float] = 1.0,
        properties: tp.Sequence[str] = (),
        drop_last: bool = False,
    ):
        self.div = split
        self.properties = properties
        self.transform = transform
        self._batch_paths = self._get_batch_paths(Path(store_dir).resolve() / split)

        if drop_last and self._batch_size(self[0]) > self._batch_size(self[-1]):
            self._batch_paths.pop()
        self._batch_paths = self._limit_batches(self._batch_paths, limit)

    def _get_batch_paths(self, batches_dir: Path) -> tp.List[Path]:
        # We assume batch names are prefixed by a zero-filled number so that
        # sorting alphabetically sorts batch numbers
        try:
            batch_paths = sorted(batches_dir.iterdir())
            first_batch = batch_paths[0]
            # notadirectory error is handled
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The dir {str(batches_dir.parent)} exists,"
                f" but the division {str(batches_dir)} does not"
            ) from None
        except IndexError:
            raise FileNotFoundError(
                f"The dir {str(batches_dir)} has no files"
            ) from None

        if any(f.suffix != first_batch.suffix for f in batch_paths):
            raise RuntimeError(
                f"Files with different extensions found in {str(batches_dir)}"
            )

        if any(f.is_dir() for f in batch_paths):
            raise RuntimeError(f"Subdirectories found in {str(batches_dir)}")
        return batch_paths

    def cache(
        self,
        verbose: bool = True,
        pin_memory: tp.Optional[bool] = None,
    ) -> ANIBatchedInMemoryDataset:
        r"""Saves the full dataset into RAM"""
        desc = f"Cacheing {self.div}, Warning: this may use a lot of RAM!"
        batches = [
            self._extract_batch(idx)
            for idx in tqdm(
                range(len(self)),
                total=len(self),
                disable=not verbose,
                desc=desc,
                leave=False,
            )
        ]
        ds = ANIBatchedInMemoryDataset(batches, self.transform)
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        if pin_memory:
            ds.pin_memory(verbose=verbose)
        return ds

    def _extract_batch(self, idx: int) -> Conformers:
        with h5py.File(self._batch_paths[idx], "r") as f:
            batch = {
                k: torch.from_numpy(v[()])
                for k, v in f["/"].items()
                if not self.properties or k in self.properties
            }
        return batch

    def __getitem__(self, idx: int) -> Conformers:
        with torch.no_grad():
            batch = self.transform(self._extract_batch(idx))
        return batch

    def __len__(self) -> int:
        return len(self._batch_paths)


@dataclass
class Div:
    name: str
    indices: Tensor  # dtype=torch.long, shape=(num-conformers, 2)
    path: Path


class Batcher:
    def __init__(
        self,
        dest_root: tp.Union[Path, tp.Literal["ram"], tp.Literal["default"]] = "default",
        max_batches_per_packet: int = 200,
        verbose: bool = False,
    ) -> None:
        self.max_batches_per_packet = max_batches_per_packet
        self.verbose = verbose
        if dest_root == "default":
            dest_root = datasets_dir()
        self.store_on_disk = dest_root != "ram"
        if not self.store_on_disk:
            if max_batches_per_packet != 200:
                raise ValueError(
                    "max_batches_per_packet can't be provided if saving in ram"
                )
            max_batches_per_packet = 2**31  # arbitrarily high number
        self._dest_root = Path("/tmp") if dest_root == "ram" else dest_root
        self._shuffle = True

    # Used only for debugging
    def _no_shuffle(self) -> None:
        self._shuffle = False

    @classmethod
    def in_ram(cls, verbose: bool = False) -> tpx.Self:
        return cls("ram", verbose=verbose)

    def divide_and_batch(
        self,
        src: tp.Union[tp.Iterable[StrPath], StrPath, ANIDataset],
        dest_dir: str = "",
        # Dataset modifications and options
        splits: tp.Optional[tp.Dict[str, float]] = None,
        folds: tp.Optional[int] = None,
        batch_size: int = 2560,
        padding: tp.Optional[tp.Dict[str, float]] = None,
        transform: Transform = identity,
        properties: tp.Iterable[str] = (),
        # rng seeds
        divs_seed: tp.Optional[int] = None,
        batch_seed: tp.Optional[int] = None,
    ) -> tp.Dict[str, BatchedDataset]:
        padding = PADDING if padding is None else padding
        if (not self.store_on_disk) and dest_dir:
            raise ValueError("dest_dir can't be passed if saving in ram")

        if not dest_dir:
            dest_dir = "batched_dataset"

        dest_path = self._dest_root / dest_dir

        if (not self._shuffle) and (
            (divs_seed is not None) or (batch_seed is not None)
        ):
            raise ValueError("Seeds must be None if not shuffling")
        dataset = src if isinstance(src, ANIDataset) else ANIDataset(src)

        if not properties:
            properties = dataset.tensor_properties
        elif isinstance(properties, str):
            properties = [properties]
        else:
            properties = sorted(set(properties))

        if splits is None and folds is None:
            splits = {"training": 1.0}
        if splits is not None and folds is not None:
            raise ValueError("'splits' and 'folds' can't be simultaneously specified")

        divs_rng = torch.Generator()
        if divs_seed is None:
            divs_seed = divs_rng.seed()
        else:
            divs_rng.manual_seed(divs_seed)

        batch_rng = torch.Generator()
        if batch_seed is None:
            batch_seed = batch_rng.seed()
        else:
            batch_rng.manual_seed(batch_seed)

        # (1) Get all indices and shuffle them
        #
        # These are pairs of indices that index first the group and then the
        # specific conformer, it is possible to just use one index for
        # everything but this is simpler at the cost of slightly more memory.
        #
        # First get all group sizes for all datasets concatenated in a tensor,
        # in the same
        # order as h5_map
        sizes = torch.tensor(tuple(dataset.group_sizes.values()), dtype=torch.long)
        # conformer_idxs holds pairs [group-idx, conformer-idx]
        pad_idxs = torch.arange(max(sizes), dtype=torch.long).repeat(len(sizes), 1)
        mask = pad_idxs < sizes.view(-1, 1)
        indices = torch.masked_select(pad_idxs, mask).view(-1, 1)
        repeat_sizes = torch.repeat_interleave(sizes).view(-1, 1)
        # shape (num_conformers, 2)
        conformer_idxs = torch.cat((repeat_sizes, indices), dim=1)
        # Shuffle divisions
        if self._shuffle:
            shuffle_idxs = torch.randperm(dataset.num_conformers, generator=divs_rng)
            conformer_idxs = conformer_idxs[shuffle_idxs]

        if splits is not None:
            if not math.isclose(sum(list(splits.values())), 1.0):
                raise ValueError("The sum of the split fractions must add up to one")
            divs = self._divide_in_splits(conformer_idxs, splits, dest_path)
        else:
            assert folds is not None  # mypy
            if not folds > 1:
                raise ValueError("Folds must be an integer > 1")
            divs = self._divide_in_folds(conformer_idxs, folds, dest_path)

        if self.store_on_disk:
            for div in divs:
                div.path.mkdir(parents=True, exist_ok=False)

        if self.verbose:
            print("Divisions will have sizes:")
            for div in divs:
                print(f"    {div.name}: {len(div.indices)}")

        # Shuffle batches inside divisions
        # NOTE: Each division gets a different make-up of batches, so symlinking
        # is not possible
        if self._shuffle:
            for div in divs:
                shuffle_idxs = torch.randperm(len(div.indices), generator=batch_rng)
                div.indices = div.indices[shuffle_idxs]

        batched_datasets = self._batch_and_save_divisions(
            dest_path,
            dataset,
            divs,
            batch_size,
            padding,
            transform,
            properties,
        )

        if self.store_on_disk:
            self._log_creation_data(
                dest_path,
                dataset,
                batch_size,
                divs_seed,
                batch_seed,
                padding,
                splits,
                folds,
                properties,
            )

        return batched_datasets

    def _divide_in_splits(
        self,
        conformer_idxs: Tensor,
        splits: tp.Dict[str, float],
        dest_path: Path,
    ) -> tp.List[Div]:
        if self.verbose:
            print(f"Dividing dataset in splits with fractions {splits}")

        # Sort alphabetically and divide into "names" and "sizes"
        num_conformers = len(conformer_idxs)
        split_names: tp.List[str] = []
        split_sizes: tp.List[int] = []
        for k, v in sorted(splits.items(), key=lambda kv: kv[0]):
            split_names.append(k)
            split_sizes.append(int(v * num_conformers))

        # Slightly modify the first split if the fractions don't span the
        # dataset perfectly. This automatically takes care of the cases
        # leftover > 0 and leftover < 0
        leftover = num_conformers - sum(split_sizes)
        if leftover != 0:
            split_sizes[0] += leftover

        assert sum(split_sizes) == num_conformers

        conformer_splits = torch.split(conformer_idxs, split_sizes)
        divs: tp.List[Div] = []
        for name, idxs in zip(split_names, conformer_splits):
            divs.append(Div(name=name, indices=idxs, path=dest_path / name))
        return divs

    def _divide_in_folds(
        self,
        conformer_idxs: Tensor,
        folds: int,
        dest_path: Path,
    ) -> tp.List[Div]:
        if self.verbose:
            print(f"Dividing dataset in {folds} folds (for CV or ensemble training)")

        # First divide into "blocks" of shape (num_conformers / folds, 2)
        # For the ith fold take the ith block (always a different one) and assign it
        # to division "ith-validation", and the rest to "ith-training".
        divs: tp.List[Div] = []
        conformer_blocks = torch.chunk(conformer_idxs, folds)
        for i in range(folds):
            ith_valid_div = conformer_blocks[i]
            ith_train_div = torch.cat(conformer_blocks[:i] + conformer_blocks[i + 1:])
            train = f"training{i}"
            valid = f"validation{i}"
            divs.extend(
                (
                    Div(name=train, indices=ith_train_div, path=dest_path / train),
                    Div(name=valid, indices=ith_valid_div, path=dest_path / valid),
                )
            )
        return divs

    # Select some batches (a "packet"), scan the dataset to find the conformers
    # needed to create them and keep them in RAM. Save the batches in the
    # packet, and fetch the next packet.
    #
    # It is important to select a good number of batches-per-packet
    # in order to batch the dataset fast and with an acceptable RAM cost, since:
    #
    # - Doing this for 1-batch-per-packet is very slow. The number of slow
    # reads to the files on disk is approx batches * conformer_groups * 3
    # For example for the ANI1x dataset this is approx 9M reads.
    #
    # - Doing it for all batches at the same time is (reasonably) fast, ~9000
    # reads, but it requires keeping a huge part of the dataset in memory,
    # which is not feasible for large datasets.
    def _batch_and_save_divisions(
        self,
        dest_path: Path,
        dataset: ANIDataset,
        divs: tp.List[Div],
        batch_size: int,
        padding: tp.Dict[str, float],
        transform: Transform,
        properties: tp.Sequence[str],
    ) -> tp.Dict[str, BatchedDataset]:
        group_names = list(dataset.keys())
        batched_datasets: tp.Dict[str, BatchedDataset] = dict()
        with dataset.keep_open() as readonly_ds:
            for div in divs:
                # Attach a batch index to each batch of the split div indices
                # Each batch has shape (batch_size, 3)
                # where 0: group_idx, 1: conformer_idx in the group, 2: batch_idx
                batches: tp.List[Tensor] = []
                for j, b in enumerate(torch.split(div.indices, batch_size)):
                    batches.append(
                        torch.cat(
                            (b, torch.full((b.shape[0], 1), j, dtype=torch.long)), dim=1
                        )
                    )
                num_batches = len(batches)

                # Combine batches into packets.
                # Each packet has shape (packet_size, 3)
                # where 0: group_idx, 1: conformer_idx in the group, 2: batch_idx
                packets: tp.List[Tensor] = []
                step = self.max_batches_per_packet
                for j in range(0, num_batches, step):
                    packets.append(
                        torch.cat(
                            batches[j:j + step],
                            dim=0,
                        )
                    )
                num_packets = len(packets)

                in_memory_batches: tp.List[Conformers] = []
                for i, packet in enumerate(packets):
                    packet_unique_group_idxs = torch.unique(packet[:, 0])
                    packet_unique_batch_idxs = torch.unique(packet[:, 2])

                    packet_conformers_list: tp.List[Conformers] = []
                    packet_batch_idx_list: tp.List[Tensor] = []
                    for group_idx in tqdm(
                        packet_unique_group_idxs,
                        desc=f"{div.name}: Collecting packet {i + 1}/{num_packets}",
                        disable=not self.verbose,
                        leave=False,
                        total=len(packet_unique_group_idxs),
                    ):
                        conformer_is_in_packet = packet[:, 0] == group_idx
                        conformers = readonly_ds.get_conformers(
                            group_names[group_idx.item()],
                            packet[conformer_is_in_packet, 1],
                            properties=properties,
                        )
                        packet_batch_idx_list.append(packet[conformer_is_in_packet, 2])
                        packet_conformers_list.append(conformers)

                    # Dict of properties, each of shape (packet_size, ...)
                    packet_conformers = pad_atomic_properties(
                        packet_conformers_list, padding
                    )
                    #  packet_batch_idxs is the same as packet[:, 2] but sorted
                    #  in the group-idx order instead of batch-idx order. shape
                    #  (packet_size,) This is useful to index the
                    #  packet_conformers, since they are fetched from the
                    #  dataset in the group-idx order, in order to fetch all
                    #  conformers of the same group at the same time.
                    packet_batch_idxs = torch.cat(packet_batch_idx_list, dim=0)
                    for batch_idx in tqdm(
                        packet_unique_batch_idxs,
                        desc=f"{div.name}: Saving packet {i + 1}/{num_packets}",
                        disable=not self.verbose,
                        leave=False,
                        total=len(packet_unique_batch_idxs),
                    ):
                        batch = {
                            k: v[packet_batch_idxs == batch_idx]
                            for k, v in packet_conformers.items()
                        }
                        batch = strip_redundant_padding(batch)
                        batch = transform(batch)
                        if self.store_on_disk:
                            # The batch file names are e.g. 00034_batch.h5
                            max_digits = len(str(num_batches))
                            pre = str(batch_idx.item()).zfill(max_digits)
                            with h5py.File(
                                (dest_path / div.name) / f"{pre}_batch.h5", "w-"
                            ) as f:
                                for k, v in batch.items():
                                    f.create_dataset(k, data=v.numpy())
                        else:
                            in_memory_batches.append(batch)

                if self.store_on_disk:
                    batched_datasets[div.name] = ANIBatchedDataset(
                        store_dir=div.path.parent, split=div.name
                    )
                else:
                    batched_ds = ANIBatchedInMemoryDataset(
                        batches=in_memory_batches,
                        split=div.name,
                    )
                    if torch.cuda.is_available():
                        batched_ds.pin_memory(verbose=self.verbose)
                    batched_datasets[div.name] = batched_ds
        return batched_datasets

    def _log_creation_data(
        self,
        dest_path: Path,
        dataset: ANIDataset,
        batch_size: int,
        divs_seed: int,
        batch_seed: int,
        padding: tp.Dict[str, float],
        splits: tp.Optional[tp.Dict[str, float]],
        folds: tp.Optional[int],
        properties: tp.Sequence[str],
    ) -> None:
        split_names = sorted(splits) if splits is not None else None
        split_fracs = (
            [splits[k] for k in split_names]
            if (splits is not None and split_names is not None)
            else None
        )
        padded_properties = sorted(k for k in padding if k in properties)
        padded_values = [padding[k] for k in padded_properties]
        creation_log = {
            "datetime_created": str(datetime.datetime.now()),
            "split_names": split_names,
            "split_fractions": split_fracs,
            "folds": folds,
            "divs_seed": divs_seed,
            "batch_seed": batch_seed if self._shuffle else None,
            "batch_size": batch_size if self._shuffle else None,
            "padded_properties": padded_properties,
            "padded_values": padded_values,
            "symbols": dataset.symbols if dataset.grouping != "legacy" else ("?",),
            "properties": properties,
            "store_locations": dataset.store_locations,
            "num_conformers": dataset.num_conformers,
        }
        with open(dest_path / "creation_log.json", "wt") as logfile:
            json.dump(creation_log, logfile, indent=4)


# Kept for bw compat
def create_batched_dataset(
    src: tp.Union[tp.Collection[StrPath], StrPath, ANIDataset],
    dest_path: tp.Optional[StrPath] = None,
    # Dataset modifications and options
    batch_size: int = 2560,
    properties: tp.Iterable[str] = (),
    padding: tp.Optional[tp.Dict[str, float]] = None,
    splits: tp.Optional[tp.Dict[str, float]] = None,
    folds: tp.Optional[int] = None,
    transform: Transform = identity,
    # rng seeds
    divs_seed: tp.Optional[int] = None,
    batch_seed: tp.Optional[int] = None,
    # Performance
    direct_cache: bool = False,
    max_batches_per_packet: int = 200,
    # Verbosity
    verbose: bool = True,
    _shuffle: bool = True,
) -> tp.Dict[str, BatchedDataset]:
    dest_root: tp.Union[Path, tp.Literal["ram"]]

    if direct_cache:
        dest_root = "ram"
        if dest_path is not None:
            raise ValueError("dest_path can't be passed if saving in ram")
        dest_dir = ""
    else:
        if dest_path is None:
            dest_root = Path.cwd()
            dest_dir = "batched_dataset"
        else:
            dest_path = Path(dest_path).resolve()
            dest_root = dest_path.parent
            dest_dir = dest_path.name

    batcher = Batcher(
        dest_root,
        max_batches_per_packet,
        verbose,
    )
    if not _shuffle:
        batcher._no_shuffle()
    return batcher.divide_and_batch(
        src=src,
        dest_dir=dest_dir,
        splits=splits,
        folds=folds,
        batch_size=batch_size,
        padding=padding,
        transform=transform,
        properties=properties,
        divs_seed=divs_seed,
        batch_seed=batch_seed,
    )


def batch_all_in_ram(
    src: tp.Union[tp.Collection[StrPath], StrPath, ANIDataset],
    batch_size: int = 2560,
    properties: tp.Iterable[str] = (),
    padding: tp.Optional[tp.Dict[str, float]] = None,
    transform: Transform = identity,
    # rng seeds
    divs_seed: tp.Optional[int] = None,
    batch_seed: tp.Optional[int] = None,
    verbose: bool = True,
) -> ANIBatchedInMemoryDataset:
    batcher = Batcher.in_ram(verbose)
    splits = batcher.divide_and_batch(
        src=src,
        padding=padding,
        transform=transform,
        properties=properties,
        batch_seed=batch_seed,
        divs_seed=divs_seed,
        batch_size=batch_size,
    )
    return tp.cast(ANIBatchedInMemoryDataset, splits["training"])
