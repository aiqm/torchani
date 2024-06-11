r"""Functions for creating batched datasets"""
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

from torchani.utils import pad_atomic_properties, PADDING
from torchani.datasets.datasets import ANIDataset
from torchani.transforms import Transform, Identity
from torchani.datasets._annotations import Conformers, StrPath
from torchani.storage import DATASETS_DIR


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


@dataclass
class Div:
    name: str
    indices: Tensor  # dtype=torch.long, shape=(num-conformers, 2)
    path: Path


class Batcher:
    def __init__(
        self,
        dest_root: tp.Union[Path, tp.Literal["ram"]] = DATASETS_DIR,
        max_batches_per_packet: int = 350,
        verbose: bool = False,
    ) -> None:
        self.max_batches_per_packet = max_batches_per_packet
        self.verbose = verbose
        self.store_on_disk = dest_root != "ram"
        if not self.store_on_disk:
            if max_batches_per_packet != 350:
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
        transform: tp.Optional[Transform] = None,
        properties: tp.Iterable[str] = (),
        # rng seeds
        divs_seed: tp.Optional[int] = None,
        batch_seed: tp.Optional[int] = None,
    ) -> tp.Dict[str, ANIBatchedDataset]:
        padding = PADDING if padding is None else padding
        transform = Identity() if transform is None else transform
        if (not self.store_on_disk) and dest_dir:
            raise ValueError("dest_dir can't be passed if saving in ram")

        if not dest_dir:
            dest_dir = "batched_dataset"

        dest_path = self._dest_root / dest_dir

        if not self._shuffle and divs_seed is not None or batch_seed is not None:
            raise ValueError("Seeds must be None if not shuffling")
        dataset = src if isinstance(src, ANIDataset) else ANIDataset(src)

        if not properties:
            properties = dataset.tensor_properties
        elif isinstance(properties, str):
            properties = [properties]
        else:
            properties = sorted(properties)

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
            batch_seed = divs_rng.seed()
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
    ) -> tp.Dict[str, ANIBatchedDataset]:
        group_names = list(dataset.keys())
        batched_datasets: tp.Dict[str, ANIBatchedDataset] = dict()
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
                    for group_idx in packet_unique_group_idxs:
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
                        # TODO: remove redundant padding here!
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
                    batched_ds = ANIBatchedDataset(
                        store_dir=div.path.parent, split=div.name
                    )
                else:
                    batched_ds = ANIBatchedDataset(
                        batches=in_memory_batches, split=div.name
                    ).cache(verbose=False, pin_memory=torch.cuda.is_available())
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
        creation_log = {
            "datetime_created": str(datetime.datetime.now()),
            "splits": splits,
            "folds": folds,
            "divs_seed": divs_seed,
            "batch_seed": batch_seed,
            "batch_size": batch_size,
            "padding": padding,
            "shuffle": self._shuffle,
            "symbols": dataset.symbols if dataset.grouping != "legacy" else ("?",),
            "properties": properties,
            "store_locations": dataset.store_locations,
            "num_conformers": dataset.num_conformers,
        }
        with open(dest_path / "creation_log.json", "wt") as logfile:
            json.dump(creation_log, logfile, indent=1)


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
    transform: tp.Optional[Transform] = None,
    # rng seeds
    divs_seed: tp.Optional[int] = None,
    batch_seed: tp.Optional[int] = None,
    # Performance
    direct_cache: bool = False,
    max_batches_per_packet: int = 350,
    # Verbosity
    verbose: bool = True,
    _shuffle: bool = True,
) -> tp.Dict[str, ANIBatchedDataset]:
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
        src,
        dest_dir,
        splits,
        folds,
        batch_size,
        padding,
        transform,
        properties,
    )


def batch_all_in_ram(
    src: tp.Union[tp.Collection[StrPath], StrPath, ANIDataset],
    batch_size: int = 2560,
    properties: tp.Iterable[str] = (),
    padding: tp.Optional[tp.Dict[str, float]] = None,
    transform: tp.Optional[Transform] = None,
    # rng seeds
    divs_seed: tp.Optional[int] = None,
    batch_seed: tp.Optional[int] = None,
    verbose: bool = True,
) -> ANIBatchedDataset:
    batcher = Batcher.in_ram(verbose)
    splits = batcher.divide_and_batch(
        src, padding=padding, transform=transform, properties=properties
    )
    return splits["training"]
