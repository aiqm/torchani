import typing as tp
from pathlib import Path
from functools import partial

import h5py
from numpy.typing import NDArray
from tqdm import tqdm

from torchani.annotations import StrPath, Grouping, Backend
from torchani.datasets.backends.interface import (
    RootKind,
    Metadata,
    _ConformerGroup,
    _ConformerWrapper,
    Cache,
    _HierarchicalStore,
)


class _HDF5Store(_HierarchicalStore):
    suffix: str = ".h5"
    root_kind: RootKind = "file"
    backend: Backend = "hdf5"
    BACKEND_AVAILABLE: bool = True

    def __init__(
        self,
        root: StrPath,
        dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
        grouping: tp.Optional[Grouping] = None,
    ):
        super().__init__(root, dummy_properties, grouping)
        self._has_flat_format = True
        self._tried_to_infer_flat_format = False

    def setup(self, root: Path, mode: str) -> None:
        file = h5py.File(root, mode)
        grouping: tp.Union[Grouping, tp.Literal["legacy"]] = "legacy"  # default

        # This detects Roman's formatting style which doesn't have a
        # 'grouping' key but is still grouped by num atoms.
        try:
            file.attrs["readme"]
            grouping = "by_num_atoms"
        except Exception:
            pass

        try:
            grouping = file.attrs["grouping"]
            if grouping not in ("by_num_atoms", "legacy", "by_formula"):
                raise RuntimeError(f"Unknown grouping: {grouping}")
        except Exception:
            pass

        meta = Metadata(
            grouping=grouping,
            dims=dict(),
            dtypes=dict(),
            units=dict(),
        )
        self.set_data(file, mode)
        self.set_meta(meta, mode)

    def teardown(self) -> None:
        self.data.close()

    @staticmethod
    def init_new(
        root: Path,
        grouping: Grouping,
    ) -> None:
        with h5py.File(str(root), "r+") as f:
            f.attrs["grouping"] = grouping
        # TODO: make sure initialized class has _has_flat_format = True

    def __getitem__(self, name: str) -> "_ConformerGroup":
        return _HDF5ConformerGroup(
            self.data[name], dummy_properties=self._dummy_properties
        )

    def update_cache(
        self, check_properties: bool = False, verbose: bool = True
    ) -> tp.Tuple[tp.OrderedDict[str, int], tp.Set[str]]:
        cache = Cache()
        # If the dataset is standarized (it is a tree with depth
        # 1, where all groups are directly joined to the root) then it is much faster
        # to traverse, so after the first recursion if this
        # structure is detected set a flag to prevent doing the recursion again.
        # This speeds up cache updates and lookup x30
        if self.grouping == "legacy" and not self._tried_to_infer_flat_format:
            self._has_flat_format = self._can_infer_flat_format()
            self._tried_to_infer_flat_format = True

        if self._has_flat_format:
            for k, g in self.data.items():
                if g.name in ["/_created", "/_meta"]:
                    continue
                self._update_properties_cache(cache, g, check_properties)
                self._update_groups_cache(cache, g)
        else:
            found_flat_format = self._update_cache_recursive_iter(
                cache, check_properties, verbose
            )
            self._has_flat_format = found_flat_format
        # By default iteration of HDF5 should be alphanumeric in which case
        # sorting should not be necessary, this internal check ensures the
        # groups were not created with 'track_order=True', and that the visitor
        # function worked properly.
        if list(cache.group_sizes) != sorted(cache.group_sizes):
            raise RuntimeError("Groups were not iterated upon alphanumerically")
        # we get rid of dummy properties if they are already in the dataset
        self._dummy_properties = {
            k: v for k, v in self._dummy_properties.items() if k not in cache.properties
        }
        return cache.group_sizes, cache.properties.union(self._dummy_properties)

    def _update_cache_recursive_iter(
        self, cache: Cache, check_properties: bool, verbose: bool
    ) -> bool:
        def visitor_fn(
            name: str,
            object_: tp.Union[h5py.Dataset, h5py.Group],
            store: "_HDF5Store",
            cache: Cache,
            check_properties: bool,
            pbar: tp.Any,
        ) -> None:
            pbar.update()
            # We make sure the node is a Dataset, and we avoid Datasets
            # called _meta or _created since if present these store units
            # or other metadata. We also check if we already visited this
            # group via one of its children.
            if (
                not isinstance(object_, h5py.Dataset)
                or object_.name in ["/_created", "/_meta"]
                or object_.parent.name in cache.group_sizes.keys()
            ):
                return
            g = object_.parent
            # Check for format correctness
            for v in g.values():
                if isinstance(v, h5py.Group):
                    raise RuntimeError(
                        f"Invalid dataset format, there shouldn't be "
                        "Groups inside Groups that have Datasets, "
                        f"but {g.name}, parent of the dataset "
                        f"{object_.name}, has group {v.name} as a "
                        "child"
                    )
            store._update_properties_cache(cache, g, check_properties)
            store._update_groups_cache(cache, g)

        with tqdm(desc="Verifying format correctness", disable=not verbose) as pbar:
            self.data.visititems(
                partial(
                    visitor_fn,
                    store=self,
                    cache=cache,
                    pbar=pbar,
                    check_properties=check_properties,
                )
            )
        # If the visitor function succeeded and this condition is met the
        # dataset must be in flat format
        return not any("/" in k[1:] for k in cache.group_sizes.keys())

    # Check if the raw hdf5 file is one of a number of known files that can be assumed
    # to have standard format.
    def _can_infer_flat_format(self) -> bool:
        # This check detects the "ani-release" files which have this property
        data = self.data
        try:
            key = next(iter(data.keys()))
            data[key]["hf_dz.energy"]
            return True
        except Exception:
            pass

        # This check tests for the '/_created' which is present in "old HTRQ style"
        try:
            data["/_created"]
            return True
        except KeyError:
            return False


class _HDF5ConformerGroup(_ConformerWrapper[h5py.Group]):
    def __init__(self, data: h5py.Group, **kwargs):
        super().__init__(data=data, **kwargs)

    def _is_resizable(self) -> bool:
        return all(ds.maxshape[0] is None for ds in self._data.values())

    def _append_to_property(self, p: str, v: NDArray[tp.Any]) -> None:
        h5_dataset = self._data[p]
        h5_dataset.resize(h5_dataset.shape[0] + v.shape[0], axis=0)
        try:
            h5_dataset[-v.shape[0]:] = v
        except TypeError:
            h5_dataset[-v.shape[0]:] = v.astype(bytes)

    def __setitem__(self, p: str, data: NDArray[tp.Any]) -> None:
        # This correctly handles strings and make the first axis resizable
        maxshape = (None,) + data.shape[1:]
        try:
            self._data.create_dataset(name=p, data=data, maxshape=maxshape)
        except TypeError:
            self._data.create_dataset(
                name=p, data=data.astype(bytes), maxshape=maxshape
            )
