from numpy.typing import NDArray
import tempfile
from dataclasses import dataclass
import shutil

# import tempfile
# import types
import typing as tp
from contextlib import contextmanager
from itertools import chain
from abc import ABC, abstractmethod
from pathlib import Path
from collections import OrderedDict

import numpy as np
import typing_extensions as tpx

from torchani.annotations import NumpyConformers, StrPath, Grouping, Backend


RootKind = tp.Literal["dir", "file"]


class UnsetDataError(Exception):
    pass


class UnsetMetadataError(Exception):
    pass


# Keeps track of variables that must be updated each time the datasets get
# modified or the first time they are read from disk
class Cache:
    group_sizes: tp.OrderedDict[str, int]
    properties: tp.Set[str]

    def __init__(self) -> None:
        self.group_sizes = OrderedDict()
        self.properties = set()


class NamedMapping(tp.Mapping):
    name: str


_MutMapSubtype = tp.TypeVar(
    "_MutMapSubtype", bound=tp.MutableMapping[str, NDArray[tp.Any]]
)

# _ConformerGroup and Store are abstract classes from which all backends
# should inherit in order to correctly interact with ANIDataset. Adding
# support for a new backend can be done just by coding these classes and
# adding the support for the backend inside interface.py


# This is like a dict, but supports "append", and rename keys, it can also
# create dummy properties on the fly.
class _ConformerGroup(tp.MutableMapping[str, NDArray[tp.Any]], ABC):
    def __init__(
        self,
        *args,
        dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
        **kwargs,
    ) -> None:
        self._dummy_properties = (
            dict() if dummy_properties is None else dummy_properties
        )

    def _is_resizable(self) -> bool:
        return True

    def append_conformers(self, conformers: NumpyConformers) -> None:
        if self._is_resizable():
            # We discard all dummy properties that we try to append if they are equal
            # to the ones already present, but if they are not equal we raise an error.
            # The responsibility of materializing the dummy properties before doing the
            # append is the caller's
            for p, v in conformers.items():
                self._append_to_property(p, v)
        else:
            raise ValueError(
                "Can't append conformers, conformer group is not resizable"
            )

    def __getitem__(self, p: str) -> NDArray[tp.Any]:
        try:
            array = self._getitem_impl(p)
        except KeyError:
            # A dummy property is defined with a padding value, a dtype, a
            # shape, and an "is atomic" flag example: dummy_params = {'dtype':
            # np.int64, 'extra_dims': (3,), 'is_atomic': True, 'fill_value':
            # 0.0} this generates a property with shape (C, A, extra_dims),
            # filled with value 0.0
            params = self._dummy_properties[p]
            array = self._make_dummy_property(**params)
        return array

    # creates a dummy property on the fly
    def _make_dummy_property(
        self,
        extra_dims: tp.Tuple[int, ...] = tuple(),
        is_atomic: bool = False,
        fill_value: float = 0.0,
        dtype=np.int64,
    ):
        try:
            species = self._getitem_impl("species")
        except KeyError:
            species = self._getitem_impl("numbers")
        if species.ndim != 2:
            raise RuntimeError("Dummy properties are not supported for legacy grouping")
        shape: tp.Tuple[int, ...] = (species.shape[0],)
        if is_atomic:
            shape += (species.shape[1],)
        return np.full(shape + extra_dims, fill_value, dtype)

    def __len__(self) -> int:
        return self._len_impl() + len(self._dummy_properties)

    def __iter__(self):
        yield from chain(self._iter_impl(), self._dummy_properties.keys())

    @abstractmethod
    def _append_to_property(self, p: str, v: NDArray[tp.Any]) -> None:
        pass

    @abstractmethod
    def _getitem_impl(self, p: str) -> NDArray[tp.Any]:
        pass

    @abstractmethod
    def _iter_impl(self):
        pass

    @abstractmethod
    def _len_impl(self) -> int:
        pass

    @abstractmethod
    def move(self, src_p: str, dest_p: str) -> None:
        pass


class _ConformerWrapper(_ConformerGroup, tp.Generic[_MutMapSubtype]):
    def __init__(self, data: _MutMapSubtype, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def __setitem__(self, p: str, v: NDArray[tp.Any]) -> None:
        self._data[p] = v

    def __delitem__(self, p: str) -> None:
        del self._data[p]

    def _getitem_impl(self, p: str) -> NDArray[tp.Any]:
        return self._data[p][...]

    def _len_impl(self) -> int:
        return len(self._data)

    def _iter_impl(self):
        yield from self._data.keys()

    def move(self, src_p: str, dest_p: str) -> None:
        self._data[dest_p] = self._data.pop(src_p)

    def _append_to_property(self, p: str, v: NDArray[tp.Any]) -> None:
        self._data[p] = np.append(self._data[p], v, axis=0)


# Base location manager for datasets that use either a single file or a structure
# of a directory with some files
class Location:
    def __init__(self, root: StrPath, suffix: str = ""):
        self._root: tp.Optional[Path] = None
        self._suffix = suffix
        self.root = Path(root).resolve()

    @property
    def root(self) -> Path:
        if self._root is None:
            raise ValueError("Tried to access an empty location")
        return self._root

    @root.setter
    def root(self, value: Path) -> None:
        if value.suffix == "":
            value = value.with_suffix(self._suffix)
        if value.suffix != self._suffix:
            raise ValueError(f"Incorrect location {value}")

        if self._root is not None:
            # pathlib.rename() fails if src and dest are in different filesystems
            shutil.move(self.root, value)
        self._root = value

    def clear(self) -> None:
        if self._root is None:
            return
        if self.root.is_dir():
            shutil.rmtree(self.root)
        else:
            self.root.unlink()
        self._root = None


# TODO: Not 100% sure how to type this. Currently Data is an unknown object that
# subclasses decide on and manipulate however they like
Data = tp.Any


@dataclass
class Metadata:
    units: tp.Dict[str, str]
    dtypes: tp.Dict[str, str]
    dims: tp.Dict[str, tp.Tuple[int, ...]]
    grouping: tp.Union[Grouping, tp.Literal["legacy"]]
    info: str = ""


# Wrap a data format (e.g. Zarr, Exedir, HDF5, Parquet)
# Wrapped data must have:
#
# The wrapped data provides some sort of conformer data, which is provided by
# the Store as a "_ConformerGroup" through __getitem__ and deleted through __delitem__

# Overridable methods are:
#
#  - init_new (required)
#  Initializes an empty instance of the wrapped data and metadata from a "root
#  path" and a "grouping"
#
# - setup (required)
#   All init routines needed by the wrapped data format must be performed here
#   (e.g. opening files, aquiring locks, etc)
#   `set_data(data, mode)` must be called by this method. `set_meta(meta, mode)`
#   can also be called here if access to the metadata is not costly, or it can't
#   be done independently from the actual data.
#   If `set_meta` is not called, metadata setup and teardown must be independently
#   implemented in `setup_meta, teardown_meta`
#
# - teardown (required)
#   All finalization routines needed by the wrapped data format must be performed here
#   If metadata was setup by `setup_meta` then, finalization routines for `meta`
#   must be performed in `teardown_meta`
#  (e.g. closing files, etc)
#
# - setup_meta, teardown_meta (optional)
#   Sometimes the data setup procedure is costly, and for performance reasons it
#   may be a good idea to implement access to the metadata only. If this is the
#   case then the following methods can be implemented:
#   All init routines needed by the wrapped metadata can be performed in this
#   pair of methods. If one is implemented the other must be implemented too


class Store(tp.MutableMapping[str, "_ConformerGroup"], ABC):
    root_kind: RootKind
    suffix: str = ""
    backend: Backend
    BACKEND_AVAILABLE: bool = False

    # Root must be a path to an already existing dataset
    def __init__(
        self,
        root: StrPath,
        dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
        grouping: tp.Optional[Grouping] = None,
    ):
        if not self.BACKEND_AVAILABLE:
            raise ValueError(f"{self.backend} could not be found")

        self._dummy_properties = (
            dict() if dummy_properties is None else dummy_properties
        )

        self._meta: tp.Optional[Metadata] = None
        self._meta_mode: tp.Optional[str] = None

        self._data: tp.Optional[Data] = None
        self._data_mode: tp.Optional[str] = None

        self.location = self._build_location(root, self.suffix)

        with self.open(mode="r", only_meta_needed=True) as open_self:
            if grouping is not None and open_self.grouping != grouping:
                raise RuntimeError(
                    f"Attempted to open a dataset with grouping {grouping},"
                    f" but found grouping {open_self.grouping} in provided root path"
                )

    # Overridable
    @staticmethod
    @abstractmethod
    def init_new(
        root: Path,
        grouping: Grouping,
    ) -> None:
        pass

    @abstractmethod
    def update_cache(
        self, check_properties: bool = False, verbose: bool = True
    ) -> tp.Tuple[tp.OrderedDict[str, int], tp.Set[str]]:
        pass

    @abstractmethod
    def setup(self, root: Path, mode: str) -> None:
        pass

    @abstractmethod
    def teardown(self) -> None:
        pass

    def setup_meta(self, root: Path, mode: str) -> None:
        raise NotImplementedError

    def teardown_meta(self) -> None:
        raise NotImplementedError

    # End overridable

    def _build_location(self, location: StrPath, suffix: str) -> Location:
        return Location(location, suffix)

    def overwrite(self, other: "Store") -> None:
        root = Path(other.location.root).with_suffix("")
        other.location.clear()
        self.location.root = root

    @property
    def dummy_properties(self) -> tp.Dict[str, tp.Any]:
        return self._dummy_properties.copy()

    @classmethod
    def make_tmp(
        cls,
        dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
        grouping: tp.Optional[Grouping] = None,
    ) -> tpx.Self:
        if grouping is None:
            grouping = "by_num_atoms"
        if grouping not in ("by_num_atoms", "by_formula"):
            raise RuntimeError(f"Invalid grouping for new dataset: {grouping}")

        if cls.root_kind == "file":
            _, _tmp_root = tempfile.mkstemp(suffix=cls.suffix)
        elif cls.root_kind == "dir":
            _tmp_root = tempfile.mkdtemp(suffix=cls.suffix)
        else:
            raise ValueError(f"Unknown root kind {cls.root_kind}")

        tmp_root = Path(_tmp_root).resolve()
        cls.init_new(tmp_root, grouping)
        return cls(tmp_root, dummy_properties, grouping)

    @classmethod
    def make_new(
        cls,
        root: StrPath,
        dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
        grouping: tp.Optional[Grouping] = None,
    ) -> tpx.Self:
        if grouping is None:
            grouping = "by_num_atoms"
        if grouping not in ("by_num_atoms", "by_formula"):
            raise RuntimeError(f"Invalid grouping for new dataset: {grouping}")
        root = Path(root).resolve()
        if root.suffix != cls.suffix:
            raise ValueError(
                f"Unexpected root suffix {root.suffix}."
                f" For backend {cls.backend} expecting {cls.suffix}"
            )

        if cls.root_kind == "file":
            root.touch(exist_ok=False)
        elif cls.root_kind == "dir":
            root.mkdir(exist_ok=False)
        else:
            raise ValueError(f"Unknown root kind {cls.root_kind}")

        cls.init_new(root, grouping)
        return cls(root, dummy_properties, grouping)

    def set_data(self, data: Data, mode: str) -> None:
        self._data = data
        self._data_mode = mode

    @property
    def data(self) -> Data:
        data, _ = self.get_data()
        return data

    @property
    def meta(self) -> Metadata:
        meta, _ = self.get_meta()
        return meta

    # Getter for "data" guarantees that "data, data_mode" are set up
    def get_data(self) -> tp.Tuple[Data, str]:
        if self._data is None:
            raise UnsetDataError("Data not set")
        if self._data_mode is None:
            raise UnsetDataError("Data mode not set")
        return self._data, self._data_mode

    def set_meta(self, meta: Metadata, mode: str) -> None:
        self._meta = meta
        self._meta_mode = mode

    # Getter for "meta" guarantees that "meta, meta_mode" are set up
    def get_meta(self) -> tp.Tuple[Metadata, str]:
        if self._meta is None:
            raise UnsetMetadataError("Metadata not set")
        if self._meta_mode is None:
            raise UnsetMetadataError("Metadata mode not set")
        return self._meta, self._meta_mode

    @contextmanager
    def open(self, mode: str, only_meta_needed: bool = False) -> tp.Iterator[tpx.Self]:
        try:
            if only_meta_needed:
                try:
                    self.try_open_only_meta(mode)
                except NotImplementedError:
                    self.open_meta_and_data(mode)
            else:
                self.open_meta_and_data(mode)
            yield self
        finally:
            self.close_meta_and_data()

    def try_open_only_meta(self, mode: str) -> None:
        # In case of reentry this is a no-op
        try:
            meta, current_mode = self.get_meta()
            if current_mode != mode:
                raise RuntimeError(f"Metadata already open in mode {current_mode}")
            return
        except UnsetMetadataError:
            pass

        # Try to setup the metadata only, if not implemented then setup
        # everything
        try:
            self.setup_meta(self.location.root, mode)
        except NotImplementedError:
            raise

        try:
            self.get_meta()
        except RuntimeError:
            raise RuntimeError("Metadata not correctly set in setup_meta") from None

    def open_meta_and_data(self, mode: str) -> None:
        # In case of reentry this is a no-op, unless mode is incompatible
        try:
            data, current_mode = self.get_data()
            if current_mode != mode:
                raise RuntimeError(f"Data already open in mode {current_mode}")
            return
        except UnsetDataError:
            pass

        self.setup(self.location.root, mode)
        try:
            self.setup_meta(self.location.root, mode)
        except NotImplementedError:
            pass

        try:
            self.get_data()
            self.get_meta()
        except UnsetDataError:
            raise RuntimeError("Data not correctly set in `setup`") from None
        except UnsetMetadataError:
            raise RuntimeError(
                "Metadata not correctly set `setup` or `setup_meta`"
            ) from None

    def close_meta_and_data(self):
        try:
            self.get_data()  # can raise UnsetDataError
            self.teardown()
        except UnsetDataError:
            pass

        self._data_mode = None
        self._data = None
        try:
            self.get_meta()  # can raise UnsetMetadataError
            self.teardown_meta()
        except (UnsetMetadataError, NotImplementedError):
            pass

        self._meta_mode = None
        self._meta = None

    @property
    def grouping(self) -> tp.Union[Grouping, tp.Literal["legacy"]]:
        return self.meta.grouping


# Wrap a hierarchical data format (e.g. Zarr, Exedir, HDF5)
# Wrapped data must implement:
# create_group, __len__, __iter__ -> Iterator[str], __delitem__, items(), __getitem__
class _HierarchicalStore(Store):
    def update_cache(
        self, check_properties: bool = False, verbose: bool = True
    ) -> tp.Tuple[tp.OrderedDict[str, int], tp.Set[str]]:
        cache = Cache()
        for k, g in self.data.items():
            self._update_properties_cache(cache, g, check_properties)
            self._update_groups_cache(cache, g)
        if list(cache.group_sizes) != sorted(cache.group_sizes):
            raise RuntimeError("Groups were not iterated upon alphanumerically")
        self._dummy_properties = {
            k: v for k, v in self._dummy_properties.items() if k not in cache.properties
        }
        return cache.group_sizes, cache.properties.union(self._dummy_properties)

    def _update_properties_cache(
        self,
        cache: Cache,
        conformers: NamedMapping,
        check_properties: bool = False,
    ) -> None:
        if not cache.properties:
            cache.properties = set(conformers.keys())
        elif check_properties and not set(conformers.keys()) == cache.properties:
            raise RuntimeError(
                f"Group {conformers.name} has bad keys, "
                f"found {set(conformers.keys())}, but expected "
                f"{cache.properties}"
            )

    # updates "group_sizes" which holds the batch dimension (number of
    # molecules) of all groups in the dataset.
    def _update_groups_cache(self, cache: Cache, group: NamedMapping) -> None:
        present_keys = {"coordinates", "coord", "energies"}.intersection(
            set(group.keys())
        )
        try:
            any_key = next(iter(present_keys))
        except StopIteration:
            raise RuntimeError(
                'To infer conformer size need one of "coordinates", "coord", "energies"'
            )
        cache.group_sizes.update({group.name[1:]: group[any_key].shape[0]})

    def __delitem__(self, k: str) -> None:
        del self.data[k]

    def __setitem__(self, name: str, conformers: "_ConformerGroup") -> None:
        self.data.create_group(name)
        self[name].update(conformers)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> tp.Iterator[str]:
        return iter(self.data)
