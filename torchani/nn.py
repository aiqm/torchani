r"""
Modules that represent groups of element-specific neural networks, and perform different
reduction operations over them.
"""

import warnings
import typing as tp

import torch
from torch import Tensor

from torchani.constants import PERIODIC_TABLE, ATOMIC_NUMBER
from torchani.tuples import SpeciesEnergies
from torchani.atomics import AtomicContainer, AtomicNetwork
from torchani.infer import BmmEnsemble, InferModel


class ANINetworks(AtomicContainer):
    r"""
    Predict molecular or atomic scalars from a set of element-specific atomic networks

    ``ANINetworks`` will iterate over atomic networks and calculate the corresponding
    atomic scalars. By default the outputs are summed over atoms to obtain molecular
    quantities. This can be disabled with ``atomic=True``. If you want to allow
    different elements to map to the same network, pass ``alias=True``, otherwise
    elemetns are required to be mapped to different, element-specific networks.

    .. warning::

        The species input to this module must be indexed with 0, 1, 2, 3, ..., and not
        with atomic numbers.

    Arguments:
        modules (dict[str, AtomicNetwork]): symbol - network pairs for each supported
            element. Different elements will share networks if the same ref is used for
            different keys
        alias (bool): Allow the class to map different elements to the same atomic
            network.
    """

    # Needed for bw compatibility
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        old_keys = list(state_dict.keys())
        even = list(range(0, 10, 2))
        for k in old_keys:
            suffix = k.split(prefix)[-1] if prefix else k
            new_key = k
            if not suffix.startswith("atomics."):
                new_key = "".join((prefix, "atomics.", suffix))
            if ("layers" not in k) and ("final_layer" not in k):
                parts = new_key.split(".")
                if int(parts[-2]) == 6:
                    parts[-2] = "final_layer"
                else:
                    parts.insert(-2, "layers")
                    parts[-2] = str(even.index(int(parts[-2])))
                new_key = ".".join(parts)

            state_dict[new_key] = state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __init__(self, modules: tp.Dict[str, AtomicNetwork], alias: bool = False):
        super().__init__()
        if any(s not in PERIODIC_TABLE for s in modules):
            raise ValueError("All modules should be mapped to valid chemical symbols")
        if not alias and len(set(id(m) for m in modules.values())) != len(modules):
            raise ValueError("Symbols map to same module. If intended use `alias=True`")
        self.atomics = torch.nn.ModuleDict(modules)
        self.num_species = len(self.atomics)
        self.total_members_num = 1
        self.active_members_idxs = [0]

    def forward(
        self,
        elem_idxs: Tensor,
        aevs: Tensor,
        atomic: bool = False,
    ) -> Tensor:
        assert elem_idxs.shape == aevs.shape[:-1]
        flat_elem_idxs = elem_idxs.flatten()
        aev = aevs.flatten(0, 1)
        scalars = aev.new_zeros(flat_elem_idxs.shape)
        for i, m in enumerate(self.atomics.values()):
            selected_idxs = (flat_elem_idxs == i).nonzero().view(-1)
            if selected_idxs.shape[0] > 0:
                input_ = aev.index_select(0, selected_idxs)
                scalars.index_add_(0, selected_idxs, m(input_).view(-1))
        scalars = scalars.view_as(elem_idxs)
        if atomic:
            return scalars
        return scalars.sum(dim=1)

    @torch.jit.unused
    def to_infer_model(self, use_mnp: bool = False) -> AtomicContainer:
        return InferModel(self, use_mnp=use_mnp)

    # Legacy API
    def call(self, species_aevs: tp.Tuple[Tensor, Tensor]) -> SpeciesEnergies:
        warnings.warn(".call is a deprecated API and will be removed in the future")
        species, aevs = species_aevs
        return SpeciesEnergies(species, self(species, aevs))


class ANIEnsemble(AtomicContainer):
    r"""
    Calculate output scalars by averaging over a set of AtomicContainer

    Arguments:
        modules (iterable[AtomicContainer]): Set of networks to average.
        repeats (bool): Allow repeated networks (False by default)
    """

    # Needed for bw compatibility
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        old_keys = list(state_dict.keys())
        for k in old_keys:
            suffix = k.split(prefix)[-1] if prefix else k
            if not suffix.startswith("members."):
                state_dict["".join((prefix, "members.", suffix))] = state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __init__(self, modules: tp.Iterable[AtomicContainer], repeats: bool = False):
        super().__init__()
        if not repeats and len(set(map(id, modules))) != len(tuple(modules)):
            raise ValueError("Modules are repeated. If intended use `repeats=True`")
        self.members = torch.nn.ModuleList(modules)
        self.total_members_num = len(self.members)
        self.active_members_idxs = list(range(self.total_members_num))
        self.num_species = next(iter(modules)).num_species
        if any(m.num_species != self.num_species for m in modules):
            raise ValueError("All modules must support the same number of elements")

    def forward(self, elem_idxs: Tensor, aevs: Tensor, atomic: bool = False) -> Tensor:
        if atomic:
            scalars = aevs.new_zeros(elem_idxs.shape)
        else:
            scalars = aevs.new_zeros(elem_idxs.shape[0])
        for j, nnp in enumerate(self.members):
            if j in self.active_members_idxs:
                scalars += nnp(elem_idxs, aevs, atomic=atomic)
        return scalars / self.get_active_members_num()

    def ensemble_values(
        self, elem_idxs: Tensor, aevs: Tensor, atomic: bool = False
    ) -> Tensor:
        r"""Don't reduce over the set of networks"""
        size = len(self.active_members_idxs)
        if atomic:
            energies = aevs.new_zeros((size, elem_idxs.shape[0], elem_idxs.shape[1]))
        else:
            energies = aevs.new_zeros((size, elem_idxs.shape[0]))
        idx = 0
        for j, nnp in enumerate(self.members):
            if j in self.active_members_idxs:
                energies[idx] = nnp(elem_idxs, aevs, atomic=atomic)
                idx += 1
        return energies

    @torch.jit.unused
    def member(self, idx: int) -> AtomicContainer:
        return self.members[idx]

    @torch.jit.unused
    def to_infer_model(self, use_mnp: bool = False) -> AtomicContainer:
        if use_mnp:
            return InferModel(self, use_mnp=True)
        return BmmEnsemble(self)


class SpeciesConverter(torch.nn.Module):
    """Converts tensors with species labeled as atomic numbers into tensors
    labeled with internal torchani indices according to a custom ordering
    scheme. It takes a custom species ordering as initialization parameter. If
    the class is initialized with ['H', 'C', 'N', 'O'] for example, it will
    convert a tensor [1, 1, 6, 7, 1, 8] into a tensor [0, 0, 1, 2, 0, 3]

    Arguments:
        species (:class:`collections.abc.Sequence` of :class:`str`):
        sequence of all supported species, in order (it is recommended to order
        according to atomic number).
    """

    conv_tensor: Tensor

    def __init__(self, species: tp.Sequence[str]):
        super().__init__()
        rev_idx = {s: k for k, s in enumerate(PERIODIC_TABLE)}
        maxidx = max(rev_idx.values())
        self.register_buffer(
            "conv_tensor", torch.full((maxidx + 2,), -1, dtype=torch.long)
        )
        for i, s in enumerate(species):
            self.conv_tensor[rev_idx[s]] = i
        self.atomic_numbers = torch.tensor(
            [ATOMIC_NUMBER[e] for e in species], dtype=torch.long
        )

    def forward(self, atomic_nums: Tensor, nop: bool = False) -> Tensor:
        r"""Convert species from atomic numbers to 0, 1, 2, 3, ... indexing"""

        # Consider as element idxs and check that its not too large, otherwise its
        # a no-op TODO: unclear documentation for this, possibly remove
        if nop:
            if atomic_nums.max() >= len(self.atomic_numbers):
                raise ValueError(f"Unsupported element idx in {atomic_nums}")
            return atomic_nums
        # NOTE: Casting is necessary in C++ due to a LibTorch bug
        elem_idxs = self.conv_tensor.to(torch.long)[atomic_nums]
        if (elem_idxs[atomic_nums != -1] == -1).any():
            raise ValueError(
                f"Model doesn't support some elements in input"
                f" Input elements include: {torch.unique(atomic_nums)}"
                f" Supported elements are: {self.atomic_numbers}"
            )
        return elem_idxs.to(atomic_nums.device)


# Hack: ANINetworks that return zeros
class _ZeroANINetworks(ANINetworks):
    def forward(self, elem_idxs: Tensor, aevs: Tensor, atomic: bool = False) -> Tensor:
        if atomic:
            return aevs.new_zeros(elem_idxs.shape)
        return aevs.new_zeros(elem_idxs.shape[0])


# Hack: Wrapper that Grabs a network with "bad first scalar", and discards it
class _ANINetworksDiscardFirstScalar(ANINetworks):
    def forward(self, elem_idxs: Tensor, aevs: Tensor, atomic: bool = False) -> Tensor:
        assert elem_idxs.shape == aevs.shape[:-1]
        flat_elem_idxs = elem_idxs.flatten()
        aev = aevs.flatten(0, 1)
        scalars = aev.new_zeros(flat_elem_idxs.shape)
        for i, m in enumerate(self.atomics.values()):
            selected_idxs = (flat_elem_idxs == i).nonzero().view(-1)
            if selected_idxs.shape[0] > 0:
                input_ = aev.index_select(0, selected_idxs)
                scalars.index_add_(0, selected_idxs, m(input_)[:, 1].view(-1))
        scalars = scalars.view_as(elem_idxs)
        if atomic:
            return scalars
        return scalars.sum(dim=1)

    @torch.jit.unused
    def to_infer_model(self, use_mnp: bool = False) -> AtomicContainer:
        return self


# Legacy API
class Sequential(torch.nn.ModuleList):
    """Modified Sequential module that accept Tuple type as input"""

    def __init__(self, *modules):
        warnings.warn("Use of `torchani.nn.Sequential` is strongly discouraged.")
        super().__init__(modules)

    def forward(
        self,
        input_: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ):
        for module in self:
            input_ = module(input_, cell=cell, pbc=pbc)
        return input_


class Ensemble(ANIEnsemble):
    def __init__(self, modules: tp.Any) -> None:
        warnings.warn("torchani.Ensemble is deprecated, use torchani.nn.ANIEnsemble")
        super().__init__(modules, repeats=True)

    # Signature is incompatible since this class is legacy
    def forward(  # type: ignore
        self,
        species_aevs: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        return self.call(species_aevs)


class ANIModel(ANINetworks):
    def __init__(self, modules: tp.Any) -> None:
        warnings.warn("torchani.ANIModel is deprecated, use torchani.nn.ANINetworks")
        super().__init__(modules, alias=True)

    # Signature is incompatible since this class is legacy
    def forward(  # type: ignore
        self,
        species_aevs: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        return self.call(species_aevs)
