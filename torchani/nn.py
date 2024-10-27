r"""
Core modules used by the ANI-style models
"""

import warnings
import typing as tp

import torch
from torch import Tensor

from torchani.constants import PERIODIC_TABLE, ATOMIC_NUMBER
from torchani.tuples import SpeciesEnergies
from torchani.atomics import AtomicContainer, AtomicNetwork
from torchani.infer import BmmEnsemble, InferModel


class ANIModel(AtomicContainer):
    """Module that compute energies from species and AEVs.

    Different atom types might have different modules, when computing
    energies, for each atom, the module for its corresponding atom type will
    be applied to its AEV, after that, outputs of modules will be reduced along
    different atoms to obtain molecular energies.

    .. warning::

        The species must be indexed in 0, 1, 2, 3, ..., not the element
        index in periodic table. Check :class:`torchani.SpeciesConverter`
        if you want periodic table indexing.

    .. note:: The resulting energies are in Hartree.

    Arguments:
        modules (Dict[str, AtomicNetwork]): symbol - network pairs for each
        supported element. Different elements will share networks if the same
        ref is used for different keys
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

    def __init__(self, modules: tp.Dict[str, AtomicNetwork]):
        super().__init__()
        if any(s not in PERIODIC_TABLE for s in modules):
            raise ValueError("All modules should be mapped to valid chemical symbols")
        self.atomics = torch.nn.ModuleDict(modules)
        self.num_species = len(self.atomics)
        self.total_members_num = 1
        self.active_members_idxs = [0]

    def forward(
        self,
        species_aev: tp.Tuple[Tensor, Tensor],
        atomic: bool = False,
    ) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]
        species_ = species.flatten()
        aev = aev.flatten(0, 1)
        output = aev.new_zeros(species_.shape)
        for i, m in enumerate(self.atomics.values()):
            midx = (species_ == i).nonzero().view(-1)
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.index_add_(0, midx, m(input_).view(-1))
        output = output.view_as(species)
        if not atomic:
            output = output.sum(dim=-1)
        return SpeciesEnergies(species, output)

    @torch.jit.unused
    def to_infer_model(self, use_mnp: bool = False) -> AtomicContainer:
        return InferModel(self, use_mnp=use_mnp)


class Ensemble(AtomicContainer):
    """Compute the average output of a sequence of AtomicContainer"""

    # Needed for bw compatibility
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        old_keys = list(state_dict.keys())
        for k in old_keys:
            suffix = k.split(prefix)[-1] if prefix else k
            if not suffix.startswith("members."):
                state_dict["".join((prefix, "members.", suffix))] = state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __init__(self, modules: tp.Sequence[AtomicContainer]):
        super().__init__()
        self.members = torch.nn.ModuleList(modules)
        self.total_members_num = len(self.members)
        self.active_members_idxs = list(range(self.total_members_num))

        if any(m.num_species != modules[0].num_species for m in modules):
            raise ValueError(
                "All modules in the ensemble must support the same number of species"
            )
        self.num_species = modules[0].num_species

    def forward(
        self,
        species_aevs: tp.Tuple[Tensor, Tensor],
        atomic: bool = False,
    ) -> SpeciesEnergies:
        elem_idxs, aevs = species_aevs
        if atomic:
            energies = aevs.new_zeros(elem_idxs.shape)
        else:
            energies = aevs.new_zeros(elem_idxs.shape[0])
        for j, nnp in enumerate(self.members):
            if j in self.active_members_idxs:
                energies += nnp((elem_idxs, aevs), atomic=atomic)[1]
        energies = energies / self.get_active_members_num()
        return SpeciesEnergies(elem_idxs, energies)

    def ensemble_values(
        self,
        species_aevs: tp.Tuple[Tensor, Tensor],
        atomic: bool = False,
    ) -> Tensor:
        elem_idxs, aevs = species_aevs
        size = len(self.active_members_idxs)
        if atomic:
            energies = aevs.new_zeros((size, elem_idxs.shape[0], elem_idxs.shape[1]))
        else:
            energies = aevs.new_zeros((size, elem_idxs.shape[0]))
        idx = 0
        for j, nnp in enumerate(self.members):
            if j in self.active_members_idxs:
                energies[idx] = nnp((elem_idxs, aevs), atomic=atomic)[1]
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


# Dummy model that just returns zeros
class DummyANIModel(ANIModel):
    def forward(
        self,
        species_aev: tp.Tuple[Tensor, Tensor],
        atomic: bool = False,
    ) -> SpeciesEnergies:
        elem_idxs, aevs = species_aev
        if atomic:
            energies = aevs.new_zeros(elem_idxs.shape)
        else:
            energies = aevs.new_zeros(elem_idxs.shape[0])
        return SpeciesEnergies(elem_idxs, energies)


# Hack: Grab a network with "bad first scalar", discard it and only outputs 2nd
class _ANIModelDiscardFirstScalar(ANIModel):
    def forward(
        self,
        species_aev: tp.Tuple[Tensor, Tensor],
        atomic: bool = False,
    ) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]
        species_ = species.flatten()
        aev = aev.flatten(0, 1)
        output = aev.new_zeros(species_.shape)
        for i, m in enumerate(self.atomics.values()):
            midx = (species_ == i).nonzero().view(-1)
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.index_add_(0, midx, m(input_)[:, 1].view(-1))
        output = output.view_as(species)
        if not atomic:
            output = output.sum(dim=-1)
        return SpeciesEnergies(species, output)

    @torch.jit.unused
    def to_infer_model(self, use_mnp: bool = False) -> AtomicContainer:
        return self


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
        elem_idxs = self.conv_tensor[atomic_nums]
        if (elem_idxs[atomic_nums != -1] == -1).any():
            raise ValueError(
                f"Model doesn't support some elements in input"
                f" Input elements include: {torch.unique(atomic_nums)}"
                f" Supported elements are: {self.atomic_numbers}"
            )
        return elem_idxs.to(atomic_nums.device)


class Sequential(torch.nn.ModuleList):
    """Modified Sequential module that accept Tuple type as input"""

    def __init__(self, *modules):
        super().__init__(modules)
        warnings.warn("Use of `torchani.nn.Sequential` is strongly discouraged.")

    def forward(
        self,
        input_: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ):
        for module in self:
            if hasattr(module, "neighborlist"):
                input_ = module(input_, cell=cell, pbc=pbc)
            else:
                input_ = module(input_)
        return input_
