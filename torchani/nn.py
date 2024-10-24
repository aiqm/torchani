r"""
Core modules used by the ANI-style models
"""

import typing as tp

import torch
from torch import Tensor

from torchani.constants import PERIODIC_TABLE
from torchani.tuples import SpeciesCoordinates, SpeciesEnergies
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
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        atomic_energies = self.members_atomic_energies(species_aev).squeeze(0)
        return SpeciesEnergies(species_aev[0], torch.sum(atomic_energies, dim=1))

    @torch.jit.export
    def members_atomic_energies(
        self,
        species_aev: tp.Tuple[Tensor, Tensor],
    ) -> Tensor:
        # Obtain the atomic energies associated with a given tensor of AEV's
        # Note that the output is of shape (1, C, A)
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
        return output.unsqueeze(0)

    def to_infer_model(self, use_mnp: bool = False) -> AtomicContainer:
        return InferModel(self, use_mnp=use_mnp)


class Ensemble(AtomicContainer):
    """Compute the average output of an ensemble of modules."""

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
        species_input: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        species, input = species_input
        sum_ = torch.zeros(species.shape[0], dtype=input.dtype, device=input.device)
        for j, x in enumerate(self.members):
            if j in self.active_members_idxs:
                sum_ += x((species, input))[1]
        return SpeciesEnergies(species, sum_ / self.get_active_members_num())

    @torch.jit.ignore
    def member(self, idx: int) -> AtomicContainer:
        return self.members[idx]

    @torch.jit.export
    def members_atomic_energies(self, species_aev: tp.Tuple[Tensor, Tensor]) -> Tensor:
        #  Note that the output is of shape (M, C, A)
        members_list = []
        for nnp in self.members:
            members_list.append(nnp.members_atomic_energies((species_aev)))
        members_atomic_energies = torch.cat(members_list, dim=0)
        # out shape is (M, C, A)
        return members_atomic_energies

    def to_infer_model(self, use_mnp: bool = False) -> AtomicContainer:
        if use_mnp:
            return InferModel(self, use_mnp=True)
        return BmmEnsemble(self)


class Sequential(torch.nn.ModuleList):
    """Modified Sequential module that accept Tuple type as input"""

    def __init__(self, *modules):
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

    def forward(
        self,
        input_: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ):
        r"""Convert species from atomic numbers to 0, 1, 2, 3, ... indexing"""
        species, coordinates = input_
        converted_species = self.conv_tensor[species]

        # check if unknown species are included
        if converted_species[species.ne(-1)].lt(0).any():
            raise ValueError(f"Unknown species found in {species}")

        return SpeciesCoordinates(converted_species.to(species.device), coordinates)


# Model that just returns zeros
class DummyANIModel(ANIModel):
    @torch.jit.export
    def members_atomic_energies(
        self,
        species_aev: tp.Tuple[Tensor, Tensor],
    ) -> Tensor:
        species, aev = species_aev
        return torch.zeros(species.shape, device=aev.device, dtype=aev.dtype).unsqueeze(
            0
        )

    def to_infer_model(self, use_mnp: bool = False) -> AtomicContainer:
        return self


# Hack: Grab a network with "bad first scalar", discard it and only outputs 2nd
class _ANIModelDiscardFirstScalar(ANIModel):
    @torch.jit.export
    def members_atomic_energies(
        self,
        species_aev: tp.Tuple[Tensor, Tensor],
    ) -> Tensor:
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
        return output.unsqueeze(0)

    def to_infer_model(self, use_mnp: bool = False) -> AtomicContainer:
        return self
