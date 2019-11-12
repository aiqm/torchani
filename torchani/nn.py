import torch
from torch import Tensor
from typing import Tuple, NamedTuple


class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor


class ANIModel(torch.nn.ModuleList):
    """ANI model that compute properties from species and AEVs.

    Different atom types might have different modules, when computing
    properties, for each atom, the module for its corresponding atom type will
    be applied to its AEV, after that, outputs of modules will be reduced along
    different atoms to obtain molecular properties.

    Arguments:
        modules (:class:`collections.abc.Sequence`): Modules for each atom
            types. Atom types are distinguished by their order in
            :attr:`modules`, which means, for example ``modules[i]`` must be
            the module for atom type ``i``. Different atom types can share a
            module by putting the same reference in :attr:`modules`.
    """

    def __init__(self, modules):
        super(ANIModel, self).__init__(modules)

    def forward(self, species_aev: Tuple[Tensor, Tensor]) -> SpeciesEnergies:
        species, aev = species_aev
        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        output = aev.new_zeros(species_.shape)

        for i, m in enumerate(self):
            mask = (species_ == i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species)
        return SpeciesEnergies(species, torch.sum(output, dim=1))


class Ensemble(torch.nn.ModuleList):
    """Compute the average output of an ensemble of modules."""

    def __init__(self, modules):
        super(Ensemble, self).__init__(modules)
        self.size = len(modules)

    def forward(self, species_input: Tuple[Tensor, Tensor]) -> SpeciesEnergies:
        sum_ = 0
        for x in self:
            sum_ += x(species_input)[1]
        species, _ = species_input
        return SpeciesEnergies(species, sum_ / self.size)


class Sequential(torch.nn.ModuleList):
    """Modified Sequential module that accept Tuple type as input"""

    def __init__(self, *modules):
        super(Sequential, self).__init__(modules)

    def forward(self, input_: Tuple[Tensor, Tensor]):
        for module in self:
            input_ = module(input_)
        return input_


class Gaussian(torch.nn.Module):
    """Gaussian activation"""
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(- x * x)
