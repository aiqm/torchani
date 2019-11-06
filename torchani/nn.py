import torch
from typing import Tuple


class ANIModel(torch.nn.Module):
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
        super(ANIModel, self).__init__()
        self.module_list = torch.nn.ModuleList(modules)

    def __getitem__(self, i):
        return self.module_list[i]

    def forward(self, species_aev):
        # type: (Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
        species, aev = species_aev
        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        output = aev.new_zeros(species_.shape)

        for i, m in enumerate(self.module_list):
            mask = (species_ == i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species)
        return species, torch.sum(output, dim=1)


class Ensemble(torch.nn.Module):
    """Compute the average output of an ensemble of modules."""

    def __init__(self, modules):
        super(Ensemble, self).__init__()
        self.modules_list = torch.nn.ModuleList(modules)
        self.size = len(self.modules_list)

    def forward(self, species_input):
        # type: (Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
        sum_ = 0
        for x in self.modules_list:
            sum_ += x(species_input)[1]
        species, _ = species_input
        return species, sum_ / self.size

    def __getitem__(self, i):
        return self.modules_list[i]


class Sequential(torch.nn.Module):
    """Modified Sequential module that accept Tuple type as input"""

    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.modules_list = torch.nn.ModuleList(modules)

    def forward(self, input_):
        # type: (Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
        for module in self.modules_list:
            input_ = module(input_)
        return input_


class Gaussian(torch.nn.Module):
    """Gaussian activation"""
    def forward(self, x):
        return torch.exp(- x * x)
