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
        padding_fill (float): The value to fill output of padding atoms.
            Padding values will participate in reducing, so this value should
            be appropriately chosen so that it has no effect on the result. For
            example, if the reducer is :func:`torch.sum`, then
            :attr:`padding_fill` should be 0, and if the reducer is
            :func:`torch.min`, then :attr:`padding_fill` should be
            :obj:`math.inf`.
    """

    def __init__(self, modules, padding_fill=0):
        super(ANIModel, self).__init__()
        self.module_list = torch.nn.ModuleList(modules)
        self.padding_fill = padding_fill

    def __getitem__(self, i):
        return self.module_list[i]

    def forward(self, species_aev):
        # type: (Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
        species, aev = species_aev
        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        output = torch.full(species_.shape, self.padding_fill,
                            dtype=aev.dtype)
        i = 0
        for m in self.module_list:
            mask = (species_ == i)
            i += 1
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species)
        return species, torch.sum(output, dim=1)


class Ensemble(torch.nn.Module):
    """Compute the average output of an ensemble of modules."""

    # FIXME: due to PyTorch bug, we have to hard code the
    # ensemble size to 8.

    # def __init__(self, modules):
    #     super(Ensemble, self).__init__()
    #     self.modules_list = torch.nn.ModuleList(modules)

    # def forward(self, species_input):
    #     # type: (Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
    #     outputs = [x(species_input)[1] for x in self.modules_list]
    #     species, _ = species_input
    #     return species, sum(outputs) / len(outputs)

    def __init__(self, modules):
        super(Ensemble, self).__init__()
        assert len(modules) == 8
        self.model0 = modules[0]
        self.model1 = modules[1]
        self.model2 = modules[2]
        self.model3 = modules[3]
        self.model4 = modules[4]
        self.model5 = modules[5]
        self.model6 = modules[6]
        self.model7 = modules[7]

    def __getitem__(self, i):
        return [self.model0, self.model1, self.model2, self.model3,
                self.model4, self.model5, self.model6, self.model7][i]

    def forward(self, species_input):
        # type: (Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
        species, _ = species_input
        sum_ = self.model0(species_input)[1] + self.model1(species_input)[1] \
            + self.model2(species_input)[1] + self.model3(species_input)[1] \
            + self.model4(species_input)[1] + self.model5(species_input)[1] \
            + self.model6(species_input)[1] + self.model7(species_input)[1]
        return species, sum_ / 8.0


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
