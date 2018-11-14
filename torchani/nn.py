import torch
from . import utils


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
        reducer (:class:`collections.abc.Callable`): The callable that reduce
            atomic outputs into molecular outputs. It must have signature
            ``(tensor, dim)->tensor``.
        padding_fill (float): The value to fill output of padding atoms.
            Padding values will participate in reducing, so this value should
            be appropriately chosen so that it has no effect on the result. For
            example, if the reducer is :func:`torch.sum`, then
            :attr:`padding_fill` should be 0, and if the reducer is
            :func:`torch.min`, then :attr:`padding_fill` should be
            :obj:`math.inf`.
    """

    def __init__(self, modules, reducer=torch.sum, padding_fill=0):
        super(ANIModel, self).__init__(modules)
        self.reducer = reducer
        self.padding_fill = padding_fill

    def forward(self, species_aev):
        species, aev = species_aev
        species_ = species.flatten()
        present_species = utils.present_species(species)
        aev = aev.flatten(0, 1)

        output = torch.full_like(species_, self.padding_fill,
                                 dtype=aev.dtype)
        for i in present_species:
            mask = (species_ == i)
            input_ = aev.index_select(0, mask.nonzero().squeeze())
            output.masked_scatter_(mask, self[i](input_).squeeze())
        output = output.view_as(species)
        return species, self.reducer(output, dim=1)


class Ensemble(torch.nn.ModuleList):
    """Compute the average output of an ensemble of modules."""

    def forward(self, species_input):
        outputs = [x(species_input)[1] for x in self]
        species, _ = species_input
        return species, sum(outputs) / len(outputs)


class Gaussian(torch.nn.Module):
    """Gaussian activation"""
    def forward(self, x):
        return torch.exp(-x*x)
