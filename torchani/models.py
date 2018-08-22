import torch
from . import utils


class ANIModel(torch.nn.ModuleList):

    def __init__(self, modules, reducer=torch.sum, padding_fill=0):
        """
        Parameters
        ----------
        modules : seq(torch.nn.Module)
            Modules for all species.
        reducer : function
            Function of (input, dim)->output that reduce the input tensor along
            the given dimension to get an output tensor. This function will be
            called with the per atom output tensor with internal shape as input
            , and desired reduction dimension as dim, and should reduce the
            input into the tensor containing desired output.
        padding_fill : float
            Default value used to fill padding atoms
        """
        super(ANIModel, self).__init__(modules)
        self.reducer = reducer
        self.padding_fill = padding_fill

    def forward(self, species_aev):
        """Compute output from aev

        Parameters
        ----------
        (species, aev)
        species : torch.Tensor
            Tensor storing the species for each atom.
        aev : torch.Tensor
            Pytorch tensor of shape (conformations, atoms, aev_length) storing
            the computed AEVs.

        Returns
        -------
        (species, output)
        species : torch.Tensor
            Tensor storing the species for each atom.
        output : torch.Tensor
            Pytorch tensor of shape (conformations, output_length) for the
            output of each conformation.
        """
        species, aev = species_aev
        species_ = species.flatten()
        present_species = utils.present_species(species)
        aev = aev.flatten(0, 1)

        output = torch.full_like(species_, self.padding_fill,
                                 dtype=aev.dtype)
        for i in present_species:
            mask = (species_ == i)
            input = aev.index_select(0, mask.nonzero().squeeze())
            output[mask] = self[i](input).squeeze()
        output = output.view_as(species)
        return species, self.reducer(output, dim=1)


class Ensemble(torch.nn.ModuleList):

    def forward(self, species_aev):
        outputs = [x(species_aev)[1] for x in self]
        species, _ = species_aev
        return species, sum(outputs) / len(outputs)
