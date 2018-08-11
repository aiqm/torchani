import torch
from ..benchmarked import BenchmarkedModule


class ANIModel(BenchmarkedModule):
    """Subclass of `torch.nn.Module` for the [xyz]->[aev]->[per_atom_y]->y
    pipeline.

    Attributes
    ----------
    species : list
        Chemical symbol of supported atom species.
    suffixes : sequence
        Different suffixes denote different models in an ensemble.
    model_<X><suffix> : nn.Module
        Model of suffix <suffix> for species <X>. There should be one such
        attribute for each supported species.
    reducer : function
        Function of (input, dim)->output that reduce the input tensor along the
        given dimension to get an output tensor. This function will be called
        with the per atom output tensor with internal shape as input, and
        desired reduction dimension as dim, and should reduce the input into
        the tensor containing desired output.
    padding_fill : float
        Default value used to fill padding atoms
    output_length : int
        Length of output of each submodel.
    timers : dict
        Dictionary storing the the benchmark result. It has the following keys:
            forward : total time for the forward pass
    """

    def __init__(self, species, suffixes, reducer, padding_fill, models,
                 benchmark=False): 
        super(ANIModel, self).__init__(benchmark)
        self.species = species
        self.suffixes = suffixes
        self.reducer = reducer
        self.padding_fill = padding_fill
        for i in models:
            setattr(self, i, models[i])

        if benchmark:
            self.forward = self._enable_benchmark(self.forward, 'forward')

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
        aev = aev.flatten(0,1)
        outputs = []
        for suffix in self.suffixes:
            output = torch.full_like(species_, self.padding_fill,
                                     dtype=aev.dtype)
            for i in species.unique().tolist():
                s = self.species[i]
                model_X = getattr(self, 'model_' + s + suffix)
                mask = (species_ == i)
                input = aev.index_select(0, mask.nonzero().squeeze())
                output[mask] = model_X(input).squeeze()
            output = output.view_as(species)
            outputs.append(self.reducer(output, dim=1))

        return species, sum(outputs) / len(outputs)
