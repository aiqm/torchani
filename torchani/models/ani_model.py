import torch
from ..benchmarked import BenchmarkedModule


class ANIModel(BenchmarkedModule):
    """Subclass of `torch.nn.Module` for the [xyz]->[aev]->[per_atom_y]->y
    pipeline.

    Attributes
    ----------
    species : list
        Chemical symbol of supported atom species.
    output_length : int
        The length of output vector.
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
    output_length : int
        Length of output of each submodel.
    timers : dict
        Dictionary storing the the benchmark result. It has the following keys:
            forward : total time for the forward pass
    """

    def __init__(self, species, suffixes, reducer, output_length, models,
                 benchmark=False):
        super(ANIModel, self).__init__(benchmark)
        self.species = species
        self.suffixes = suffixes
        self.reducer = reducer
        self.output_length = output_length
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
        torch.Tensor
            Pytorch tensor of shape (conformations, output_length) for the
            output of each conformation.
        """
        species, aev = species_aev
        conformations = aev.shape[0]
        atoms = len(species)
        rev_species = species.__reversed__()
        species_dedup = species.unique()
        per_species_outputs = []
        species = species.tolist()
        rev_species = rev_species.tolist()
        for s in species_dedup:
            begin = species.index(s)
            end = atoms - rev_species.index(s)
            y = aev[:, begin:end, :].flatten(0, 1)

            def apply_model(suffix):
                model_X = getattr(self, 'model_' +
                                  self.species[s] + suffix)
                return model_X(y)
            ys = [apply_model(suffix) for suffix in self.suffixes]
            y = sum(ys) / len(ys)
            y = y.view(conformations, -1, self.output_length)
            per_species_outputs.append(y)

        per_species_outputs = torch.cat(per_species_outputs, dim=1)
        molecule_output = self.reducer(per_species_outputs, dim=1)
        return molecule_output
