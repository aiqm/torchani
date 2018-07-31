from ..aev import AEVComputer
import torch
from ..benchmarked import BenchmarkedModule


class ANIModel(BenchmarkedModule):
    """Subclass of `torch.nn.Module` for the [xyz]->[aev]->[per_atom_y]->y
    pipeline.

    Attributes
    ----------
    aev_computer : AEVComputer
        The AEV computer.
    output_length : int
        The length of output vector
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
    derivative : boolean
        Whether to support computing the derivative w.r.t coordinates,
        i.e. d(output)/dR
    derivative_graph : boolean
        Whether to generate a graph for the derivative. This would be required
        only if the derivative is included as part of the loss function.
    timers : dict
        Dictionary storing the the benchmark result. It has the following keys:
            aev : time spent on computing AEV.
            nn : time spent on computing output from AEV.
            derivative : time spend on computing derivative w.r.t. coordinates
                after the outputs is given. This key is only available if
                derivative computation is turned on.
            forward : total time for the forward pass
    """

    def __init__(self, aev_computer, suffixes, reducer, output_length, models,
                 benchmark=False):
        super(ANIModel, self).__init__(benchmark)
        if not isinstance(aev_computer, AEVComputer):
            raise TypeError(
                "ModelOnAEV: aev_computer must be a subclass of AEVComputer")
        self.aev_computer = aev_computer

        self.suffixes = suffixes
        self.reducer = reducer
        self.output_length = output_length
        for i in models:
            setattr(self, i, models[i])

        if benchmark:
            self.aev_to_output = self._enable_benchmark(
                self.aev_to_output, 'nn')
            self.forward = self._enable_benchmark(self.forward, 'forward')

    def aev_to_output(self, aev, species):
        """Compute output from aev

        Parameters
        ----------
        aev : torch.Tensor
            Pytorch tensor of shape (conformations, atoms, aev_length) storing
            the computed AEVs.
        species : torch.Tensor
            Tensor storing the species for each atom.

        Returns
        -------
        torch.Tensor
            Pytorch tensor of shape (conformations, output_length) for the
            output of each conformation.
        """
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
            y = aev[:, begin:end, :].reshape(-1, self.aev_computer.aev_length)

            def apply_model(suffix):
                model_X = getattr(self, 'model_' +
                                  self.aev_computer.species[s] + suffix)
                return model_X(y)
            ys = [apply_model(suffix) for suffix in self.suffixes]
            y = sum(ys) / len(ys)
            y = y.view(conformations, -1, self.output_length)
            per_species_outputs.append(y)

        per_species_outputs = torch.cat(per_species_outputs, dim=1)
        molecule_output = self.reducer(per_species_outputs, dim=1)
        return molecule_output

    def forward(self, coordinates, species):
        """Feed forward

        Parameters
        ----------
        coordinates : torch.Tensor
            The pytorch tensor of shape (conformations, atoms, 3) storing
            the coordinates of all atoms of all conformations.
        species : list of string
            List of string storing the species for each atom.

        Returns
        -------
        torch.Tensor
            Tensor of shape (conformations, output_length) for the
            output of each conformation.
        """
        species = self.aev_computer.species_to_tensor(species)
        _species, _coordinates, = self.aev_computer.sort_by_species(
            species, coordinates)
        aev = self.aev_computer((_coordinates, _species))
        return self.aev_to_output(aev, _species)
