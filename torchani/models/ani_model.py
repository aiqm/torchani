from ..aev_base import AEVComputer
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
                 derivative=False, derivative_graph=False, benchmark=False):
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

        self.derivative = derivative
        if not derivative and derivative_graph:
            raise ValueError(
                '''BySpeciesModel: can not create graph for derivative if the
                computation of derivative is turned off''')
        self.derivative_graph = derivative_graph
        if derivative and self.output_length != 1:
            raise ValueError(
                'derivative can only be computed for output length 1')

        if benchmark:
            self.compute_aev = self._enable_benchmark(self.compute_aev, 'aev')
            self.aev_to_output = self._enable_benchmark(
                self.aev_to_output, 'nn')
            if derivative:
                self.compute_derivative = self._enable_benchmark(
                    self.compute_derivative, 'derivative')
            self.forward = self._enable_benchmark(self.forward, 'forward')

    def compute_aev(self, coordinates, species):
        """Compute full AEV

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
            Pytorch tensor of shape (conformations, atoms, aev_length) storing
            the computed AEVs.
        """
        radial_aev, angular_aev = self.aev_computer(coordinates, species)
        fullaev = torch.cat([radial_aev, angular_aev], dim=2)
        return fullaev

    def aev_to_output(self, aev, species):
        """Compute output from aev

        Parameters
        ----------
        aev : torch.Tensor
            Pytorch tensor of shape (conformations, atoms, aev_length) storing
            the computed AEVs.
        species : list of string
            List of string storing the species for each atom.

        Returns
        -------
        torch.Tensor
            Pytorch tensor of shape (conformations, output_length) for the
            output of each conformation.
        """
        conformations = aev.shape[0]
        atoms = len(species)
        rev_species = species[::-1]
        species_dedup = sorted(
            set(species), key=self.aev_computer.species.index)
        per_species_outputs = []
        for s in species_dedup:
            begin = species.index(s)
            end = atoms - rev_species.index(s)
            y = aev[:, begin:end, :].reshape(-1, self.aev_computer.aev_length)

            def apply_model(suffix):
                model_X = getattr(self, 'model_' + s + suffix)
                return model_X(y)
            ys = [apply_model(suffix) for suffix in self.suffixes]
            y = sum(ys) / len(ys)
            y = y.view(conformations, -1, self.output_length)
            per_species_outputs.append(y)

        per_species_outputs = torch.cat(per_species_outputs, dim=1)
        molecule_output = self.reducer(per_species_outputs, dim=1)
        return molecule_output

    def compute_derivative(self, output, coordinates):
        """Compute the gradient d(output)/d(coordinates)"""
        # Since different conformations are independent, computing
        # the derivatives of all outputs w.r.t. its own coordinate is
        # equivalent to compute the derivative of the sum of all outputs
        # w.r.t. all coordinates.
        return torch.autograd.grad(output.sum(), coordinates,
                                   create_graph=self.derivative_graph)[0]

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
        torch.Tensor or (torch.Tensor, torch.Tensor)
            If derivative is turned off, then this function will return a
            pytorch tensor of shape (conformations, output_length) for the
            output of each conformation.
            If derivative is turned on, then this function will return a pair
            of pytorch tensors where the first tensor is the output tensor as
            when the derivative is off, and the second tensor is a tensor of
            shape (conformation, atoms, 3) storing the d(output)/dR.
        """
        if not self.derivative:
            coordinates = coordinates.detach()
        else:
            coordinates = torch.tensor(coordinates, requires_grad=True)
        _coordinates, _species = self.aev_computer.sort_by_species(
            coordinates, species)
        aev = self.compute_aev(_coordinates, _species)
        output = self.aev_to_output(aev, _species)
        if not self.derivative:
            return output
        else:
            derivative = self.compute_derivative(output, coordinates)
            return output, derivative
