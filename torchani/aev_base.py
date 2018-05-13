import torch
import torch.nn as nn
import numpy
from . import buildin_const_file, default_dtype, default_device
from .benchmarked import BenchmarkedModule


class AEVComputer(BenchmarkedModule):
    """Base class of various implementations of AEV computer

    Attributes
    ----------
    benchmark : boolean
        Whether to enable benchmark
    dtype : torch.dtype
        Data type of pytorch tensors for all the computations. This is also used
        to specify whether to use CPU or GPU.
    device : torch.Device
        The device where tensors should be.
    const_file : str
        The name of the original file that stores constant.
    Rcr, Rca : float
        Cutoff radius
    EtaR, ShfR, Zeta, ShfZ, EtaA, ShfA : torch.Tensor
        Tensor storing constants.
    """

    def __init__(self, benchmark=False, dtype=default_dtype, device=default_device, const_file=buildin_const_file):
        super(AEVComputer, self).__init__(benchmark)

        self.dtype = dtype
        self.const_file = const_file
        self.device = device

        # load constants from const file
        with open(const_file) as f:
            for i in f:
                try:
                    line = [x.strip() for x in i.split('=')]
                    name = line[0]
                    value = line[1]
                    if name == 'Rcr' or name == 'Rca':
                        setattr(self, name, float(value))
                    elif name in ['EtaR', 'ShfR', 'Zeta', 'ShfZ', 'EtaA', 'ShfA']:
                        value = [float(x.strip()) for x in value.replace(
                            '[', '').replace(']', '').split(',')]
                        value = torch.tensor(value, dtype=dtype, device=device)
                        setattr(self, name, value)
                    elif name == 'Atyp':
                        value = [x.strip() for x in value.replace(
                            '[', '').replace(']', '').split(',')]
                        self.species = value
                except:
                    raise ValueError('unable to parse const file')

    @staticmethod
    def _cutoff_cosine(distances, cutoff):
        """Compute the elementwise cutoff cosine function

        The cutoff cosine function is define in https://arxiv.org/pdf/1610.08935.pdf equation 2

        Parameters
        ----------
        distances : torch.Tensor
            The pytorch tensor that stores Rij values. This tensor can have any shape since the cutoff
            cosine function is computed elementwise.
        cutoff : float
            The cutoff radius, i.e. the Rc in the equation. For any Rij > Rc, the function value is defined to be zero.

        Returns
        -------
        torch.Tensor
            The tensor of the same shape as `distances` that stores the computed function values.
        """
        return torch.where(distances <= cutoff, 0.5 * torch.cos(numpy.pi * distances / cutoff) + 0.5, torch.zeros_like(distances))

    def per_species_radial_length(self):
        """Returns the radial subaev length per species"""
        return self.EtaR.shape[0] * self.ShfR.shape[0]

    def radial_length(self):
        """Returns the full radial aev length"""
        return len(self.species) * self.per_species_radial_length()

    def per_species_angular_length(self):
        """Returns the angular subaev length per species"""
        return self.EtaA.shape[0] * self.Zeta.shape[0] * self.ShfA.shape[0] * self.ShfZ.shape[0]

    def angular_length(self):
        """Returns the full angular aev length"""
        species = len(self.species)
        return int((species * (species + 1)) / 2) * self.per_species_angular_length()

    def aev_length(self):
        return self.radial_length() + self.angular_length()

    def forward(self, coordinates, species):
        """Compute AEV from coordinates and species

        Parameters
        ----------
        coordinates : torch.Tensor
            The tensor that specifies the xyz coordinates of atoms in the molecule.
            The tensor must have shape (conformations, atoms, 3)
        species : list of string
            The list that specifies the species of each atom. The length of the list
            must match with `coordinates.shape[1]`.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Returns (radial AEV, angular AEV), both are pytorch tensor of `dtype`.
            The radial AEV must be of shape (conformations, atoms, radial_length())
            The angular AEV must be of shape (conformations, atoms, angular_length())
        """
        raise NotImplementedError('subclass must override this method')
