import torch
import torch.nn as nn
from . import buildin_const_file, default_dtype, default_device
from .benchmarked import BenchmarkedModule


class AEVComputer(BenchmarkedModule):
    __constants__ = ['Rcr', 'Rca', 'dtype', 'device', 'radial_sublength',
        'radial_length', 'angular_sublength', 'angular_length', 'aev_length']

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
    radial_sublength : int
        The length of radial subaev of a single species
    radial_length : int
        The length of full radial aev
    angular_sublength : int
        The length of angular subaev of a single species
    angular_length : int
        The length of full angular aev
    aev_length : int
        The length of full aev
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

        # Compute lengths
        self.radial_sublength = self.EtaR.shape[0] * self.ShfR.shape[0]
        self.radial_length = len(self.species) * self.radial_sublength
        self.angular_sublength = self.EtaA.shape[0] * \
            self.Zeta.shape[0] * self.ShfA.shape[0] * self.ShfZ.shape[0]
        species = len(self.species)
        self.angular_length = int(
            (species * (species + 1)) / 2) * self.angular_sublength
        self.aev_length = self.radial_length + self.angular_length

        # convert constant tensors to a ready-to-broadcast shape
        # shape convension (..., EtaR, ShfR)
        self.EtaR = self.EtaR.view(-1, 1)
        self.ShfR = self.ShfR.view(1, -1)
        # shape convension (..., EtaA, Zeta, ShfA, ShfZ)
        self.EtaA = self.EtaA.view(-1, 1, 1, 1)
        self.Zeta = self.Zeta.view(1, -1, 1, 1)
        self.ShfA = self.ShfA.view(1, 1, -1, 1)
        self.ShfZ = self.ShfZ.view(1, 1, 1, -1)

    def sort_by_species(self, data, species):
        """Sort the data by its species according to the order in `self.species`

        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape (conformations, atoms, ...) for data.
        species : list
            List storing species of each atom.

        Returns
        -------
        (torch.Tensor, list)
            Tuple of (sorted data, sorted species).
        """
        atoms = list(zip(species, torch.unbind(data, 1)))
        atoms = sorted(atoms, key=lambda x: self.species.index(x[0]))
        species = [s for s, _ in atoms]
        data = torch.stack([c for _, c in atoms], dim=1)
        return data, species

    def forward(self, coordinates, species):
        """Compute AEV from coordinates and species

        Parameters
        ----------
        coordinates : torch.Tensor
            The tensor that specifies the xyz coordinates of atoms in the molecule.
            The tensor must have shape (conformations, atoms, 3)
        species : torch.LongTensor
            Long tensor for the species, where a value k means the species is
            the same as self.species[k]

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Returns (radial AEV, angular AEV), both are pytorch tensor of `dtype`.
            The radial AEV must be of shape (conformations, atoms, radial_length)
            The angular AEV must be of shape (conformations, atoms, angular_length)
        """
        raise NotImplementedError('subclass must override this method')
