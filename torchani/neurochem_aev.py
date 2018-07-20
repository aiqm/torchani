import torch
import ase
import pyNeuroChem
import ase_interface
import numpy
from .aev_base import AEVComputer
from . import buildin_const_file, buildin_sae_file, buildin_network_dir, default_dtype, default_device


class NeuroChemAEV (AEVComputer):
    """The AEV computer that dump out AEV from pyNeuroChem

    Attributes
    ----------
    sae_file : str
        The name of the original file that stores self atomic energies.
    network_dir : str
        The name ending with '/' of the directory that stores networks in NeuroChem's format.
    nc : pyNeuroChem.molecule
        The internal object of pyNeuroChem which can be used to dump out AEVs, energies, forces,
        activations, etc.
    """

    def __init__(self, dtype=default_dtype, device=default_device, const_file=buildin_const_file, sae_file=buildin_sae_file, network_dir=buildin_network_dir):
        super(NeuroChemAEV, self).__init__(False, dtype, device, const_file)
        self.sae_file = sae_file
        self.network_dir = network_dir
        self.nc = pyNeuroChem.molecule(const_file, sae_file, network_dir, 0)

    def _get_radial_part(self, fullaev):
        """Get the radial part of AEV from the full AEV

        Parameters
        ----------
        fullaev : pytorch tensor of `dtype`
            The full AEV in shape (conformations, atoms, `radial_length()+angular_length()`).

        Returns
        -------
        pytorch tensor of `dtype`
            The radial AEV in shape(conformations, atoms, `radial_length()`)
        """
        radial_size = self.radial_length
        return fullaev[:, :, :radial_size]

    def _get_angular_part(self, fullaev):
        """Get the angular part of AEV from the full AEV

        Parameters
        ----------
        fullaev : pytorch tensor of `dtype`
            The full AEV in shape (conformations, atoms, `radial_length()+angular_length()`).

        Returns
        -------
        pytorch tensor of `dtype`
            The radial AEV in shape (conformations, atoms, `angular_length()`)
        """
        radial_size = self.radial_length
        return fullaev[:, :, radial_size:]

    def _compute_neurochem_aevs_per_conformation(self, coordinates, species):
        """Get the full AEV for a single conformation

        Parameters
        ----------
        coordinates : pytorch tensor of `dtype`
            The xyz coordinates in shape (atoms, 3).
        species : list of str
            The list specifying the species of each atom. The length of the
            list must be the same as the number of atoms.

        Returns
        -------
        pytorch tensor of `dtype`
            The full AEV for all atoms in shape (atoms, `radial_length()+angular_length()`)
        """
        atoms = coordinates.shape[0]
        mol = ase.Atoms(''.join(species), positions=coordinates)
        mol.set_calculator(ase_interface.ANI(False))
        mol.calc.setnc(self.nc)
        _ = mol.get_potential_energy()
        aevs = [self.nc.atomicenvironments(j) for j in range(atoms)]
        aevs = numpy.stack(aevs)
        return aevs

    def __call__(self, coordinates, species):
        conformations = coordinates.shape[0]
        aevs = [self._compute_neurochem_aevs_per_conformation(
            coordinates[i], species) for i in range(conformations)]
        aevs = torch.from_numpy(numpy.stack(aevs)).type(
            self.dtype).to(self.device)
        return self._get_radial_part(aevs), self._get_angular_part(aevs)
