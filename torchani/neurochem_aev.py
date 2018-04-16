import torch
import ase
import pyNeuroChem
import ase_interface
import numpy
from .aev_base import AEVComputer
from . import buildin_const_file, buildin_sae_file, buildin_network_dir


class NeuroChemAEV (AEVComputer):

    def __init__(self, dtype=torch.cuda.float32, const_file=buildin_const_file, sae_file=buildin_sae_file, network_dir=buildin_network_dir):
        super(NeuroChemAEV, self).__init__(dtype, const_file)
        self.const_file = const_file
        self.sae_file = sae_file
        self.network_dir = network_dir
        self.nc = pyNeuroChem.molecule(const_file, sae_file, network_dir, 0)

    def _get_radial_part(self, fullaev):
        radial_size = self.radial_length()
        return fullaev[:, :, :radial_size]

    def _get_angular_part(self, fullaev):
        radial_size = self.radial_length()
        return fullaev[:, :, radial_size:]

    def _compute_neurochem_aevs_per_conformation(self, coordinates, species):
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
        aevs = torch.from_numpy(numpy.stack(aevs)).type(self.dtype)
        return self._get_radial_part(aevs), self._get_angular_part(aevs)
