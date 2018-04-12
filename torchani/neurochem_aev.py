import torch
import ase
import pyNeuroChem
import ase_interface
import pkg_resources
from .aev_base import AEVComputer

default_const_file = pkg_resources.resource_filename(__name__, 'data/rHCNO-4.6R_16-3.1A_a4-8_3.params')
default_sae_file = pkg_resources.resource_filename(__name__, 'data/sae_linfit.dat')
default_network_dir = pkg_resources.resource_filename(__name__, 'data/networks/')

class NeuroChemAEV (AEVComputer):
    
    def __init__(self, dtype=torch.cuda.float32, const_file=default_const_file, sae_file=default_sae_file, network_dir=default_network_dir):
        super(NeuroChemAEV, self).__init__(dtype, const_file)
        self.nc = pyNeuroChem.molecule(const_file, sae_file, network_dir, 0)

    def _get_radial_part(self, fullaev):
        radial_size = self.radial_length()
        return fullaev[:,:,:radial_size]

    def _get_angular_part(self, fullaev):
        radial_size = self.radial_length()
        return fullaev[:,:,radial_size:]

    def _compute_neurochem_aevs_per_conformation(self, coordinates, species):
        atoms = coordinates.shape[0]
        mol = ase.Atoms(''.join(species), positions=coordinates)
        mol.set_calculator(ase_interface.ANI(False))
        mol.calc.setnc(self.nc)
        ei = mol.get_potential_energy()
        aevs = [ self.nc.atomicenvironments(j) for j in range(atoms) ]
        aevs = numpy.stack(aevs)
        return aevs

    def __call__(self, coordinates, species):
        conformations = coordinates.shape[0]
        aevs = [self._compute_neurochem_aevs_per_conformation(coordinates[i], species) for i in range(conformations)]
        aevs = torch.from_numpy(numpy.stack(aevs)).type(self.dtype)
        return self._get_radial_part(aevs), self._get_angular_part(aevs)