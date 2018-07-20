import torch
import os
import ase
import pyNeuroChem
import ase_interface
import numpy
import torchani
import pickle
from torchani import buildin_const_file, buildin_sae_file, buildin_network_dir, default_dtype, default_device
import torchani.pyanitools

path = os.path.dirname(os.path.realpath(__file__))

class NeuroChem (torchani.aev_base.AEVComputer):

    def __init__(self, dtype=default_dtype, device=default_device, const_file=buildin_const_file, sae_file=buildin_sae_file, network_dir=buildin_network_dir):
        super(NeuroChem, self).__init__(False, dtype, device, const_file)
        self.sae_file = sae_file
        self.network_dir = network_dir
        self.nc = pyNeuroChem.molecule(const_file, sae_file, network_dir, 0)

    def _get_radial_part(self, fullaev):
        radial_size = self.radial_length
        return fullaev[:, :, :radial_size]

    def _get_angular_part(self, fullaev):
        radial_size = self.radial_length
        return fullaev[:, :, radial_size:]

    def _per_conformation(self, coordinates, species):
        atoms = coordinates.shape[0]
        mol = ase.Atoms(''.join(species), positions=coordinates)
        mol.set_calculator(ase_interface.ANI(False))
        mol.calc.setnc(self.nc)
        energy = mol.get_potential_energy()
        force = mol.get_forces()
        aevs = [self.nc.atomicenvironments(j) for j in range(atoms)]
        aevs = numpy.stack(aevs)
        return aevs, energy, force

    def forward(self, coordinates, species):
        conformations = coordinates.shape[0]
        results = [self._per_conformation(coordinates[i], species) for i in range(conformations)]
        aevs, energies, forces = zip(*results)
        aevs = torch.from_numpy(numpy.stack(aevs)).type(self.dtype).to(self.device)
        energies = torch.from_numpy(numpy.stack(energies)).type(self.dtype).to(self.device)
        forces = torch.from_numpy(numpy.stack(forces)).type(self.dtype).to(self.device)
        return self._get_radial_part(aevs), self._get_angular_part(aevs), energies, forces

ncaev = NeuroChem()
mol_count = 0

for i in [1,2,3,4]:
    data_file = os.path.join(path, 'dataset/ani_gdb_s0{}.h5'.format(i))
    adl = torchani.pyanitools.anidataloader(data_file)
    for data in adl:
        coordinates = data['coordinates'][:10, :]
        species = data['species']
        smiles = ''.join(data['smiles'])
        radial, angular, energies, forces = ncaev(coordinates, species)
        pickleobj = (coordinates, species, radial, angular, energies, forces)
        dumpfile = os.path.join(path, 'test_data/{}'.format(mol_count))
        with open(dumpfile, 'wb') as f:
            pickle.dump(pickleobj, f)
        mol_count += 1
        
