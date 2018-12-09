import torch
import os
import ase
import pyNeuroChem
import ase_interface
import numpy
import torchani
import pickle
from torchani import buildin_const_file, buildin_sae_file, \
    buildin_network_dir
import json

path = os.path.dirname(os.path.realpath(__file__))
conv_au_ev = 27.21138505


class NeuroChem (torchani.aev.AEVComputer):

    def __init__(self, const_file=buildin_const_file,
                 sae_file=buildin_sae_file,
                 network_dir=buildin_network_dir):
        super(NeuroChem, self).__init__(False, const_file)
        self.sae_file = sae_file
        self.network_dir = network_dir
        self.nc = pyNeuroChem.molecule(
            self.const_file, self.sae_file, self.network_dir, 0)

    def _get_radial_part(self, fullaev):
        return fullaev[:, :, :self.radial_length]

    def _get_angular_part(self, fullaev):
        return fullaev[:, :, self.radial_length:]

    def _per_conformation(self, coordinates, species):
        atoms = coordinates.shape[0]
        mol = ase.Atoms(''.join(species), positions=coordinates)
        mol.set_calculator(ase_interface.ANI(False))
        mol.calc.setnc(self.nc)
        energy = mol.get_potential_energy() / conv_au_ev
        aevs = [mol.calc.nc.atomicenvironments(j) for j in range(atoms)]
        force = mol.get_forces() / conv_au_ev
        aevs = numpy.stack(aevs)
        return aevs, energy, force

    def forward(self, coordinates, species):
        conformations = coordinates.shape[0]
        results = [self._per_conformation(
            coordinates[i], species) for i in range(conformations)]
        aevs, energies, forces = zip(*results)
        aevs = torch.from_numpy(numpy.stack(aevs)).type(
            self.EtaR.dtype).to(self.EtaR.device)
        energies = torch.from_numpy(numpy.stack(energies)).type(
            self.EtaR.dtype).to(self.EtaR.device)
        forces = torch.from_numpy(numpy.stack(forces)).type(
            self.EtaR.dtype).to(self.EtaR.device)
        return self._get_radial_part(aevs), \
            self._get_angular_part(aevs), \
            energies, forces


consts = torchani.neurochem.Constants()
aev = torchani.AEVComputer(**consts)
ncaev = NeuroChem().to(torch.device('cpu'))
mol_count = 0


species_indices = {consts.species[i]: i for i in range(len(aev.species))}
for i in [1, 2, 3, 4]:
    data_file = os.path.join(
        path, '../dataset/ani_gdb_s0{}.h5'.format(i))
    adl = torchani.training.pyanitools.anidataloader(data_file)
    for data in adl:
        coordinates = data['coordinates'][:10, :]
        coordinates = torch.from_numpy(coordinates).type(ncaev.EtaR.dtype)
        species = torch.tensor([species_indices[i] for i in data['species']],
                               dtype=torch.long, device=torch.device('cpu')) \
                       .expand(10, -1)
        smiles = ''.join(data['smiles'])
        radial, angular, energies, forces = ncaev(coordinates, data['species'])
        pickleobj = (coordinates, species, radial, angular, energies, forces)
        dumpfile = os.path.join(
            path, '../tests/test_data/{}'.format(mol_count))
        with open(dumpfile, 'wb') as f:
            pickle.dump(pickleobj, f, protocol=2)
        mol_count += 1
