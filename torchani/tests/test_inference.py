import torch
import numpy
import torchani
import unittest
import pkg_resources
import logging
import pyanitools
import ase
import pyNeuroChem
import ase_interface


class TestInference(unittest.TestCase):

    def setUp(self, dtype=torch.cuda.float32):
        self.tolerance = 1e-5
        self.ncaev = torchani.NeuroChemAEV()
        self.nn = torchani.NeuralNetworkOnAEV(
            self.ncaev, from_pync=self.ncaev.network_dir)
        self.logger = logging.getLogger('smiles')
        self.shift_energy = torchani.EnergyShifter(self.ncaev.sae_file)

    def _get_neurochem_energies(self, coordinates, species):
        conformations = coordinates.shape[0]
        nc_energies = []
        for i in range(conformations):
            c = coordinates[i]
            mol = ase.Atoms(''.join(species), positions=c)
            mol.set_calculator(ase_interface.ANI(False))
            mol.calc.setnc(self.ncaev.nc)
            _ = mol.get_potential_energy()
            e = self.ncaev.nc.energy()[0]
            nc_energies.append(e)
        nc_energies = torch.DoubleTensor(nc_energies)
        return nc_energies.type(self.ncaev.dtype)

    def _test_molecule_energy(self, coordinates, species):
        energies = self._get_neurochem_energies(coordinates, species)
        energies = self.shift_energy(energies, species)
        coordinates = torch.from_numpy(coordinates).type(self.ncaev.dtype)
        pred_energies = self.nn(coordinates, species)
        maxdiff = torch.max(torch.abs(pred_energies - energies)).item()
        maxdiff_per_atom = maxdiff / len(species)
        self.assertLess(maxdiff_per_atom, self.tolerance)

    def _test_activations(self, coordinates, species):
        conformations = coordinates.shape[0]
        atoms = coordinates.shape[1]
        for i in range(conformations):
            for j in range(atoms):
                layers = self.nn.layers[species[j]]
                for layer in range(layers):
                    # get activation from NeuroChem
                    c = coordinates[i]
                    mol = ase.Atoms(''.join(species), positions=c)
                    mol.set_calculator(ase_interface.ANI(False))
                    mol.calc.setnc(self.ncaev.nc)
                    _ = mol.get_potential_energy()
                    nca = self.ncaev.nc.activations(j, layer, 0)
                    nca = torch.from_numpy(nca).type(self.ncaev.dtype)
                    # get activation from NeuralNetworkOnAEV
                    a = self.nn.get_activations(
                        c.reshape(1, -1, 3), species, layer)[j].view(-1)
                    # compute diff
                    maxdiff = torch.max(torch.abs(nca - a)).item()
                    self.assertLess(maxdiff, self.tolerance)

    def _test_by_file(self, number):
        data_file = pkg_resources.resource_filename(
            torchani.__name__, 'data/ani_gdb_s0{}.h5'.format(number))
        adl = pyanitools.anidataloader(data_file)
        for data in adl:
            coordinates = data['coordinates'][:10, :]
            species = data['species']
            smiles = ''.join(data['smiles'])
            self._test_activations(coordinates, species)
            self._test_molecule_energy(coordinates, species)
            self.logger.info('Test pass: ' + smiles)

    def testGDB01(self):
        self._test_by_file(1)

    def testGDB02(self):
        self._test_by_file(2)

    def testGDB03(self):
        self._test_by_file(3)

    def testGDB04(self):
        self._test_by_file(4)


if __name__ == '__main__':
    unittest.main()
