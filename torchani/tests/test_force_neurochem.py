import torch
import numpy
import torchani
import unittest
import ase
import pyNeuroChem
import ase_interface
import pyanitools
import pkg_resources
import logging


class TestForceNeuroChem(unittest.TestCase):

    def setUp(self, dtype=torchani.default_dtype, device=torchani.default_device):
        self.tolerance = 1e-5
        self.logger = logging.getLogger('smiles')
        self.ncaev = torchani.NeuroChemAEV(dtype=dtype, device=device)
        self.aev_computer = torchani.AEV(dtype=dtype, device=device)
        self.model = torchani.ModelOnAEV(
            self.aev_computer, derivative=True, device=device, from_nc=None)

    def _test_molecule(self, coordinates, species):
        _, force = self.model(coordinates, species)
        conformations = coordinates.shape[0]
        for i in range(conformations):
            c = coordinates[i]
            mol = ase.Atoms(''.join(species), positions=c)
            mol.set_calculator(ase_interface.ANI(False))
            mol.calc.setnc(self.ncaev.nc)
            _ = mol.get_potential_energy()
            force_nc = self.ncaev.nc.force()
            force_nc = torch.from_numpy(force_nc).type(
                self.aev_computer.dtype).to(self.aev_computer.device)
            max_diff = torch.max(force_nc+force[i])
            self.assertLess(max_diff, self.tolerance)

    def testCH4(self):
        coordinates = torch.tensor([[[0.03192167,  0.00638559,  0.01301679],
                                     [-0.83140486,  0.39370209, -0.26395324],
                                     [-0.66518241, -0.84461308,  0.20759389],
                                     [0.45554739,   0.54289633,  0.81170881],
                                     [0.66091919,  -0.16799635, -0.91037834]],
                                    [[0,  0,  0],
                                     [0,  0,  1],
                                     [1,  0,  0],
                                     [0,  1,  0],
                                     [-1, -1, -1]],
                                    ], dtype=self.aev_computer.dtype, device=self.aev_computer.device)
        species = ['C', 'H', 'H', 'H', 'H']
        self._test_molecule(coordinates, species)

    def _test_by_file(self, number):
        data_file = pkg_resources.resource_filename(
            torchani.__name__, 'data/ani_gdb_s0{}.h5'.format(number))
        adl = pyanitools.anidataloader(data_file)
        for data in adl:
            coordinates = data['coordinates'][:10, :]
            coordinates = torch.from_numpy(coordinates).type(
                self.aev_computer.dtype).to(self.aev_computer.device)
            species = data['species']
            smiles = ''.join(data['smiles'])
            self._test_molecule(coordinates, species)
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
