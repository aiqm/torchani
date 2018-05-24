import torch
import numpy
import torchani
import unittest
import pyanitools
import os
import logging


class TestAEV(unittest.TestCase):

    def setUp(self, dtype=torchani.default_dtype, device=torchani.default_device):
        self.aev = torchani.SortedAEV(dtype=dtype, device=device)
        self.ncaev = torchani.NeuroChemAEV(dtype=dtype, device=device)
        self.tolerance = 1e-5
        self.logger = logging.getLogger('smiles')

    def _test_molecule(self, coordinates, species):
        coordinates = torch.from_numpy(coordinates).type(
            self.aev.dtype).to(self.aev.device)
        coordinates, species = self.aev.sort_by_species(coordinates, species)
        radial_neurochem, angular_neurochem = self.ncaev(coordinates, species)
        radial_torchani, angular_torchani = self.aev(coordinates, species)
        radial_diff = radial_neurochem - radial_torchani
        radial_max_error = torch.max(torch.abs(radial_diff))
        angular_diff = angular_neurochem - angular_torchani
        angular_max_error = torch.max(torch.abs(angular_diff))
        if radial_max_error > self.tolerance:
            print(species)
            for i in range(len(species)):
                r1 = radial_neurochem[0, i, :]
                r2 = radial_torchani[0, i, :]
                max_err = torch.max(torch.abs(r1 - r2))
                if max_err > self.tolerance:
                    print('atom', i, 'species', species[i], 'radial aevs:')
                    print(torch.stack([r1, r2, r1-r2], dim=1))
                    break
        if angular_max_error > self.tolerance:
            print(species)
            for i in range(len(species)):
                r1 = angular_neurochem[0, i, :]
                r2 = angular_torchani[0, i, :]
                max_err = torch.max(torch.abs(r1 - r2))
                if max_err > self.tolerance:
                    print('atom', i, 'species', species[i], 'angular aevs:')
                    print(torch.stack([r1, r2, r1-r2], dim=1))
                    break
        self.assertLess(radial_max_error, self.tolerance)
        self.assertLess(angular_max_error, self.tolerance)

    def _test_datafile(self, number):
        data_file = os.path.join(torchani.buildin_dataset_dir, 'ani_gdb_s0{}.h5'.format(number))
        adl = pyanitools.anidataloader(data_file)
        for data in adl:
            coordinates = data['coordinates'][:10, :]
            species = data['species']
            smiles = ''.join(data['smiles'])
            self._test_molecule(coordinates, species)
            self.logger.info('Test pass: ' + smiles)

    def testGDB01(self):
        self._test_datafile(1)

    def testGDB02(self):
        self._test_datafile(2)

    def testGDB03(self):
        self._test_datafile(3)

    def testGDB04(self):
        self._test_datafile(4)

    def testCH4(self):
        # return
        coordinates = numpy.array([[[0.03192167,  0.00638559,  0.01301679],
                                    [-0.83140486,  0.39370209, -0.26395324],
                                    [-0.66518241, -0.84461308,  0.20759389],
                                    [0.45554739,  0.54289633,  0.81170881],
                                    [0.66091919, -0.16799635, -0.91037834]]], numpy.float32)
        species = ['C', 'H', 'H', 'H', 'H']
        self._test_molecule(coordinates, species)


if __name__ == '__main__':
    unittest.main()
