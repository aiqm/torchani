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
        self.nn = torchani.NeuralNetworkOnAEV(self.ncaev, from_pync=self.ncaev.network_dir)
        self.logger = logging.getLogger('smiles')
    
    def _get_neurochem_energies(self, coordinates, species):
        conformations = coordinates.shape[0]
        nc_energies = []
        for i in range(conformations):
            c = coordinates[i]
            mol = ase.Atoms(''.join(species), positions=c)
            mol.set_calculator(ase_interface.ANI(False))
            mol.calc.setnc(self.ncaev.nc)
            e = mol.get_potential_energy()
            nc_energies.append(e)
        nc_energies = torch.DoubleTensor(nc_energies)
        nc_energies /= 627.509
        return nc_energies.type(self.ncaev.dtype)

    def _test_molecule(self, coordinates, species):
        energies = self._get_neurochem_energies(coordinates, species)
        coordinates = torch.from_numpy(coordinates).type(self.ncaev.dtype)
        pred_energies = self.nn(coordinates, species)
        energy_diff = pred_energies - energies
        print(pred_energies, energies)

    def _test_datafile(self, number):
        data_file = pkg_resources.resource_filename(torchani.__name__, 'data/ani_gdb_s0{}.h5'.format(number))
        adl = pyanitools.anidataloader(data_file)
        for data in adl:
            coordinates = data['coordinates'][:10,:]
            species = data['species']
            smiles = ''.join(data['smiles'])
            print(smiles)
            self._test_molecule(coordinates, species)
            self.logger.info('Test pass: ' + smiles)

    def testGDB01(self):
        self._test_datafile(1)

    # def testGDB02(self):
    #     self._test_datafile(2)

    # def testGDB03(self):
    #     self._test_datafile(3)
    
    # def testGDB04(self):
    #     self._test_datafile(4)

if __name__ == '__main__':
    unittest.main()