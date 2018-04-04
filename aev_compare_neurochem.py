import numpy
import torch
import pyanitools
import ase
import pyNeuroChem
import ase_interface
import torchani
import unittest

def sorted_diff(a, b):
    sorted_a, _ = torch.sort(a)
    sorted_b, _ = torch.sort(b)
    return sorted_a - sorted_b

class TestAgainstNeuroChem(unittest.TestCase):

    def setUp(self):
        self.nc = pyNeuroChem.molecule('data/rHCNO-4.6R_16-3.1A_a4-8_3.params', 'data/sae_linfit.dat', 'data/networks/', 0)
        self.ani = torchani.ani('data/rHCNO-4.6R_16-3.1A_a4-8_3.params', 'data/sae_linfit.dat')
        self.tolerance = 1e-5

    def _compute_neurochem_aevs_per_conformation(self, coordinates, species):
        atoms = coordinates.shape[0]
        mol = ase.Atoms(''.join(species), positions=coordinates)
        mol.set_calculator(ase_interface.ANI(False))
        mol.calc.setnc(self.nc)
        ei = mol.get_potential_energy()
        aevs = [ self.nc.atomicenvironments(j) for j in range(atoms) ]
        aevs = numpy.stack(aevs)
        return aevs

    def _compute_neurochem_aevs(self, coordinates, species):
        conformations = coordinates.shape[0]
        aevs = [self._compute_neurochem_aevs_per_conformation(coordinates[i], species) for i in range(conformations)]
        aevs = torch.from_numpy(numpy.stack(aevs)).cuda()
        return self._get_radial_part(aevs), self._get_angular_part(aevs)

    def _get_radial_part(self, fullaev):
        radial_size = self.ani.radial_length() * len(self.ani.species)
        return fullaev[:,:,:radial_size]

    def _get_angular_part(self, fullaev):
        radial_size = self.ani.radial_length() * len(self.ani.species)
        return fullaev[:,:,radial_size:]

    def _test_molecule(self, coordinates, species):
        radial_neurochem, angular_neurochem = self._compute_neurochem_aevs(coordinates, species)
        radial_torchani, angular_torchani = self.ani.compute_aev(torch.from_numpy(coordinates).cuda(), species)
        radial_diff = sorted_diff(radial_neurochem, radial_torchani)
        radial_max_error = torch.max(torch.abs(radial_diff))
        angular_diff = sorted_diff(angular_neurochem, angular_torchani)
        angular_max_error = torch.max(torch.abs(angular_diff))
        self.assertLess(radial_max_error, self.tolerance)
        self.assertLess(angular_max_error, self.tolerance)

    def testGDB05(self):
        adl = pyanitools.anidataloader("data/ani_gdb_s05.h5")
        for data in adl:
            coordinates = data['coordinates']
            species = data['species']
            smiles = ''.join(data['smiles'])
            self._test_molecule(coordinates, species)
            print('Test pass:', smiles)

    def testCH4(self):
        coordinates = numpy.array([[[ 0.03192167,  0.00638559,  0.01301679],
                                    [-0.83140486,  0.39370209, -0.26395324],
                                    [-0.66518241, -0.84461308,  0.20759389],
                                    [ 0.45554739,  0.54289633,  0.81170881],
                                    [ 0.66091919, -0.16799635, -0.91037834]]], numpy.float32)
        species = ['C', 'H', 'H', 'H', 'H']
        self._test_molecule(coordinates, species)


if __name__ == '__main__':
    unittest.main()