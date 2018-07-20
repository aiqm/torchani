import torch
import numpy
import torchani
import unittest
import os
import logging
import pyanitools
import ase
import pyNeuroChem
import ase_interface


class TestInference(unittest.TestCase):

    def setUp(self, dtype=torchani.default_dtype, device=torchani.default_device):
        self.tolerance = 1e-5
        self.ncaev = torchani.NeuroChemAEV(dtype=dtype, device=device)
        self.nn = torchani.ModelOnAEV(
            self.ncaev, from_nc=self.ncaev.network_dir)
        self.nn1 = torchani.ModelOnAEV(self.ncaev, from_nc=None)
        self.nn2 = torchani.ModelOnAEV(
            self.ncaev, from_nc=torchani.buildin_model_prefix, ensemble=1)
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
        return nc_energies.type(self.ncaev.dtype).to(self.ncaev.device)

    def _test_molecule_energy(self, coordinates, species):
        energies = self._get_neurochem_energies(coordinates, species)
        energies = self.shift_energy.subtract_sae(energies, species)
        coordinates = torch.from_numpy(coordinates).type(
            self.ncaev.dtype).to(self.ncaev.device)
        pred_energies1 = self.nn1(coordinates, species).squeeze()
        pred_energies2 = self.nn2(coordinates, species).squeeze()
        maxdiff1 = torch.max(torch.abs(pred_energies1 - energies)).item()
        maxdiff2 = torch.max(torch.abs(pred_energies2 - energies)).item()
        maxdiff = max(maxdiff1, maxdiff2)
        maxdiff_per_atom = maxdiff / len(species)
        self.assertLess(maxdiff_per_atom, self.tolerance)

    def _test_activations(self, coordinates, species):
        conformations = coordinates.shape[0]
        atoms = coordinates.shape[1]
        radial_aev, angular_aev = self.nn.aev_computer(coordinates, species)
        aev = torch.cat([radial_aev, angular_aev], dim=2)
        for i in range(conformations):
            for j in range(atoms):
                model_X = getattr(self.nn, 'model_' + species[j])
                layers = model_X.layers
                for layer in range(layers):
                    # get activation from NeuroChem
                    c = coordinates[i]
                    mol = ase.Atoms(''.join(species), positions=c)
                    mol.set_calculator(ase_interface.ANI(False))
                    mol.calc.setnc(self.ncaev.nc)
                    _ = mol.get_potential_energy()
                    nca = self.ncaev.nc.activations(j, layer, 0)
                    nca = torch.from_numpy(nca).type(
                        self.ncaev.dtype).to(self.ncaev.device)
                    # get activation from ModelOnAEV
                    atom_aev = aev[:, j, :]
                    a = model_X.get_activations(atom_aev, layer)
                    a = a[i].view(-1)
                    # compute diff
                    maxdiff = torch.max(torch.abs(nca - a)).item()
                    self.assertLess(maxdiff, self.tolerance)

    def _test_by_file(self, number):
        data_file = os.path.join(
            torchani.buildin_dataset_dir, 'ani_gdb_s0{}.h5'.format(number))
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
