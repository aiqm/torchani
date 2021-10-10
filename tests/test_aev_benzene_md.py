import os
import torch
import pickle
import unittest
from torchani.aev import CellList
from common_aev_test import _TestAEVBase

path = os.path.dirname(os.path.realpath(__file__))


class TestAEVBenzeneMD(_TestAEVBase):

    def testBenzeneMD(self):
        for i in [2, 8]:
            datafile = os.path.join(path, 'test_data/benzene-md/{}.dat'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, expected_radial, expected_angular, _, _, cell, pbc \
                    = pickle.load(f)
                coordinates = torch.from_numpy(coordinates).float().unsqueeze(0)
                species = torch.from_numpy(species).unsqueeze(0)
                expected_radial = torch.from_numpy(expected_radial).float().unsqueeze(0)
                expected_angular = torch.from_numpy(expected_angular).float().unsqueeze(0)
                cell = torch.from_numpy(cell).float()
                pbc = torch.from_numpy(pbc).bool()
                _, aev = self.aev_computer((species, coordinates), cell=cell, pbc=pbc)
                self.assertAEVEqual(expected_radial, expected_angular, aev)

    def testBenzeneMDCellList(self):
        for i in [2, 8]:
            datafile = os.path.join(path, 'test_data/benzene-md/{}.dat'.format(i))
            self.aev_computer.neighborlist = CellList(cutoff=self.aev_computer.radial_terms.cutoff)
            with open(datafile, 'rb') as f:
                coordinates, species, expected_radial, expected_angular, _, _, cell, pbc \
                    = pickle.load(f)
                coordinates = torch.from_numpy(coordinates).float().unsqueeze(0)
                species = torch.from_numpy(species).unsqueeze(0)
                expected_radial = torch.from_numpy(expected_radial).float().unsqueeze(0)
                expected_angular = torch.from_numpy(expected_angular).float().unsqueeze(0)
                cell = torch.from_numpy(cell).float()
                pbc = torch.from_numpy(pbc).bool()
                _, aev = self.aev_computer((species, coordinates), cell=cell, pbc=pbc)
                self.assertAEVEqual(expected_radial, expected_angular, aev)


if __name__ == '__main__':
    unittest.main()
