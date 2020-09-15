import torch
import torchani
import unittest
import os
import pickle


path = os.path.dirname(os.path.realpath(__file__))
N = 97


class TestEnergies(unittest.TestCase):

    def setUp(self):
        self.tolerance = 5e-5
        model = torchani.models.ANI1x(model_index=0)
        self.aev_computer = model.aev_computer
        self.nnp = model.neural_networks
        self.energy_shifter = model.energy_shifter
        self.nn = torchani.nn.Sequential(self.nnp, self.energy_shifter)
        self.model = torchani.nn.Sequential(self.aev_computer, self.nnp, self.energy_shifter)

    def testIsomers(self):
        for i in range(N):
            datafile = os.path.join(path, 'test_data/ANI1_subset/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, energies, _ = pickle.load(f)
                coordinates = torch.from_numpy(coordinates).to(torch.float)
                species = torch.from_numpy(species)
                energies = torch.from_numpy(energies).to(torch.float)
                energies_ = self.model((species, coordinates)).energies
                max_diff = (energies - energies_).abs().max().item()
                self.assertLess(max_diff, self.tolerance)

    def testPadding(self):
        species_coordinates = []
        energies = []
        for i in range(N):
            datafile = os.path.join(path, 'test_data/ANI1_subset/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, e, _ = pickle.load(f)
                coordinates = torch.from_numpy(coordinates).to(torch.float)
                species = torch.from_numpy(species)
                e = torch.from_numpy(e).to(torch.float)
                species_coordinates.append(
                    torchani.utils.broadcast_first_dim({'species': species, 'coordinates': coordinates}))
                energies.append(e)
        species_coordinates = torchani.utils.pad_atomic_properties(
            species_coordinates)
        energies = torch.cat(energies)
        energies_ = self.model((species_coordinates['species'], species_coordinates['coordinates'])).energies
        max_diff = (energies - energies_).abs().max().item()
        self.assertLess(max_diff, self.tolerance)


class TestEnergiesEnergyShifterJIT(TestEnergies):

    def setUp(self):
        super().setUp()
        self.energy_shifter = torch.jit.script(self.energy_shifter)
        self.nn = torchani.nn.Sequential(self.nnp, self.energy_shifter)
        self.model = torchani.nn.Sequential(self.aev_computer, self.nnp, self.energy_shifter)


class TestEnergiesANIModelJIT(TestEnergies):

    def setUp(self):
        super().setUp()
        self.nnp = torch.jit.script(self.nnp)
        self.nn = torchani.nn.Sequential(self.nnp, self.energy_shifter)
        self.model = torchani.nn.Sequential(self.aev_computer, self.nnp, self.energy_shifter)


class TestEnergiesJIT(TestEnergies):

    def setUp(self):
        super().setUp()
        self.model = torch.jit.script(self.model)


if __name__ == '__main__':
    unittest.main()
