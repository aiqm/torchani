import torch
import torchani
import unittest
import os
import pickle
from torchani.testing import TestCase


path = os.path.dirname(os.path.realpath(__file__))
N = 97


class TestActivation(TestCase):

    def testFittedSoftplus(self):
        celu = torch.nn.CELU(alpha=0.1)
        fsp = torchani.nn.FittedSoftplus()
        x = torch.linspace(-3, 3, 1000)
        # this functions are approximately equal to 1e-2
        self.assertEqual(celu(x), fsp(x), atol=1e-2, rtol=1e-2)


class TestANI2x(TestCase):

    def setUp(self):
        self.model_pti = torchani.models.ANI2x(model_index=0, periodic_table_index=True)
        self.model = torchani.models.ANI2x(model_index=0, periodic_table_index=False)

    def testDiatomics(self):
        coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 2.0]]])
        coordinates = coordinates.repeat(4, 1, 1)
        # F2, S2, O2, Cl2
        species_pti = torch.tensor([[9, 9], [16, 16], [8, 8], [17, 17]])
        # in 2x the species are not in periodic table order unfortunately
        species = torch.tensor([[5, 5], [4, 4], [3, 3], [6, 6]])
        e_pti = self.model_pti((species_pti, coordinates)).energies
        e = self.model((species, coordinates)).energies
        self.assertEqual(e_pti, e)

        # compare against 2x energies calculated directly from neurochem by kdavis
        e = torchani.units.hartree2kcalmol(e)
        e_expect = torch.tensor([-125100.7729, -499666.2354, -94191.3460, -577504.1792])
        self.assertEqual(e_expect.to(torch.float), e.to(torch.float))


class TestCorrectInput(TestCase):

    def setUp(self):
        self.model = torchani.models.ANI1x(model_index=0, periodic_table_index=False)
        self.converter = torchani.nn.SpeciesConverter(['H', 'C', 'N', 'O'])
        self.aev_computer = self.model.aev_computer
        self.ani_model = self.model.neural_networks

    def testUnknownSpecies(self):
        # unsupported atomic number raises a value error
        self.assertRaises(ValueError, self.converter, (torch.tensor([[1, 1, 7, 10]]), torch.zeros((1, 4, 3))))
        # larger index than supported by the model raises a value error
        self.assertRaises(ValueError, self.model, (torch.tensor([[0, 1, 2, 4]]), torch.zeros((1, 4, 3))))

    def testIncorrectShape(self):
        # non matching shapes between species and coordinates
        self.assertRaises(AssertionError, self.model, (torch.tensor([[0, 1, 2, 3]]), torch.zeros((1, 3, 3))))
        self.assertRaises(AssertionError, self.aev_computer, (torch.tensor([[0, 1, 2, 3]]), torch.zeros((1, 3, 3))))
        self.assertRaises(AssertionError, self.ani_model, (torch.tensor([[0, 1, 2, 3]]), torch.zeros((1, 3, 384))))
        self.assertRaises(AssertionError, self.model, (torch.tensor([[0, 1, 2, 3]]), torch.zeros((1, 4, 4))))
        self.assertRaises(AssertionError, self.model, (torch.tensor([0, 1, 2, 3]), torch.zeros((4, 3))))


class TestEnergies(TestCase):
    # tests the predicions for a torchani.nn.Sequential(AEVComputer(),
    # ANIModel(), EnergyShifter()) against precomputed values

    def setUp(self):
        model = torchani.models.ANI1x(model_index=0)
        self.aev_computer = model.aev_computer
        self.nnp = model.neural_networks
        self.energy_shifter = model.energy_shifter
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
                self.assertEqual(energies, energies_, exact_dtype=False)

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
        self.assertEqual(energies, energies_, exact_dtype=False)


class TestEnergiesEnergyShifterJIT(TestEnergies):
    # only JIT compile the energy shifter and repeat all tests

    def setUp(self):
        super().setUp()
        self.energy_shifter = torch.jit.script(self.energy_shifter)
        self.model = torchani.nn.Sequential(self.aev_computer, self.nnp, self.energy_shifter)


class TestEnergiesANIModelJIT(TestEnergies):
    # only JIT compile the ANI nnp ANIModel and repeat all tests

    def setUp(self):
        super().setUp()
        self.nnp = torch.jit.script(self.nnp)
        self.model = torchani.nn.Sequential(self.aev_computer, self.nnp, self.energy_shifter)


class TestEnergiesJIT(TestEnergies):
    # JIT compile the whole model and repeat all tests

    def setUp(self):
        super().setUp()
        self.model = torch.jit.script(self.model)


if __name__ == '__main__':
    unittest.main()
