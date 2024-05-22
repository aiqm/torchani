import unittest
import torch
import torchani
from torchani.testing import TestCase, ANITest, expand
from torchani.grad import energies_and_forces


@expand()
class TestSpeciesConverter(ANITest):
    def testSpeciesConverter(self):
        input_ = torch.tensor(
            [
                [1, 6, 7, 8, -1],
                [1, 1, -1, 8, 1],
            ],
            device=self.device,
            dtype=torch.long,
        )
        expect = torch.tensor(
            [
                [0, 1, 2, 3, -1],
                [0, 0, -1, 3, 0],
            ],
            device=self.device,
            dtype=torch.long,
        )
        dummy_coordinates = torch.empty(2, 5, 3, device=self.device)
        converter = self._setup(torchani.SpeciesConverter(["H", "C", "N", "O"]))
        output = converter((input_, dummy_coordinates)).species
        self.assertEqual(output, expect)


class TestBuiltinEnsemblePeriodicTableIndex(TestCase):
    def setUp(self):
        self.model1 = torchani.models.ANI1x(periodic_table_index=False)
        self.model2 = torchani.models.ANI1x()
        self.coordinates = torch.tensor(
            [
                [
                    [0.03192167, 0.00638559, 0.01301679],
                    [-0.83140486, 0.39370209, -0.26395324],
                    [-0.66518241, -0.84461308, 0.20759389],
                    [0.45554739, 0.54289633, 0.81170881],
                    [0.66091919, -0.16799635, -0.91037834],
                ]
            ],
        )
        self.species1 = self.model1.species_to_tensor(
            ["C", "H", "H", "H", "H"]
        ).unsqueeze(0)
        self.species2 = torch.tensor([[6, 1, 1, 1, 1]])

    def testCH4Ensemble(self):
        energy1, force1 = energies_and_forces(
            self.model1, self.species1, self.coordinates
        )
        energy2, force2 = energies_and_forces(
            self.model2, self.species2, self.coordinates
        )
        self.assertEqual(energy1, energy2)
        self.assertEqual(force1, force2)

    def testCH4Single(self):
        energy1 = self.model1((self.species1, self.coordinates)).energies
        energy1, force1 = energies_and_forces(
            self.model1[0], self.species1, self.coordinates
        )
        energy2, force2 = energies_and_forces(
            self.model2[0], self.species2, self.coordinates
        )
        self.assertEqual(energy1, energy2)
        self.assertEqual(force1, force2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
