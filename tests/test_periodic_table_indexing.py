import unittest
import torch

from torchani._testing import ANITestCase, expand
from torchani.nn import SpeciesConverter
from torchani.utils import ChemicalSymbolsToInts
from torchani.models import ANI1x
from torchani.grad import energies_and_forces


@expand()
class TestSpeciesConverter(ANITestCase):
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
        converter = self._setup(SpeciesConverter(["H", "C", "N", "O"]))
        output = converter(input_)
        self.assertEqual(output, expect)


@expand(device="cpu", jit=False)
class TestBuiltinEnsemblePeriodicTableIndex(ANITestCase):
    def setUp(self):
        model1 = ANI1x(periodic_table_index=False)
        model2 = ANI1x()
        self.model1 = self._setup(model1)
        self.model2 = self._setup(model2)
        self.single_model1 = self._setup(model1[0])
        self.single_model2 = self._setup(model2[0])
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
            device=self.device,
            dtype=torch.float,
        )
        symbols_to_idxs = self._setup(ChemicalSymbolsToInts(["H", "C", "N", "O"]))
        self.species1 = symbols_to_idxs(["C", "H", "H", "H", "H"]).unsqueeze(0)
        self.species2 = torch.tensor(
            [[6, 1, 1, 1, 1]], device=self.device, dtype=torch.long
        )

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
        energy1, force1 = energies_and_forces(
            self.single_model1, self.species1, self.coordinates
        )
        energy2, force2 = energies_and_forces(
            self.single_model2, self.species2, self.coordinates
        )
        self.assertEqual(energy1, energy2)
        self.assertEqual(force1, force2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
