from pathlib import Path
import unittest
import pickle

import torch

from torchani._testing import ANITestCase, expand
from torchani.utils import pad_atomic_properties
from torchani.models import ANI1x, ANI2x, ANI2dr, ANImbisv
from torchani.units import hartree2kcalpermol


@expand()
class TestANI2x(ANITestCase):
    def setUp(self):
        self.model_pti = self._setup(ANI2x(model_index=0))
        self.model = self._setup(ANI2x(model_index=0, periodic_table_index=False))

    def testDiatomics(self):
        coordinates = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]], device=self.device
        )
        coordinates = coordinates.repeat(4, 1, 1)
        # F2, S2, O2, Cl2
        species_pti = torch.tensor(
            [[9, 9], [16, 16], [8, 8], [17, 17]], device=self.device, dtype=torch.long
        )
        # in 2x the species are not in periodic table order unfortunately
        species = torch.tensor(
            [[5, 5], [4, 4], [3, 3], [6, 6]], device=self.device, dtype=torch.long
        )
        e_pti = self.model_pti((species_pti, coordinates)).energies
        e = self.model((species, coordinates)).energies
        self.assertEqual(e_pti, e)

        # compare against 2x energies calculated directly from original model
        e = hartree2kcalpermol(e)
        e_expect = torch.tensor(
            [-125100.7729, -499666.2354, -94191.3460, -577504.1792], device=self.device
        )
        self.assertEqual(e_expect, e)


@expand()
class TestANI2dr(ANITestCase):
    def setUp(self):
        self.model = self._setup(ANI2dr(model_index=0))

    def testDiatomics(self):
        coordinates = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]], dtype=torch.float, device=self.device
        )
        coordinates = coordinates.repeat(4, 1, 1)
        # F2, S2, O2, Cl2
        species = torch.tensor(
            [[9, 9], [16, 16], [8, 8], [17, 17]], dtype=torch.long, device=self.device
        )
        e = self.model((species, coordinates)).energies

        e = hartree2kcalpermol(e)
        torch.set_printoptions(precision=9)
        e_expect = torch.tensor(
            [-125121.867187500, -499631.781250000, -94128.179687500, -577468.500000000],
            device=self.device,
            dtype=torch.float,
        )
        self.assertEqual(e_expect, e)


@expand()
class TestANImbisv(ANITestCase):
    def setUp(self):
        self.model = self._setup(ANImbisv())

    def testAgainstOriginalModel(self):
        # CH2FCl (padded) and H2N-C(=O)-SH
        species = torch.tensor(
            [[6, 9, 17, 1, 1, -1, -1], [6, 8, 7, 1, 1, 16, 1]],
            dtype=torch.long,
            device=self.device,
        )
        coords = torch.tensor(
            [
                [
                    [0.000, 0.000, 0.000],
                    [1.380, 0.000, 0.000],
                    [-0.590, 1.650, 0.000],
                    [-0.360, -0.510, 0.890],
                    [-0.360, -0.510, -0.890],
                    [0.000, 0.000, 0.000],
                    [0.000, 0.000, 0.000],
                ],
                [
                    [0.000, 0.000, 0.000],
                    [1.210, 0.000, 0.000],
                    [-0.680, 1.170, 0.000],
                    [-1.690, 1.170, 0.000],
                    [-0.170, 2.040, 0.000],
                    [-0.900, -1.530, 0.000],
                    [-0.050, -2.560, 0.000],
                ],
            ],
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        _, energies, results = self.model((species, coords))
        forces = -torch.autograd.grad(energies.sum(), coords)[0]

        # compare against values calculated directly from original model
        energies_expect = torch.tensor(
            [-599.281433105, -568.010070801], dtype=torch.float, device=self.device
        )
        forces_expect = torch.tensor(
            [
                [
                    [0.031957537, -0.012144636, 0.0],
                    [-0.026019368, -0.002398453, 0.0],
                    [-0.006510776, 0.010823604, 0.0],
                    [0.000286304, 0.001859743, 0.007532505],
                    [0.000286302, 0.001859743, -0.007532501],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.083600380, 0.009646112, 0.0],
                    [0.024611920, -0.008209614, 0.0],
                    [-0.056278415, -0.024862716, 0.0],
                    [0.000436414, 0.010749113, 0.0],
                    [0.004568741, -0.006140794, 0.0],
                    [-0.091223456, -0.008092706, 0.0],
                    [0.034284428, 0.026910603, 0.0],
                ],
            ],
            dtype=torch.float,
            device=self.device,
        )
        atomic_charges_expect = torch.tensor(
            [
                [0.144452095, -0.206449732, -0.181856558, 0.121927090, 0.121927112]
                + [0.0, 0.0],
                [0.758769035, -0.573841512, -0.903622448, 0.421272844, 0.422852546]
                + [-0.349366039, 0.223935500],
            ],
            dtype=torch.float,
            device=self.device,
        )
        atomic_volumes_expect = torch.tensor(
            [
                [30.005834579, 17.939876556, 68.033920288, 2.593828201, 2.593828201]
                + [0.0, 0.0],
                [21.535203934, 25.622909546, 34.980548859, 1.346864820, 1.372277856]
                + [82.506187439, 2.327467680],
            ],
            dtype=torch.float,
            device=self.device,
        )
        self.assertEqual(energies_expect, energies)
        self.assertEqual(forces_expect, forces)
        self.assertEqual(atomic_charges_expect, results["atomic_charges"])
        self.assertEqual(atomic_volumes_expect, results["atomic_volumes"])


@expand(device="cpu", jit=False)
class TestCorrectInput(ANITestCase):
    def setUp(self):
        model = ANI1x(model_index=0, periodic_table_index=False)
        self.model = self._setup(model)
        self.converter = self._setup(model.species_converter)
        self.aev_computer = self._setup(model.aev_computer)
        self.ani_networks = self._setup(model.neural_networks)

    def testUnknownSpecies(self):
        # unsupported atomic number raises a value error
        with self.assertRaises(ValueError):
            _ = self.converter(torch.tensor([[1, 1, 7, 10]]))

        # larger index than supported by the model raises a value error
        with self.assertRaises(ValueError):
            _ = self.model((torch.tensor([[0, 1, 2, 4]]), torch.zeros((1, 4, 3))))

    def testIncorrectShape(self):
        # non matching shapes between species and coordinates
        self.assertRaises(
            AssertionError,
            self.model,
            (torch.tensor([[0, 1, 2, 3]]), torch.zeros((1, 3, 3))),
        )
        self.assertRaises(
            AssertionError,
            self.aev_computer,
            torch.tensor([[0, 1, 2, 3]]),
            torch.zeros((1, 3, 3)),
        )
        self.assertRaises(
            AssertionError,
            self.ani_networks,
            torch.tensor([[0, 1, 2, 3]]),
            torch.zeros((1, 3, 384)),
        )
        self.assertRaises(
            AssertionError,
            self.model,
            (torch.tensor([[0, 1, 2, 3]]), torch.zeros((1, 4, 4))),
        )
        self.assertRaises(
            AssertionError,
            self.model,
            (torch.tensor([0, 1, 2, 3]), torch.zeros((4, 3))),
        )


@expand()
class TestEnergies(ANITestCase):
    def setUp(self):
        self.model = self._setup(ANI1x(model_index=0, periodic_table_index=False))
        self.num_conformers = 50
        self.file_path = (Path(__file__).resolve().parent / "resources") / "ANI1_subset"

    def testIsomers(self):
        for i in range(self.num_conformers):
            with open(self.file_path / str(i), "rb") as f:
                coordinates, species, _, _, energies, _ = pickle.load(f)
                coordinates = torch.tensor(
                    coordinates, dtype=torch.float, device=self.device
                )
                species = torch.tensor(species, device=self.device, dtype=torch.long)
                energies = torch.tensor(energies, dtype=torch.float, device=self.device)
                energies_ = self.model((species, coordinates)).energies
                self.assertEqual(energies, energies_)

    def testPadding(self):
        batch = []
        for i in range(self.num_conformers):
            with open(self.file_path / str(i), "rb") as f:
                coordinates, species, _, _, energies, _ = pickle.load(f)
                coordinates = torch.tensor(
                    coordinates, dtype=torch.float, device=self.device
                )
                species = torch.tensor(species, device=self.device, dtype=torch.long)
                energies = torch.tensor(energies, dtype=torch.float, device=self.device)
                batch.append(
                    {
                        "species": species,
                        "coordinates": coordinates,
                        "energies": energies,
                    }
                )
        padded_batch = pad_atomic_properties(batch)
        species = padded_batch["species"]
        coordinates = padded_batch["coordinates"]
        energies_expect = padded_batch["energies"]
        energies = self.model((species, coordinates)).energies
        self.assertEqual(energies, energies_expect)


if __name__ == "__main__":
    unittest.main(verbosity=2)
