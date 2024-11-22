import torch
from torch import Tensor
import pickle
from pathlib import Path
import typing as tp
import unittest

from torchani.neighbors import Neighbors
from torchani._testing import ANITestCase, expand, make_molecs
from torchani.potentials import PairPotential

this_dir = Path(__file__).parent


@expand()
class TestCustomPotential(ANITestCase):
    def setUp(self) -> None:
        self.sym = ("H", "C", "O")
        molecs = make_molecs(3, 5, seed=1234, device=self.device)
        self.coords = molecs.coords
        self.znums = molecs.atomic_nums

    def testCreateCustom(self):
        _ = self._setup(self._makeCustomPot())

    def testCreateCustomTrainable(self):
        _ = self._setup(self._makeCustomPot(trainable=("eq",)))

    def testCustomEnergies(self):
        pot = self._setup(self._makeCustomPot(trainable=("eq",)))
        energies = pot(self.znums, self.coords)
        with open(
            Path(this_dir, "resources", "potentials", "custom-energies.pkl"), mode="rb"
        ) as f:
            energies_expect = pickle.load(f)
        self.assertEqual(energies_expect, energies)

    def testCustomForces(self):
        pot = self._setup(self._makeCustomPot(trainable=("eq",)))
        self.coords.requires_grad_(True)
        energies = pot(self.znums, self.coords)
        forces = -torch.autograd.grad(energies.sum(), self.coords)[0]
        with open(
            Path(this_dir, "resources", "potentials", "custom-energies-forces.pkl"),
            mode="rb",
        ) as f:
            energies_expect, forces_expect = pickle.load(f)
        self.assertEqual(energies_expect, energies)
        self.assertEqual(forces_expect, forces)

    def _makeCustomPot(self, trainable: tp.Sequence[str] = ()) -> PairPotential:

        class Square(PairPotential):
            tensors = ["bias"]  # Vectors (all with the same len) or scalars
            elem_tensors = ["pair_bias"]
            pair_elem_tensors = ["k", "eq"]  # shape (num-sym * (num-sym + 1) / 2)

            def pair_energies(self, elem_idxs: Tensor, neighbors: Neighbors) -> Tensor:
                elem_pairs = elem_idxs.view(-1)[neighbors.indices]

                eq = self.to_pair_values(self.eq, elem_pairs)
                k = self.to_pair_values(self.k, elem_pairs)
                # Manually combine. TorchScript doesn't support .unbind()
                pair_bias = torch.outer(self.pair_bias, self.pair_bias)[
                    elem_pairs[0], elem_pairs[1]
                ]
                return self.bias + k / 2 * (neighbors.distances - eq) ** 2 + pair_bias

        num = len(self.sym)
        k = (1.0,) * (num * (num + 1) // 2)
        eq = (1.5,) * (num * (num + 1) // 2)
        pair_bias = (1.0, 2.0, 3.0)
        return Square(
            symbols=self.sym,
            pair_bias=pair_bias,
            k=k,
            eq=eq,
            bias=0.1,
            trainable=trainable,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
