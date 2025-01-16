import unittest

import torch

from torchani.nn import AtomicOneHot, AtomicEmbedding
from torchani._testing import ANITestCase, make_elem_idxs, expand
from torchani.utils import SYMBOLS_1X


@expand(device="cpu")
class TestEmbeddings(ANITestCase):
    def setUp(self) -> None:
        self.symbols = SYMBOLS_1X

    def testContinuousEmbedding(self) -> None:
        seed = 1234
        idxs = make_elem_idxs(2, 3, self.symbols, seed=seed)
        idxs[0, 1] = -1
        torch.manual_seed(seed)
        embed = self._setup(AtomicEmbedding(self.symbols, 3))
        out = embed(idxs.view(-1))
        torch.set_printoptions(precision=15)
        expect = torch.tensor(
            [[2.178465843200684, 0.102137297391891, -0.259009689092636],
             [0.000000000000000, 0.000000000000000, 0.000000000000000],
             [0.230966806411743, 0.693077325820923, -0.266860574483871],
             [0.216738238930702, -0.612273097038269, 0.503610730171204],
             [0.046130463480949, 0.402402818202972, -1.011529088020325],
             [0.046130463480949, 0.402402818202972, -1.011529088020325]],
        )
        self.assertEqual(expect, out)

    def testOneHotEmbedding(self) -> None:
        seed = 1234
        idxs = make_elem_idxs(2, 3, self.symbols, seed=seed)
        idxs[0, 1] = -1
        torch.manual_seed(seed)
        embed = self._setup(AtomicOneHot(self.symbols))
        out = embed(idxs.view(-1))
        expect = torch.tensor(
            [[0., 0., 0., 1.],
             [0., 0., 0., 0.],
             [0., 0., 1., 0.],
             [0., 1., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.]],
        )
        self.assertEqual(expect, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
