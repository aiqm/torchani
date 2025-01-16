import unittest

import torch

from torchani.nn import ANINetworks, ANISharedNetworks, SingleNN
from torchani._testing import ANITestCase, make_elem_idxs, expand
from torchani.utils import SYMBOLS_1X


@expand()
class TestNNContainers(ANITestCase):
    def setUp(self) -> None:
        self.symbols = SYMBOLS_1X
        seed = 1234
        self.idxs = make_elem_idxs(2, 4, self.symbols, seed=seed,)
        self.idxs[0, 1] = -1
        self.idxs[1, 2] = -1
        self.in_dim = 300
        self.idxs = self.idxs.to(self.device)
        self.aevs = torch.randn((2, 4, self.in_dim)).to(self.device)
        torch.manual_seed(seed)

    # For now just test that no errors are raised
    def testNetworks(self) -> None:
        net = self._setup(ANINetworks.default(self.symbols, self.in_dim))
        out = net(self.idxs, self.aevs)
        out.sum().backward()

    def testSingleNN(self) -> None:
        net = self._setup(SingleNN.default(self.symbols, self.in_dim))
        out = net(self.idxs, self.aevs)
        out.sum().backward()

    def testSharedNetworks(self) -> None:
        net = self._setup(ANISharedNetworks.default(self.symbols, self.in_dim))
        out = net(self.idxs, self.aevs)
        out.sum().backward()


if __name__ == "__main__":
    unittest.main(verbosity=2)
