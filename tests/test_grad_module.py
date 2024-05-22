import unittest

import torch

from torchani.testing import ANITest, expand
from torchani.grad import (
    forces_and_hessians,
    forces,
    hessians,
)


@expand(device="cpu", jit=True)
class TestScripting(ANITest):
    def testHessians(self):
        torch.jit.script(hessians)

    def testForcesAndHessians(self) -> None:
        torch.jit.script(forces_and_hessians)

    def testForces(self) -> None:
        torch.jit.script(forces)


if __name__ == "__main__":
    unittest.main(verbosity=2)
