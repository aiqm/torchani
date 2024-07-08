import unittest

import torch
import torchani

from torch import Tensor
from typing import Optional, Tuple

from torchani.testing import TestCase


class TestAEVHook(TestCase):
    def test_aev_hook(self):
        # Create a test module.

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()

                # Create a ANI2x model.
                self._model = torchani.models.ANI2x()

                # Define a dummy hook. This doesn't do anything with the output
                # but triggers the exception when kwargs are used when calling
                # AEVComputer.forward.
                def hook(
                    module,
                    input: Tuple[
                        Tuple[Tensor, Tensor], Optional[Tensor], Optional[Tensor]
                    ],
                    output: torchani.aev.SpeciesAEV,
                ):
                    pass

                # Register the hook.
                self._aev_hook = self._model.aev_computer.register_forward_hook(hook)

            def forward(self, species, coordinates):
                return self._model((species, coordinates))

        # Create a test module.
        model = TestModule()

        # Convert the model to TorchScript.
        model = torch.jit.script(model)


if __name__ == "__main__":
    unittest.main()
