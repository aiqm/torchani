import typing as tp
import os

import torch
from torch import Tensor

from torchani._testing import TestCase
from torchani.aev import AEVComputerForThermoIntegration, AEVComputer
from torchani.nn import SpeciesConverter
from torchani.utils import SYMBOLS_2X
from test_aev import TestAEV, TestIsolated, TestAEVOnBenzenePBC


path = os.path.dirname(os.path.realpath(__file__))
N = 97


class ThermoIntegrationWrapper(torch.nn.Module):
    def __init__(self, aev_computer) -> None:
        super().__init__()
        self._aev_computer = aev_computer
        self.radial_len = aev_computer.radial_len
        self.angular_len = aev_computer.angular_len

    def forward(
        self,
        elem_idxs: Tensor,
        coords: tp.Optional[Tensor] = None,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> Tensor:
        assert coords is not None
        # use all dummy idxs for appearing and disappearing idxs
        appearing_idxs = -elem_idxs.new_ones((elem_idxs.size(0), 3))
        disappearing_idxs = appearing_idxs.clone()
        ti_factor = coords.new_zeros(1).view(-1)
        return self._aev_computer.forward_for_ti(
            elem_idxs, coords, ti_factor, appearing_idxs, disappearing_idxs
        )


class TestAEVSanityChecks(TestCase):
    def setUp(self):
        super().setUp()
        self.aev_computer = AEVComputerForThermoIntegration.like_2x()
        # O=C(H)H -> O=C(Cl)H
        self.species = torch.tensor([[6, 8, 1, 1, 17]])
        # TODO: It is possible that superimposing both atoms in the AEV
        # of a different atom makes the energy explode, even if the atoms do not
        # see each other
        # TODO: It is unclear how the thermo integration scheme works in the sense
        # of the *absolute energies*
        #
        # TODO: It is unclear whether it is possible to directly use the output energies
        # and make atoms appear and disappear by taking (1 - lambda) and (lambda)
        # in those energies, maybe this is dumber than we think
        # Why is the parametrization *inside the variables themselves*???
        # TODO: I believe this is dumb, actually we don't need all of this
        # "lambda parametrized aev" situation
        self.coords = torch.tensor(
            [
                [
                    [-0.0398460080, -0.0054042378, -0.0006821430],
                    [1.1699090417, -0.1651235922, 0.0390627287],
                    [-0.7099690966, -0.8354632204, 0.0312009563],
                    [-0.4200939371, 1.0059910505, -0.0695815420],
                    [-0.4200939371, 1.0059910505, -0.0695815420],
                ]
            ]
        )

    def testIdxs(self):
        disappearing_idxs = torch.tensor([[3]])
        appearing_idxs = torch.tensor([[4]])
        ti_factor = torch.tensor(0.0)
        converter = SpeciesConverter(SYMBOLS_2X)
        species = converter(self.species)
        aevs_0 = self.aev_computer.forward_for_ti(
            species, self.coords, ti_factor, appearing_idxs, disappearing_idxs
        )

        radial_len = self.aev_computer.radial_len
        angular_len = self.aev_computer.angular_len
        angular_slc = slice(radial_len, radial_len + angular_len)

        aevs_std_0 = self.aev_computer(species[:, :-1], self.coords[:, :-1])
        for i, aev in enumerate(aevs_std_0[0, :]):
            self.assertEqual(aevs_0[0, i, :radial_len], aev[:radial_len])
            self.assertEqual(aevs_0[0, i, angular_slc], aev[angular_slc])
        ti_factor = torch.tensor(1.0)
        aevs_1 = self.aev_computer.forward_for_ti(
            species, self.coords, ti_factor, appearing_idxs, disappearing_idxs
        )
        aevs_std_1 = self.aev_computer(
            species[:, [0, 1, 2, 4]], self.coords[:, [0, 1, 2, 4]]
        )
        for i, aev in enumerate(aevs_std_1[0, :]):
            idx = [0, 1, 2, 4][i]
            self.assertEqual(aevs_1[0, idx, :radial_len], aev[:radial_len])
            self.assertEqual(aevs_1[0, idx, angular_slc], aev[angular_slc])


class TestAEVForTI(TestAEV):
    def setUp(self):
        super().setUp()
        self.aev_computer = tp.cast(
            AEVComputer,
            ThermoIntegrationWrapper(AEVComputerForThermoIntegration.like_1x()),
        )
        self.radial_len = self.aev_computer.radial_len


class TestAEVJITForTI(TestAEVForTI):
    def setUp(self):
        super().setUp()
        self.aev_computer = tp.cast(AEVComputer, torch.jit.script(self.aev_computer))


class TestIsolatedForTI(TestIsolated):
    def setUp(self):
        super().setUp()
        self.aev_computer = AEVComputerForThermoIntegration.like_1x().to(self.device)
        self.rcr = self.aev_computer.radial.cutoff
        self.rca = self.aev_computer.angular.cutoff


class TestAEVForTIOnBenzenePBC(TestAEVOnBenzenePBC):
    def setUp(self):
        super().setUp()
        self.aev_computer = AEVComputerForThermoIntegration.like_1x()
        self.aev = self.aev_computer(self.species, self.coords, self.cell, self.pbc)
        self.natoms = self.aev.shape[1]
