from ase.lattice.cubic import Diamond
from ase.md.langevin import Langevin
from ase import units, Atoms
from ase.calculators.test import numeric_force
import torch
import torchani
import unittest
import numpy
import itertools
import math
import os
import pickle

path = os.path.dirname(os.path.realpath(__file__))
N = 97
tol = 5e-5


def get_numeric_force(atoms, eps):
    fn = torch.zeros((len(atoms), 3), dtype=torch.double)
    for i in range(len(atoms)):
        for j in range(3):
            fn[i, j] = numeric_force(atoms, i, j, eps)
    return fn


class TestASE(unittest.TestCase):

    def _testForce(self, pbc):
        atoms = Diamond(symbol="C", pbc=pbc)
        builtin = torchani.neurochem.Builtins()
        calculator = torchani.ase.Calculator(
            builtin.species, builtin.aev_computer,
            builtin.models, builtin.energy_shifter)
        atoms.set_calculator(calculator)
        dyn = Langevin(atoms, 5 * units.fs, 30000000 * units.kB, 0.002)
        dyn.run(100)
        f = torch.from_numpy(atoms.get_forces())
        fn = get_numeric_force(atoms, 0.001)
        df = (f - fn).abs().max()
        avgf = f.abs().mean()
        if avgf > 0:
            self.assertLess(df / avgf, 0.1)

    def testForceWithPBCEnabled(self):
        self._testForce(True)

    def testForceWithPBCDisabled(self):
        self._testForce(False)

    def testANIDataset(self):
        builtin = torchani.neurochem.Builtins()
        calculator = torchani.ase.Calculator(
            builtin.species, builtin.aev_computer,
            builtin.models, builtin.energy_shifter)
        default_neighborlist_calculator = torchani.ase.Calculator(
            builtin.species, builtin.aev_computer,
            builtin.models, builtin.energy_shifter, _default_neighborlist=True)
        nnp = torch.nn.Sequential(
            builtin.aev_computer,
            builtin.models,
            builtin.energy_shifter
        )
        for i in range(N):
            datafile = os.path.join(path, 'test_data/ANI1_subset/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, _, _ = pickle.load(f)
                coordinates = coordinates[0]
                species = species[0]
                species_str = [builtin.consts.species[i] for i in species]

                atoms = Atoms(species_str, positions=coordinates)
                atoms.set_calculator(calculator)
                energy1 = atoms.get_potential_energy() / units.Hartree
                forces1 = atoms.get_forces() / units.Hartree

                atoms2 = Atoms(species_str, positions=coordinates)
                atoms2.set_calculator(default_neighborlist_calculator)
                energy2 = atoms2.get_potential_energy() / units.Hartree
                forces2 = atoms2.get_forces() / units.Hartree

                coordinates = torch.tensor(coordinates,
                                           requires_grad=True).unsqueeze(0)
                _, energy3 = nnp((torch.from_numpy(species).unsqueeze(0),
                                  coordinates))
                forces3 = -torch.autograd.grad(energy3.squeeze(),
                                               coordinates)[0].numpy()
                energy3 = energy3.item()

                self.assertLess(abs(energy1 - energy2), tol)
                self.assertLess(abs(energy1 - energy3), tol)

                diff_f12 = torch.tensor(forces1 - forces2).abs().max().item()
                self.assertLess(diff_f12, tol)
                diff_f13 = torch.tensor(forces1 - forces3).abs().max().item()
                self.assertLess(diff_f13, tol)

    def testForceAgainstDefaultNeighborList(self):
        atoms = Diamond(symbol="C", pbc=False)
        builtin = torchani.neurochem.Builtins()

        calculator = torchani.ase.Calculator(
            builtin.species, builtin.aev_computer,
            builtin.models, builtin.energy_shifter)
        default_neighborlist_calculator = torchani.ase.Calculator(
            builtin.species, builtin.aev_computer,
            builtin.models, builtin.energy_shifter, _default_neighborlist=True)

        atoms.set_calculator(calculator)
        dyn = Langevin(atoms, 5 * units.fs, 50 * units.kB, 0.002)

        def test_energy(a=atoms):
            a = a.copy()
            a.set_calculator(calculator)
            e1 = a.get_potential_energy()
            a.set_calculator(default_neighborlist_calculator)
            e2 = a.get_potential_energy()
            self.assertLess(abs(e1 - e2), tol)

        dyn.attach(test_energy, interval=1)
        dyn.run(500)

    def testTranslationalInvariancePBC(self):
        atoms = Atoms('CH4', [[0, 0, 0],
                              [1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1],
                              [0, 1, 1]],
                      cell=[2, 2, 2], pbc=True)

        builtin = torchani.neurochem.Builtins()
        calculator = torchani.ase.Calculator(
            builtin.species, builtin.aev_computer,
            builtin.models, builtin.energy_shifter)
        atoms.set_calculator(calculator)
        e = atoms.get_potential_energy()

        for _ in range(100):
            positions = atoms.get_positions()
            translation = (numpy.random.rand(3) - 0.5) * 2
            atoms.set_positions(positions + translation)
            self.assertEqual(e, atoms.get_potential_energy())

    def assertTensorEqual(self, a, b):
        self.assertLess((a - b).abs().max().item(), 1e-6)

    def testPBCConnersSeeEachOther(self):
        species = torch.tensor([[0, 0]])
        neighborlist = torchani.ase.NeighborList(cell=[10, 10, 10], pbc=True)

        xyz1 = torch.tensor([0.1, 0.1, 0.1])
        xyz2s = [
            torch.tensor([9.9, 0.0, 0.0]),
            torch.tensor([0.0, 9.9, 0.0]),
            torch.tensor([0.0, 0.0, 9.9]),
            torch.tensor([9.9, 9.9, 0.0]),
            torch.tensor([0.0, 9.9, 9.9]),
            torch.tensor([9.9, 0.0, 9.9]),
            torch.tensor([9.9, 9.9, 9.9]),
        ]

        for xyz2 in xyz2s:
            coordinates = torch.stack([xyz1, xyz2]).unsqueeze(0)
            s, _, D = neighborlist(species, coordinates, 1)
            self.assertListEqual(list(s.shape), [1, 2, 1])
            neighbor_coordinate = D[0][0].squeeze() + xyz1
            mirror = xyz2
            for i in range(3):
                if mirror[i] > 5:
                    mirror[i] -= 10
            self.assertTensorEqual(neighbor_coordinate, mirror)

    def testPBCSurfaceSeeEachOther(self):
        species = torch.tensor([[0, 0]])
        neighborlist = torchani.ase.NeighborList(cell=[10, 10, 10], pbc=True)

        for i in range(3):
            xyz1 = torch.tensor([5.0, 5.0, 5.0])
            xyz1[i] = 0.1
            xyz2 = xyz1.clone()
            xyz2[i] = 9.9

            coordinates = torch.stack([xyz1, xyz2]).unsqueeze(0)
            s, _, D = neighborlist(species, coordinates, 1)
            self.assertListEqual(list(s.shape), [1, 2, 1])
            neighbor_coordinate = D[0][0].squeeze() + xyz1
            xyz2[i] = -0.1
            self.assertTensorEqual(neighbor_coordinate, xyz2)

    def testPBCEdgesSeeEachOther(self):
        species = torch.tensor([[0, 0]])
        neighborlist = torchani.ase.NeighborList(cell=[10, 10, 10], pbc=True)

        for i, j in itertools.combinations(range(3), 2):
            xyz1 = torch.tensor([5.0, 5.0, 5.0])
            xyz1[i] = 0.1
            xyz1[j] = 0.1
            for new_i, new_j in [[0.1, 9.9], [9.9, 0.1], [9.9, 9.9]]:
                xyz2 = xyz1.clone()
                xyz2[i] = new_i
                xyz2[j] = new_i

            coordinates = torch.stack([xyz1, xyz2]).unsqueeze(0)
            s, _, D = neighborlist(species, coordinates, 1)
            self.assertListEqual(list(s.shape), [1, 2, 1])
            neighbor_coordinate = D[0][0].squeeze() + xyz1

            if xyz2[i] > 5:
                xyz2[i] = -0.1
            if xyz2[j] > 5:
                xyz2[j] = -0.1

            self.assertTensorEqual(neighbor_coordinate, xyz2)

    def testNonRectangularPBCConnersSeeEachOther(self):
        species = torch.tensor([[0, 0]])
        neighborlist = torchani.ase.NeighborList(
            cell=[10, 10, 10 * math.sqrt(2), 90, 45, 90], pbc=True)

        xyz1 = torch.tensor([0.1, 0.1, 0.05])
        xyz2 = torch.tensor([10.0, 0.1, 0.1])
        mirror = torch.tensor([0.0, 0.1, 0.1])

        coordinates = torch.stack([xyz1, xyz2]).unsqueeze(0)
        s, _, D = neighborlist(species, coordinates, 1)
        self.assertListEqual(list(s.shape), [1, 2, 1])
        neighbor_coordinate = D[0][0].squeeze() + xyz1
        for i in range(3):
            if mirror[i] > 5:
                mirror[i] -= 10
        self.assertTensorEqual(neighbor_coordinate, mirror)


if __name__ == '__main__':
    unittest.main()
