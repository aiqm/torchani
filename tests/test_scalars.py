import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch

from torchani import single_point
from torchani._testing import ANITestCase, expand, make_molecs
from torchani.arch import ANI, ANIq, ANIscalars, Assembler
from torchani.io import read_orca, TorchaniIOError
from torchani.potentials import Coulomb, SeparateScalarsNNPotential
from torchani.utils import SYMBOLS_1X


def _assembler(
    keys: tuple[str, ...] = ("atomic_charges", "atomic_volumes"),
    coulomb: bool = False,
) -> Assembler:
    asm = Assembler(cls=ANIscalars)
    asm.set_symbols(SYMBOLS_1X)
    asm.set_aev_computer(radial="ani1x", angular="ani1x", strategy="pyaev")
    asm.set_atomic_networks(ctor="ani1x")
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    for k in keys:
        asm.add_scalar_networks(k, ctor="ani1x")
    if coulomb:
        asm.add_potential(Coulomb, name="coulomb")
    return asm


@expand()
class TestANIscalars(ANITestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.molecs = make_molecs(3, 6, seed=1234, device=self.device)
        self.species_coords = (self.molecs.atomic_nums, self.molecs.coords)

    def testOutputs(self):
        model = self._setup(_assembler().assemble(2))
        species, energies, scalars = model(self.species_coords)
        self.assertEqual(set(scalars), {"atomic_charges", "atomic_volumes"})
        self.assertEqual(energies.shape, (3,))
        self.assertEqual(scalars["atomic_charges"].shape, species.shape)
        self.assertEqual(scalars["atomic_volumes"].shape, species.shape)
        # Default normalizer enforces neutral molecules
        self.assertEqual(
            scalars["atomic_charges"].sum(-1), torch.zeros(3, device=self.device)
        )

    def testEnergiesIncludeNetworks(self):
        model = _assembler().assemble(2).to(self.device)
        _, energies, _ = self._setup(model)(self.species_coords)
        elem_idxs = model.species_converter(self.molecs.atomic_nums)
        neighbors = model.neighborlist(
            model.cutoff, elem_idxs, self.molecs.coords, None, None
        )
        nnp_result = model.nnp.compute_from_neighbors(
            elem_idxs, self.molecs.coords, neighbors
        )
        expect_energies = nnp_result["energies"] + model.energy_shifter(elem_idxs)
        self.assertEqual(energies, expect_energies)

    def testEnsembleValuesAndAtomic(self):
        model = self._setup(_assembler().assemble(2))
        _, energies, scalars = model(self.species_coords)
        _, ens_energies, ens_scalars = model(self.species_coords, ensemble_values=True)
        _, atomic_energies, atomic_scalars = model(self.species_coords, atomic=True)
        self.assertEqual(ens_energies.shape, (2, 3))
        self.assertEqual(ens_energies.mean(0), energies)
        self.assertEqual(atomic_energies.sum(-1), energies)
        for k, v in scalars.items():
            self.assertEqual(ens_scalars[k], v)
            self.assertEqual(atomic_scalars[k], v)

    def testComputeFromNeighbors(self):
        model = _assembler().assemble(2).to(self.device)
        _, energies, scalars = self._setup(model)(self.species_coords)
        elem_idxs = model.species_converter(self.molecs.atomic_nums)
        neighbors = model.neighborlist(
            model.cutoff, elem_idxs, self.molecs.coords, None, None
        )
        result = self._setup(model).compute_from_neighbors(
            elem_idxs, self.molecs.coords, neighbors
        )
        self.assertEqual(result["energies"], energies)
        for k, v in scalars.items():
            self.assertEqual(result[k], v)

    def testEnergiesIncludeCoulomb(self):
        model = _assembler(coulomb=True).assemble(2).to(self.device)
        self.assertEqual(list(model.potentials.keys())[-1], "coulomb")
        pot = self._setup(model.potentials["coulomb"])
        _, energies, scalars = self._setup(model)(self.species_coords)
        model.set_enabled("coulomb", False)
        _, no_coulomb_energies, _ = self._setup(model)(self.species_coords)
        coulomb_energies = pot(
            self.molecs.atomic_nums,
            self.molecs.coords,
            scalars=scalars["atomic_charges"],
        )
        self.assertEqual(energies, no_coulomb_energies + coulomb_energies)

    def testSinglePoint(self):
        model = _assembler().assemble(2).to(self.device)
        result = single_point(
            model,
            *self.species_coords,
            forces=True,
            atomic_charges=True,
            atomic_charges_grad=True,
            atomic_volumes=True,
            atomic_volumes_grad=True,
        )
        _, energies, scalars = model(self.species_coords)
        coords_shape = self.molecs.coords.shape
        self.assertEqual(result["energies"], energies)
        self.assertEqual(result["atomic_charges"], scalars["atomic_charges"])
        self.assertEqual(result["atomic_volumes"], scalars["atomic_volumes"])
        self.assertEqual(result["forces"].shape, coords_shape)
        self.assertEqual(result["atomic_charges_grad"].shape, coords_shape)
        self.assertEqual(result["atomic_volumes_grad"].shape, coords_shape)

    def testOutActivation(self):
        asm = _assembler(keys=("atomic_charges",))
        asm.add_scalar_networks(
            "atomic_volumes", ctor="ani1x", out_activation="softplus"
        )
        model = asm.assemble(2).to(self.device)
        no_act_model = deepcopy(model)
        no_act_model.nnp.scalar_out_activations = torch.nn.ModuleDict()
        _, energies, scalars = self._setup(model)(self.species_coords)
        _, no_act_energies, no_act_scalars = self._setup(no_act_model)(
            self.species_coords
        )
        softplus = torch.nn.Softplus(beta=10.0)
        self.assertEqual(energies, no_act_energies)
        self.assertEqual(scalars["atomic_charges"], no_act_scalars["atomic_charges"])
        self.assertEqual(
            scalars["atomic_volumes"], softplus(no_act_scalars["atomic_volumes"])
        )
        self.assertTrue((scalars["atomic_volumes"] > 0).all())

    def testShiftAfterOutActivation(self):
        asm = _assembler(keys=())
        asm.add_scalar_networks(
            "atomic_volumes", ctor="ani1x", out_activation="softplus"
        )
        asm.set_shift_before_out_activation(False)
        model = asm.assemble(2).to(self.device)
        shifter = model.nnp.scalar_shifters["atomic_volumes"]
        shifter.self_energies.fill_(-1.0)
        no_act_model = deepcopy(model)
        no_act_model.nnp.scalar_out_activations = torch.nn.ModuleDict()
        _, _, scalars = self._setup(model)(self.species_coords)
        _, _, no_act_scalars = self._setup(no_act_model)(self.species_coords)
        softplus = torch.nn.Softplus(beta=10.0)
        self.assertEqual(
            scalars["atomic_volumes"],
            softplus(no_act_scalars["atomic_volumes"] + 1.0) - 1.0,
        )

    def testSinglePointUnsupported(self):
        model = _assembler(keys=("atomic_charges",)).assemble(1).to(self.device)
        with self.assertRaisesRegex(ValueError, "atomic volumes"):
            single_point(model, *self.species_coords, atomic_volumes=True)


class TestAssemblerScalars(unittest.TestCase):
    def testWrongModelClass(self):
        for cls in (ANI, ANIq):
            asm = Assembler(cls=cls)
            with self.assertRaises(ValueError):
                asm.add_scalar_networks("atomic_volumes")
        asm = Assembler(cls=ANIscalars)
        with self.assertRaises(ValueError):
            asm.set_charge_networks()
        asm = Assembler(cls=ANI)
        with self.assertRaises(ValueError):
            asm.set_charge_normalizer(None)

    def testDuplicateKey(self):
        asm = _assembler()
        with self.assertRaises(ValueError):
            asm.add_scalar_networks("atomic_volumes")

    def testOutActivationUnknownKey(self):
        model = _assembler().assemble()
        with self.assertRaises(ValueError):
            SeparateScalarsNNPotential(
                model.nnp.aev_computer,
                model.nnp.neural_networks,
                dict(model.nnp.scalar_networks),
                dict(model.nnp.scalar_shifters),
                scalar_out_activations={"foo": torch.nn.Softplus()},
            )

    def testCoulombRequiresCharges(self):
        asm = _assembler(keys=("atomic_volumes",), coulomb=True)
        with self.assertRaises(ValueError):
            asm.assemble()


class TestReadOrca(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "molecule.orca.in"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _read(self, text: str):
        self.path.write_text(text)
        return read_orca(self.path)

    def testRead(self):
        species, coords = self._read(
            "! B3LYP def2-SVP\n"
            "# A comment\n"
            "\n"
            "* xyz 0 1\n"
            "  O 0.0 0.0 0.1  # Oxygen\n"
            "\n"
            "  h 0.0 0.7 -0.5\n"
            "  1 0.0 -0.7 -0.5\n"
            "*\n"
        )
        self.assertEqual(species.tolist(), [[8, 1, 1]])
        torch.testing.assert_close(
            coords,
            torch.tensor([[[0.0, 0.0, 0.1], [0.0, 0.7, -0.5], [0.0, -0.7, -0.5]]]),
        )

    def testCompactMarker(self):
        species, _ = self._read("*xyz 0 1\nH 0 0 0\nH 0 0 0.7\n*\n")
        self.assertEqual(species.tolist(), [[1, 1]])

    def testMissingOrUnclosedBlock(self):
        for text in (
            "! B3LYP\n",
            "* xyzfile 0 1 molecule.xyz\n",
            "* xyz 0 1\nH 0 0 0\n",
        ):
            with self.assertRaises(TorchaniIOError):
                self._read(text)

    def testBadAtomLine(self):
        with self.assertRaises(TorchaniIOError):
            self._read("* xyz 0 1\nH 0 0\n*\n")


if __name__ == "__main__":
    unittest.main()
