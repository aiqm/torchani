import typing as tp
from pathlib import Path
import json
import tempfile
from copy import deepcopy
import unittest

import h5py
import numpy as np
import torch

from torchani.models import ANI1x
from torchani.transforms import (
    AtomicNumbersToIndices,
    SubtractSAE,
    Compose,
)
from torchani.constants import PERIODIC_TABLE, ATOMIC_NUMBER
from torchani._testing import TestCase
from torchani.datasets import (
    concatenate,
    ANIDataset,
    ANIBatchedDataset,
    create_batched_dataset,
)
from torchani.datasets.filters import (
    filter_by_high_force,
    filter_by_high_energy_error,
)
from torchani.paths import set_data_dir

# Dynamically created attrs
from torchani.datasets import TestData

# Optional tests for zarr
try:
    import zarr

    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

try:
    import pandas

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

dataset_path = Path(Path(__file__).parent.parent, "dataset", "ani-1x", "sample.h5")
dataset_path = dataset_path.resolve()
batch_size = 256

_numbers_to_symbols = np.vectorize(lambda x: PERIODIC_TABLE[x])


TmpFileOrDir = tp.Union[tempfile._TemporaryFileWrapper, tempfile.TemporaryDirectory]


class TestDatasetUtils(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        set_data_dir(self.tmpdir.name)
        self.test_ds = TestData()
        locs = [
            loc
            for loc in self.test_ds.store_locations
            if Path(loc).resolve().stem == "test_data1"
        ]
        self.test_ds_single = ANIDataset(locs[0])

    def tearDown(self):
        set_data_dir()
        self.tmpdir.cleanup()

    def testConcatenate(self):
        ds = self.test_ds
        with tempfile.NamedTemporaryFile(dir=self.tmpdir.name, suffix=".h5") as f:
            cat_ds = concatenate(ds, f.name, verbose=False, delete_originals=False)
            self.assertEqual(cat_ds.num_conformers, ds.num_conformers)

    def testFilterForce(self):
        ds = self.test_ds_single
        ds.create_full_property(
            "forces", is_atomic=True, extra_dims=(3,), dtype=np.float32
        )
        ds.append_conformers(
            "H4",
            {
                "species": torch.ones((1, 4), dtype=torch.long),
                "coordinates": torch.ones((1, 4, 3), dtype=torch.float),
                "energies": torch.ones((1,), dtype=torch.double),
                "forces": torch.full((1, 4, 3), fill_value=3.0, dtype=torch.float),
            },
        )
        out = filter_by_high_force(
            ds,
            threshold=0.5,
            delete_inplace=True,
            verbose=False,
        )
        assert out is not None
        self.assertEqual(len(out[0]), 1)
        self.assertEqual(len(out[0][0]["coordinates"]), 1)

    def testFilterEnergyError(self):
        ds = self.test_ds_single
        model = ANI1x()[0]
        out = filter_by_high_energy_error(
            ds,
            model,
            threshold=1.0,
            delete_inplace=True,
            verbose=False,
        )
        self.assertEqual(len(out[0]), 3)
        self.assertEqual(sum(len(c["coordinates"]) for c in out[0]), 1909)


class TestTransforms(TestCase):
    def setUp(self):
        self.elements = ("H", "C", "N", "O")
        coordinates = torch.randn((2, 7, 3), dtype=torch.float)
        self.input_ = {
            "species": torch.tensor(
                [[-1, 1, 1, 6, 1, 7, 8], [1, 1, 1, 1, 1, 1, 6]], dtype=torch.long
            ),
            "energies": torch.tensor([0.0, 1.0], dtype=torch.float),
            "coordinates": coordinates,
        }
        self.expect = {k: v.clone() for k, v in self.input_.items()}
        self.tmp_dir_batched = tempfile.TemporaryDirectory()
        self.tmp_dir_batched2 = tempfile.TemporaryDirectory()

    def testAtomicNumbersToIndices(self):
        numbers_to_indices = AtomicNumbersToIndices(self.elements)
        self.expect["species"] = torch.tensor(
            [[-1, 0, 0, 1, 0, 2, 3], [0, 0, 0, 0, 0, 0, 1]], dtype=torch.long
        )
        out = numbers_to_indices(self.input_)
        for k, v in out.items():
            self.assertEqual(v, self.expect[k])

    def testSubtractSAE(self):
        subtract_sae = SubtractSAE(self.elements, [0.0, 1.0, 0.0, 1.0])
        self.expect["energies"] = torch.tensor([-2.0, 0.0], dtype=torch.float)
        out = subtract_sae(self.input_)
        for k, v in out.items():
            self.assertEqual(v, self.expect[k])

    def testCompose(self):
        subtract_sae = SubtractSAE(self.elements, [0.0, 1.0, 0.0, 1.0])
        numbers_to_indices = AtomicNumbersToIndices(self.elements)
        compose = Compose([subtract_sae, numbers_to_indices])
        self.expect["energies"] = torch.tensor([-2.0, 0.0], dtype=torch.float)
        self.expect["species"] = torch.tensor(
            [[-1, 0, 0, 1, 0, 2, 3], [0, 0, 0, 0, 0, 0, 1]], dtype=torch.long
        )
        out = compose(self.input_)
        for k, v in out.items():
            self.assertEqual(v, self.expect[k])

    def testInplaceTransform(self):
        subtract_sae = SubtractSAE(self.elements, [0.0, 1.0, 0.0, 1.0])
        numbers_to_indices = AtomicNumbersToIndices(self.elements)
        compose = Compose([subtract_sae, numbers_to_indices])

        create_batched_dataset(
            dataset_path,
            dest_path=self.tmp_dir_batched.name,
            splits={"training": 0.5, "validation": 0.5},
            batch_size=2560,
            transform=compose,
            _shuffle=False,
        )
        create_batched_dataset(
            dataset_path,
            dest_path=self.tmp_dir_batched2.name,
            splits={"training": 0.5, "validation": 0.5},
            batch_size=2560,
            _shuffle=False,
        )
        train = ANIBatchedDataset(
            self.tmp_dir_batched2.name, transform=compose, split="training"
        )
        train_inplace = ANIBatchedDataset(self.tmp_dir_batched.name, split="training")
        for b, inplace_b in zip(train, train_inplace):
            for k in b.keys():
                self.assertEqual(b[k], inplace_b[k])

    def tearDown(self):
        self.tmp_dir_batched.cleanup()
        self.tmp_dir_batched2.cleanup()


class TestANIDataset(TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.num_conformers = [7, 5, 8]
        numpy_conformers = {
            "HCNN": {
                "species": np.array(["H", "C", "N", "N"], dtype="S"),
                "coordinates": self.rng.standard_normal((self.num_conformers[0], 4, 3)),
                "energies": self.rng.standard_normal((self.num_conformers[0],)),
            },
            "HOO": {
                "species": np.array(["H", "O", "O"], dtype="S"),
                "coordinates": self.rng.standard_normal((self.num_conformers[1], 3, 3)),
                "energies": self.rng.standard_normal((self.num_conformers[1],)),
            },
            "HCHHH": {
                "species": np.array(["H", "C", "H", "H", "H"], dtype="S"),
                "coordinates": self.rng.standard_normal((self.num_conformers[2], 5, 3)),
                "energies": self.rng.standard_normal((self.num_conformers[2],)),
            },
        }

        # extra groups for appending
        self.torch_conformers = {
            "H6": {
                "species": torch.ones((5, 6), dtype=torch.long),
                "coordinates": torch.randn((5, 6, 3)),
                "energies": torch.randn((5,)),
            },
            "C6": {
                "species": torch.full((5, 6), fill_value=6, dtype=torch.long),
                "coordinates": torch.randn((5, 6, 3)),
                "energies": torch.randn((5,)),
            },
            "O6": {
                "species": torch.full((5, 6), fill_value=8, dtype=torch.long),
                "coordinates": torch.randn((5, 6, 3)),
                "energies": torch.randn((5,)),
            },
        }
        self._make_random_test_data(numpy_conformers)

    def _make_random_test_data(self, numpy_conformers):
        # create two HDF5 databases, one with 3 groups and one with one
        # group, and fill them with some random data
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_store_one_group: TmpFileOrDir = tempfile.NamedTemporaryFile(
            suffix=".h5"
        )
        self.tmp_store_three_groups: TmpFileOrDir = tempfile.NamedTemporaryFile(
            suffix=".h5"
        )
        self.new_store_name = self.tmp_dir.name / Path("new.h5")

        with h5py.File(self.tmp_store_one_group, "r+") as f1, h5py.File(
            self.tmp_store_three_groups, "r+"
        ) as f3:
            for j, (k, g) in enumerate(numpy_conformers.items()):
                f3.create_group("".join(k))
                for p, v in g.items():
                    f3[k].create_dataset(p, data=v)
                if j == 0:
                    f1.create_group("".join(k))
                    for p, v in g.items():
                        f1[k].create_dataset(p, data=v)

    def _make_new_dataset(self):
        return ANIDataset(self.new_store_name, grouping="by_formula")

    def tearDown(self):
        self.tmp_dir.cleanup()
        try:
            self.tmp_store_one_group.close()  # type: ignore
            self.tmp_store_three_groups.close()  # type: ignore
        except AttributeError:
            self.tmp_store_one_group.cleanup()
            self.tmp_store_three_groups.cleanup()

    def testSymbols(self):
        ds = self._make_new_dataset()
        new_groups = deepcopy(self.torch_conformers)
        for k in ("H6", "O6", "C6"):
            ds.append_conformers(k, new_groups[k])
        self.assertTrue(ds.symbols, ("H", "C", "O"))
        with self.assertRaisesRegex(ValueError, "Either species or numbers"):
            ds.delete_properties({"species"})
            ds.symbols

    def testGetConformers(self):
        ds = ANIDataset(self.tmp_store_three_groups.name)

        # general getter of all conformers
        self.assertEqual(
            ds.get_conformers("HOO")["coordinates"], ds["HOO"]["coordinates"].numpy()
        )
        # test getting 1, 2, ... with a list
        idxs = [1, 2, 4]
        conformers = ds.get_conformers("HCHHH", idxs)
        self.assertEqual(
            conformers["coordinates"], ds["HCHHH"]["coordinates"][torch.tensor(idxs)]
        )
        self.assertEqual(
            conformers["energies"], ds["HCHHH"]["energies"][torch.tensor(idxs)]
        )

        # same with a tensor
        conformers = ds.get_conformers("HCHHH", torch.tensor(idxs))
        self.assertEqual(
            conformers["coordinates"], ds["HCHHH"]["coordinates"][torch.tensor(idxs)]
        )
        self.assertEqual(
            conformers["energies"], ds["HCHHH"]["energies"][torch.tensor(idxs)]
        )

        # same with a ndarray
        conformers = ds.get_conformers("HCHHH", np.array(idxs))
        self.assertEqual(
            conformers["coordinates"], ds["HCHHH"]["coordinates"][torch.tensor(idxs)]
        )
        self.assertEqual(
            conformers["energies"], ds["HCHHH"]["energies"][torch.tensor(idxs)]
        )

        # indices in decreasing order
        conformers = ds.get_conformers("HCHHH", list(reversed(idxs)))
        self.assertEqual(
            conformers["coordinates"],
            ds["HCHHH"]["coordinates"][torch.tensor(list(reversed(idxs)))],
        )

        # getting some equal conformers
        conformers = ds.get_conformers("HCHHH", torch.tensor(idxs + idxs))
        self.assertEqual(
            conformers["coordinates"],
            ds["HCHHH"]["coordinates"][torch.tensor(idxs + idxs)],
        )

        # getting just the energies
        conformers = ds.get_conformers("HCHHH", idxs, properties="energies")
        self.assertEqual(
            conformers["energies"], ds["HCHHH"]["energies"][torch.tensor(idxs)]
        )
        self.assertTrue(conformers.get("species", None) is None)
        self.assertTrue(conformers.get("coordinates", None) is None)

    def testAppendAndDeleteConformers(self):
        # tests delitem and setitem analogues for the dataset
        ds = self._make_new_dataset()

        # check creation
        new_groups = deepcopy(self.torch_conformers)
        for k in ("H6", "C6", "O6"):
            ds.append_conformers(k, new_groups[k])

        for k, v in ds.items():
            self.assertEqual(v, new_groups[k])
        for k in ("H6", "C6", "O6"):
            ds.delete_conformers(k)
        self.assertTrue(len(ds.items()) == 0)

        # check appending
        new_lengths = dict()
        for k in ds.keys():
            new_lengths[k] = len(new_groups[k]["energies"]) * 2
            ds.append_conformers(k, new_groups[k])
        for k in ds.keys():
            self.assertEqual(len(ds.get_conformers(k)["energies"]), new_lengths[k])
            self.assertEqual(
                len(ds.get_conformers(k)["species"]), len(new_groups["O6"]["species"])
            )
        for k in deepcopy(ds.keys()):
            ds.delete_conformers(k)

        # rebuild dataset
        for k in ("H6", "C6", "O6"):
            ds.append_conformers(k, new_groups[k])

        with self.assertRaisesRegex(ValueError, 'Character "/" not supported'):
            ds.append_conformers("O/6", new_groups["O6"])
        with self.assertRaisesRegex(ValueError, "Expected .* but got .*"):
            new_groups_copy = deepcopy(new_groups["O6"])
            del new_groups_copy["energies"]
            ds.append_conformers("O6", new_groups_copy)
        with self.assertRaisesRegex(ValueError, "All appended conformers"):
            new_groups_copy = deepcopy(new_groups["O6"])
            new_groups_copy["species"] = torch.randint(
                size=(5, 6), low=1, high=5, dtype=torch.long
            )
            ds.append_conformers("O6", new_groups_copy)
        with self.assertRaisesRegex(ValueError, "Species needs to have two"):
            new_groups_copy = deepcopy(new_groups["O6"])
            new_groups_copy["species"] = torch.ones((5, 6, 1), dtype=torch.long)
            ds.append_conformers("O6", new_groups_copy)

    def testChunkedIteration(self):
        ds = self._make_new_dataset()
        # first we build numpy conformers with ints and str as species (both
        # allowed)
        conformers = dict()
        for gn in self.torch_conformers.keys():
            conformers[gn] = {
                k: v.detach().cpu().numpy()
                for k, v in self.torch_conformers[gn].items()
            }

        # Build the dataset using conformers
        for k, v in conformers.items():
            ds.append_conformers(k, v)

        keys: tp.Set[str] = set()
        coords = []
        for k, _, v in ds.chunked_numpy_items(max_size=10):
            coords.append(torch.from_numpy(v["coordinates"]))
            keys.add(k)

        keys_large: tp.Set[str] = set()
        coords_large = []
        for k, _, v in ds.chunked_numpy_items(max_size=100000):
            coords_large.append(torch.from_numpy(v["coordinates"]))
            keys_large.add(k)

        keys_expect: tp.Set[str] = set()
        coords_expect = []
        for k, v in ds.numpy_items():
            coords_expect.append(torch.from_numpy(v["coordinates"]))
            keys_expect.add(k)
        self.assertEqual(keys_expect, keys)
        self.assertEqual(torch.cat(coords_expect), torch.cat(coords))
        self.assertEqual(keys_expect, keys_large)
        self.assertEqual(torch.cat(coords_expect), torch.cat(coords_large))

    def testAppendAndDeleteNumpyConformers(self):
        ds = self._make_new_dataset()
        # first we build numpy conformers with ints and str as species (both
        # allowed)
        conformers_int = dict()
        conformers_str = dict()
        for gn in self.torch_conformers.keys():
            conformers_int[gn] = {
                k: v.detach().cpu().numpy()
                for k, v in self.torch_conformers[gn].items()
            }
            conformers_str[gn] = deepcopy(conformers_int[gn])
            conformers_str[gn]["species"] = _numbers_to_symbols(
                conformers_int[gn]["species"]
            )

        # Build the dataset using conformers
        for k, v in conformers_str.items():
            ds.append_conformers(k, v)
        # Check that getters give the same result as what was input
        for k, v in ds.numpy_items(chem_symbols=True):
            for key in v.keys():
                self.assertEqual(v[key], conformers_str[k][key])
        # Now we delete everything
        for k in conformers_str.keys():
            ds.delete_conformers(k)
        self.assertTrue(len(ds.items()) == 0)

        # now we do the same with conformers_int
        for k, v in conformers_int.items():
            ds.append_conformers(k, v)
        for k, v in ds.numpy_items():
            self.assertEqual(v, conformers_int[k])
        for k in conformers_str.keys():
            ds.delete_conformers(k)
        self.assertTrue(len(ds.items()) == 0)

    def testNewScalar(self):
        ds = self._make_new_dataset()
        new_groups = deepcopy(self.torch_conformers)
        initial_len = len(new_groups["C6"]["coordinates"])
        for k in ("H6", "C6", "O6"):
            ds.append_conformers(k, new_groups[k])
        ds.create_full_property("spin_multiplicities", fill_value=1)
        ds.create_full_property("charges", fill_value=0)
        self.assertEqual(len(ds["H6"].keys()), 5)
        self.assertEqual(
            ds.properties,
            {"species", "energies", "coordinates", "charges", "spin_multiplicities"},
        )
        self.assertEqual(
            ds["H6"]["spin_multiplicities"], torch.ones(initial_len, dtype=torch.long)
        )
        self.assertEqual(
            ds["C6"]["charges"], torch.zeros(initial_len, dtype=torch.long)
        )

    def testRegroupFormulas(self):
        ds = self._make_new_dataset()
        new_groups = deepcopy(self.torch_conformers)
        for j, k in enumerate(("H6", "C6", "O6")):
            ds.append_conformers(f"group{j}", new_groups[k])
        ds.regroup_by_formula()
        for k, v in ds.items():
            self.assertEqual(v, new_groups[k])

    def testRegroupNumAtoms(self):
        ds = self._make_new_dataset()
        new_groups = deepcopy(self.torch_conformers)
        for j, k in enumerate(("H6", "C6", "O6")):
            ds.append_conformers(f"group{j}", new_groups[k])
        ds.regroup_by_num_atoms()
        self.assertEqual(len(ds.items()), 1)
        self.assertEqual(len(ds["006"]["coordinates"]), 15)
        self.assertEqual(len(ds["006"]["species"]), 15)
        self.assertEqual(len(ds["006"]["energies"]), 15)

    def testDeleteProperty(self):
        ds = self._make_new_dataset()
        new_groups = deepcopy(self.torch_conformers)
        for k in ("H6", "C6", "O6"):
            ds.append_conformers(k, new_groups[k])
        ds.delete_properties({"energies"})
        for k, v in ds.items():
            self.assertEqual(set(v.keys()), {"species", "coordinates"})
            self.assertEqual(v["species"], new_groups[k]["species"])
            self.assertEqual(v["coordinates"], new_groups[k]["coordinates"])
        # deletion of everything kills all groups
        ds.delete_properties({"species", "coordinates"})
        self.assertEqual(len(ds.items()), 0)

    def testRenameProperty(self):
        ds = self._make_new_dataset()
        new_groups = deepcopy(self.torch_conformers)
        for k in ("H6", "C6", "O6"):
            ds.append_conformers(k, new_groups[k])
        ds.rename_properties({"energies": "renamed_energies"})
        for k, v in ds.items():
            self.assertEqual(
                set(v.keys()), {"species", "coordinates", "renamed_energies"}
            )
        with self.assertRaises(ValueError):
            ds.rename_properties({"null0": "null1"})
        with self.assertRaises(ValueError):
            ds.rename_properties({"species": "renamed_energies"})

    def testCreation(self):
        ANIDataset(self.new_store_name)

    def testSizesOneGroup(self):
        ds = ANIDataset(self.tmp_store_one_group.name)
        self.assertEqual(ds.num_conformers, self.num_conformers[0])
        self.assertEqual(ds.num_conformer_groups, 1)
        self.assertEqual(len(ds), ds.num_conformer_groups)

    def testSizesThreeGroups(self):
        ds = ANIDataset(self.tmp_store_three_groups.name)
        self.assertEqual(ds.num_conformers, sum(self.num_conformers))
        self.assertEqual(ds.num_conformer_groups, 3)
        self.assertEqual(len(ds), ds.num_conformer_groups)

    def testKeys(self):
        ds = ANIDataset(self.tmp_store_three_groups.name)
        keys = set()
        for k in ds.keys():
            keys.update({k})
        self.assertEqual(keys, {"HOO", "HCNN", "HCHHH"})
        self.assertEqual(len(ds.keys()), 3)

    def testValues(self):
        ds = ANIDataset(self.tmp_store_three_groups.name)
        for d in ds.values():
            self.assertEqual(set(d.keys()), {"species", "coordinates", "energies"})
            self.assertEqual(d["coordinates"].shape[-1], 3)
            self.assertEqual(d["coordinates"].shape[0], d["energies"].shape[0])
        self.assertEqual(len(ds.values()), 3)

    def testItems(self):
        ds = ANIDataset(self.tmp_store_three_groups.name)
        for k, v in ds.items():
            self.assertTrue(isinstance(k, str))
            self.assertTrue(isinstance(v, dict))
            self.assertEqual(set(v.keys()), {"species", "coordinates", "energies"})
        self.assertEqual(len(ds.items()), 3)

    def testNumpyItems(self):
        ds = ANIDataset(self.tmp_store_three_groups.name)
        for k, v in ds.numpy_items():
            self.assertTrue(isinstance(k, str))
            self.assertTrue(isinstance(v, dict))
            self.assertEqual(set(v.keys()), {"species", "coordinates", "energies"})
        self.assertEqual(len(ds.items()), 3)

    def testDummyPropertiesAlreadyPresent(self):
        # creating dummy properties in a dataset that already has them does nothing
        ds = ANIDataset(self.tmp_store_three_groups.name)
        for k, v in ds.numpy_items(limit=1):
            expect_coords = v["coordinates"]
        ds = ANIDataset(
            self.tmp_store_three_groups.name,
            dummy_properties={"coordinates": {"fill_value": 0}},
        )
        for k, v in ds.numpy_items(limit=1):
            self.assertEqual(v["coordinates"], expect_coords)

    def testDummyPropertiesRegroup(self):
        # creating dummy properties in a dataset that already has them does nothing
        ANIDataset(self.tmp_store_three_groups.name).regroup_by_num_atoms()
        ds = ANIDataset(
            self.tmp_store_three_groups.name,
            dummy_properties={"charges": dict(), "dipoles": dict()},
        )
        self.assertEqual(
            ds.properties, {"charges", "dipoles", "species", "coordinates", "energies"}
        )
        ds = ANIDataset(self.tmp_store_three_groups.name)
        self.assertEqual(ds.properties, {"species", "coordinates", "energies"})

    def testDummyPropertiesAppend(self):
        # creating dummy properties in a dataset that already has them does nothing
        ANIDataset(self.tmp_store_three_groups.name).regroup_by_num_atoms()
        ds = ANIDataset(
            self.tmp_store_three_groups.name, dummy_properties={"charges": dict()}
        )
        ds.append_conformers(
            "003",
            {
                "species": np.asarray([[1, 1, 1]], dtype=np.int64),
                "energies": np.asarray([10.0], dtype=np.float64),
                "coordinates": np.random.standard_normal((1, 3, 3)).astype(np.float32),
                "charges": np.asarray([1], dtype=np.int64),
            },
        )
        ds = ANIDataset(self.tmp_store_three_groups.name)
        self.assertEqual(
            ds.properties, {"species", "coordinates", "energies", "charges"}
        )

    def testDummyPropertiesNotPresent(self):
        charges_params = {
            "fill_value": 0,
            "dtype": np.int64,
            "extra_dims": tuple(),
            "is_atomic": False,
        }
        dipoles_params = {
            "fill_value": 1,
            "dtype": np.float64,
            "extra_dims": (3,),
            "is_atomic": True,
        }
        ANIDataset(self.tmp_store_three_groups.name).regroup_by_num_atoms()
        ds = ANIDataset(
            self.tmp_store_three_groups.name,
            dummy_properties={
                "charges": charges_params,
                "atomic_dipoles": dipoles_params,
            },
        )
        self.assertEqual(
            ds.properties,
            {"species", "coordinates", "energies", "charges", "atomic_dipoles"},
        )
        for k, v in ds.numpy_items():
            self.assertTrue((v["charges"] == 0).all())
            self.assertTrue(v["charges"].dtype == np.int64)
            self.assertEqual(v["charges"].shape, (v["species"].shape[0],))
            self.assertTrue((v["atomic_dipoles"] == 1.0).all())
            self.assertTrue(v["atomic_dipoles"].dtype == np.float64)
            self.assertEqual(v["atomic_dipoles"].shape, v["species"].shape + (3,))
            self.assertEqual(
                set(v.keys()),
                {"species", "coordinates", "energies", "charges", "atomic_dipoles"},
            )

        # renaming works as expected
        ds.rename_properties(
            {"charges": "other_charges", "atomic_dipoles": "other_atomic_dipoles"}
        )
        self.assertEqual(
            ds.properties,
            {
                "species",
                "coordinates",
                "energies",
                "other_charges",
                "other_atomic_dipoles",
            },
        )
        for k, v in ds.numpy_items():
            self.assertTrue((v["other_charges"] == 0).all())
            self.assertTrue((v["other_atomic_dipoles"] == 1.0).all())
            self.assertEqual(
                set(v.keys()),
                {
                    "species",
                    "coordinates",
                    "energies",
                    "other_charges",
                    "other_atomic_dipoles",
                },
            )

        # deleting works as expected
        ds.delete_properties(("other_charges", "other_atomic_dipoles"))
        self.assertEqual(ds.properties, {"species", "coordinates", "energies"})
        for k, v in ds.numpy_items():
            self.assertEqual(set(v.keys()), {"species", "coordinates", "energies"})

    def testIterConformers(self):
        ds = ANIDataset(self.tmp_store_three_groups.name)
        confs = []
        for c in ds.iter_conformers():
            self.assertTrue(isinstance(c, dict))
            confs.append(c)
        self.assertEqual(len(confs), ds.num_conformers)


@unittest.skipIf(not ZARR_AVAILABLE, "Zarr not installed")
class TestANIDatasetZarr(TestANIDataset):
    def _make_random_test_data(self, numpy_conformers):
        for j, (k, v) in enumerate(deepcopy(numpy_conformers).items()):
            # Zarr does not support legacy format, so we tile the species and add
            # a "grouping" attribute
            numpy_conformers[k]["species"] = np.tile(
                numpy_conformers[k]["species"].reshape(1, -1),
                (self.num_conformers[j], 1),
            )
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_store_one_group = tempfile.TemporaryDirectory(
            suffix=".zarr", dir=self.tmp_dir.name
        )
        self.tmp_store_three_groups = tempfile.TemporaryDirectory(
            suffix=".zarr", dir=self.tmp_dir.name
        )
        self.new_store_name = self.tmp_dir.name / Path("new.zarr")

        store1 = zarr.DirectoryStore(self.tmp_store_one_group.name)
        store3 = zarr.DirectoryStore(self.tmp_store_three_groups.name)
        with zarr.hierarchy.open_group(
            store1, mode="w"
        ) as f1, zarr.hierarchy.open_group(store3, mode="w") as f3:
            f3.attrs["grouping"] = "by_formula"
            f1.attrs["grouping"] = "by_formula"
            for j, (k, g) in enumerate(numpy_conformers.items()):
                f3.create_group("".join(k))
                for p, v in g.items():
                    f3[k].create_dataset(p, data=v)
                if j == 0:
                    f1.create_group("".join(k))
                    for p, v in g.items():
                        f1[k].create_dataset(p, data=v)

    def tearDown(self):
        self.tmp_store_one_group.cleanup()
        self.tmp_store_three_groups.cleanup()
        self.tmp_dir.cleanup()

    def testConvert(self):
        self._testConvert("zarr")

    def _testConvert(self, backend):
        ds = ANIDataset(self.tmp_store_three_groups.name)
        ds.to_backend("hdf5", inplace=True)
        for d in ds.values():
            self.assertEqual(set(d.keys()), {"species", "coordinates", "energies"})
            self.assertEqual(d["coordinates"].shape[-1], 3)
            self.assertEqual(d["coordinates"].shape[0], d["energies"].shape[0])
        self.assertEqual(len(ds.values()), 3)
        ds.to_backend(backend, inplace=True)
        for d in ds.values():
            self.assertEqual(set(d.keys()), {"species", "coordinates", "energies"})
            self.assertEqual(d["coordinates"].shape[-1], 3)
            self.assertEqual(d["coordinates"].shape[0], d["energies"].shape[0])
        self.assertEqual(len(ds.values()), 3)


@unittest.skipIf(not PANDAS_AVAILABLE, "pandas not installed")
class TestANIDatasetPandas(TestANIDatasetZarr):
    def _make_random_test_data(self, numpy_conformers):
        for j, (k, v) in enumerate(deepcopy(numpy_conformers).items()):
            # Parquet does not support legacy format, so we tile the species and add
            # a "grouping" attribute
            numpy_conformers[k]["species"] = np.tile(
                numpy_conformers[k]["species"].reshape(1, -1),
                (self.num_conformers[j], 1),
            )
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_store_one_group = tempfile.TemporaryDirectory(
            suffix=".pqdir", dir=self.tmp_dir.name
        )
        self.tmp_store_three_groups = tempfile.TemporaryDirectory(
            suffix=".pqdir", dir=self.tmp_dir.name
        )
        self.new_store_name = self.tmp_dir.name / Path("new.pqdir")

        f1 = pandas.DataFrame()
        f3 = pandas.DataFrame()
        attrs = {
            "grouping": "by_formula",
            "dims": {"coordinates": (3,)},
            "units": {},
            "dtypes": {
                "coordinates": np.dtype(np.float32).name,
                "species": np.dtype(np.int64).name,
                "energies": np.dtype(np.float64).name,
            },
        }
        with open(
            Path(self.tmp_store_one_group.name) / "meta.json",
            "x",
        ) as f:
            json.dump(attrs, f)

        with open(
            Path(self.tmp_store_three_groups.name) / "meta.json",
            "x",
        ) as f:
            json.dump(attrs, f)

        frames = []
        for j, (k, g) in enumerate(numpy_conformers.items()):
            num_conformations = g["species"].shape[0]
            tmp_df = pandas.DataFrame()
            tmp_df["group"] = pandas.Series([k] * num_conformations)
            tmp_df["species"] = pandas.Series(
                np.vectorize(lambda x: ATOMIC_NUMBER[x])(
                    g["species"].astype(str)
                ).tolist()
            )
            tmp_df["energies"] = pandas.Series(g["energies"])
            tmp_df["coordinates"] = pandas.Series(
                g["coordinates"].reshape(num_conformations, -1).tolist()
            )
            frames.append(tmp_df)
        f3 = pandas.concat(frames)
        f1 = frames[0]
        f1.to_parquet(Path(self.tmp_store_one_group.name) / "data.pq")
        f3.to_parquet(Path(self.tmp_store_three_groups.name) / "data.pq")

    def testConvert(self):
        self._testConvert("pandas")


if __name__ == "__main__":
    unittest.main(verbosity=2)
