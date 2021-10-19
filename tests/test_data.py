import os
from pathlib import Path
import h5py
import numpy as np
import json
import torch
import torchani
import unittest
import tempfile
import warnings
from copy import deepcopy
from torchani.transforms import AtomicNumbersToIndices, SubtractSAE, Compose, calculate_saes
from torchani.utils import PERIODIC_TABLE, ATOMIC_NUMBERS
from torchani.testing import TestCase
from torchani.datasets import ANIDataset, ANIBatchedDataset, create_batched_dataset
from torchani.datasets._builtin_datasets import _BUILTIN_DATASETS

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

path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(path, '../dataset/ani-1x/sample.h5')
dataset_path_gdb = os.path.join(path, '../dataset/ani1-up_to_gdb4/ani_gdb_s02.h5')
batch_size = 256
ani1x_sae_dict = {'H': -0.60095298, 'C': -38.08316124, 'N': -54.7077577, 'O': -75.19446356}

_numbers_to_symbols = np.vectorize(lambda x: PERIODIC_TABLE[x])


def ignore_unshuffled_warning():
    warnings.filterwarnings(action='ignore',
                            message="Dataset will not be shuffled, this should only be used for debugging")


class TestDatasetUtils(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.test_ds = torchani.datasets.TestData(self.tmpdir.name, download=True)
        self.test_ds_single = ANIDataset(self.tmpdir.name / Path('test_data1.h5'))

    def tearDown(self):
        self.tmpdir.cleanup()

    def testConcatenate(self):
        ds = self.test_ds
        with tempfile.NamedTemporaryFile(dir=self.tmpdir.name, suffix='.h5') as f:
            cat_ds = torchani.datasets.utils.concatenate(ds, f.name, verbose=False, delete_originals=False)
            self.assertEqual(cat_ds.num_conformers, ds.num_conformers)

    def testFilterForce(self):
        ds = self.test_ds_single
        ds.create_full_property('forces', is_atomic=True, extra_dims=(3,), dtype=np.float32)
        ds.append_conformers('H4', {'species': torch.ones((1, 4), dtype=torch.long),
                              'coordinates': torch.ones((1, 4, 3), dtype=torch.float),
                              'energies': torch.ones((1,), dtype=torch.double),
                              'forces': torch.full((1, 4, 3), fill_value=3.0, dtype=torch.float)})
        out = torchani.datasets.utils.filter_by_high_force(ds, threshold=0.5, delete_inplace=True)
        self.assertEqual(len(out[0]), 1)
        self.assertEqual(len(out[0][0]['coordinates']), 1)

    def testFilterEnergyError(self):
        ds = self.test_ds_single
        model = torchani.models.ANI1x(periodic_table_index=True)[0]
        out = torchani.datasets.utils.filter_by_high_energy_error(ds, model, threshold=1.0, delete_inplace=True)
        self.assertEqual(len(out[0]), 3)
        self.assertEqual(sum(len(c['coordinates']) for c in out[0]), 412)


class TestBuiltinDatasets(TestCase):

    def testSmallSample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = torchani.datasets.TestData(tmpdir, download=True)
            self.assertEqual(ds.grouping, 'by_num_atoms')

    def testDownloadSmallSample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            torchani.datasets.download_builtin_dataset('TestData', 'wb97x-631gd', tmpdir)
            num_h5_files = len(list(Path(tmpdir).glob("*.h5")))
            self.assertGreater(num_h5_files, 0)

    def testBuiltins(self):
        # all these have default levels of theory
        classes = _BUILTIN_DATASETS
        for c in classes:
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(RuntimeError, "Dataset not found"):
                    getattr(torchani.datasets, c)(tmpdir, download=False)

        # these also have the B973c/def2mTZVP LoT
        for c in ['ANI1x', 'ANI2x', 'COMP6v1', 'COMP6v2', 'AminoacidDimers']:
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(RuntimeError, "Dataset not found"):
                    getattr(torchani.datasets, c)(tmpdir, download=False, basis_set='def2mTZVP', functional='B973c')

        # these also have the wB97M-D3BJ/def2TZVPP LoT
        for c in ['ANI1x', 'ANI2x', 'COMP6v1', 'COMP6v2']:
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(RuntimeError, "Dataset not found"):
                    getattr(torchani.datasets, c)(tmpdir, download=False, basis_set='def2TZVPP', functional='wB97MD3BJ')
                # Case insensitivity
                with self.assertRaisesRegex(RuntimeError, "Dataset not found"):
                    getattr(torchani.datasets, c)(tmpdir, download=False, basis_set='DEF2tZvPp', functional='Wb97md3Bj')


class TestFineGrainedShuffle(TestCase):

    def setUp(self):
        self.tmp_dir_batched = tempfile.TemporaryDirectory()

    def testShuffleMixesManyH5(self):
        # test that shuffling correctly mixes multiple h5 files
        num_groups = 10
        num_conformers_per_group = 12
        self._create_dummy_controlled_dataset(num_groups, num_conformers_per_group, use_energy_ranges=False)

        self.train = ANIBatchedDataset(self.tmp_dir_batched.name, split='training')
        self.valid = ANIBatchedDataset(self.tmp_dir_batched.name, split='validation')
        for b, b_valid in zip(self.train, self.valid):
            self.assertNotEqual(b['species'], b_valid['species'])
            self.assertNotEqual(b['energies'], b_valid['energies'])
            self._test_for_batch_diversity(b)
            self._test_for_batch_diversity(b_valid)

    def testShuffleMixesManyH5Folds(self):
        # test that shuffling correctly mixes multiple h5 files
        num_groups = 10
        num_conformers_per_group = 12
        folds = 3
        self._create_dummy_controlled_dataset(num_groups, num_conformers_per_group, use_energy_ranges=False, folds=3)

        def check_train_valid(train, valid):
            for b, b_valid in zip(train, valid):
                self.assertNotEqual(b['species'], b_valid['species'])
                self.assertNotEqual(b['energies'], b_valid['energies'])
                self._test_for_batch_diversity(b)
                self._test_for_batch_diversity(b_valid)
        for j in range(folds):
            train = ANIBatchedDataset(self.tmp_dir_batched.name, split=f'training{j}')
            valid = ANIBatchedDataset(self.tmp_dir_batched.name, split=f'validation{j}')
            check_train_valid(train, valid)

    def testDisjointFolds(self):
        # test that shuffling generates disjoint train and validation, with no duplicates
        num_groups = 10
        num_conformers_per_group = 12
        folds = 5
        self._create_dummy_controlled_dataset(num_groups, num_conformers_per_group, use_energy_ranges=True, folds=folds)

        for j in range(folds):
            self._check_disjoint_and_nonduplicates(f'training{j}', f'validation{j}')

        for j in range(folds):
            for k in range(j + 1, folds):
                self._check_disjoint_and_nonduplicates(f'validation{j}', f'validation{k}')

    def testDisjointTrainValid(self):
        # test that shuffling generates disjoint train and validation, with no duplicates
        num_groups = 10
        num_conformers_per_group = 12
        self._create_dummy_controlled_dataset(num_groups, num_conformers_per_group, use_energy_ranges=True)
        self._check_disjoint_and_nonduplicates('training', 'validation')

    # creates a dataset inside self.tmp_dir_batched
    def _create_dummy_controlled_dataset(self, num_groups, num_conformers_per_group, use_energy_ranges=False, folds=None):

        def create_dummy_file(rng, file_, num_groups, num_conformers_per_group, element, factor, properties, range_start=None):
            with h5py.File(file_, 'r+') as f:
                for j in range(num_groups):
                    f.create_group(f'group{j}')
                    g = f[f'group{j}']
                    for k in properties:
                        if k == 'species':
                            g.create_dataset(k, data=np.array([element, element, element], dtype='S'))
                        elif k == 'coordinates':
                            g.create_dataset(k, data=rng.standard_normal((num_conformers_per_group, 3, 3)))
                        elif k == 'energies':
                            if range_start is not None:
                                g.create_dataset(k, data=np.arange(range_start + j * num_conformers_per_group,
                                                                   range_start + (j + 1) * num_conformers_per_group, dtype=float))
                            else:
                                g.create_dataset(k, data=factor * np.ones((num_conformers_per_group,)))

        with tempfile.TemporaryDirectory() as tmpdir:
            # Each file will have 120 conformers, total 360 conformers
            num_groups = 10
            num_conformers_per_group = 12
            properties = ['species', 'coordinates', 'energies']
            if use_energy_ranges:
                ranges = [0, num_groups * num_conformers_per_group, 2 * num_groups * num_conformers_per_group]
            else:
                ranges = [None, None, None]
            rng = np.random.default_rng(12345)
            with tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.h5') as dummy_h50,\
                 tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.h5') as dummy_h51,\
                 tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.h5') as dummy_h52:
                create_dummy_file(rng, dummy_h50, num_groups, num_conformers_per_group, 'H', 0.0, properties, ranges[0])
                create_dummy_file(rng, dummy_h51, num_groups, num_conformers_per_group, 'C', 1.0, properties, ranges[1])
                create_dummy_file(rng, dummy_h52, num_groups, num_conformers_per_group, 'N', 2.0, properties, ranges[2])

                # both validation and test have 3 batches of 60 each
                h5_dirs = sorted(Path(tmpdir).iterdir())
                if folds is None:
                    create_batched_dataset(h5_dirs, dest_path=self.tmp_dir_batched.name, shuffle=True, shuffle_seed=123456789,
                            splits={'training': 0.5, 'validation': 0.5}, batch_size=60)
                else:
                    create_batched_dataset(h5_dirs, dest_path=self.tmp_dir_batched.name, shuffle=True, shuffle_seed=123456789,
                            folds=folds, batch_size=60)

    def _check_disjoint_and_nonduplicates(self, name1, name2):
        train = ANIBatchedDataset(self.tmp_dir_batched.name, split=name1)
        valid = ANIBatchedDataset(self.tmp_dir_batched.name, split=name2)
        all_train_energies = []
        all_valid_energies = []
        for b, b_valid in zip(train, valid):
            all_train_energies.append(b['energies'])
            all_valid_energies.append(b_valid['energies'])
        all_train_energies = torch.cat(all_train_energies)
        all_valid_energies = torch.cat(all_valid_energies)

        all_train_list = [e.item() for e in all_train_energies]
        all_train_set = set(all_train_list)
        all_valid_list = [e.item() for e in all_valid_energies]
        all_valid_set = set(all_valid_list)
        # no duplicates and disjoint
        self.assertTrue(len(all_train_list) == len(all_train_set))
        self.assertTrue(len(all_valid_list) == len(all_valid_set))
        self.assertTrue(all_train_set.isdisjoint(all_valid_set))

    def _test_for_batch_diversity(self, b):
        zeros = (b['energies'] == 0.0).count_nonzero()
        ones = (b['energies'] == 1.0).count_nonzero()
        twos = (b['energies'] == 2.0).count_nonzero()
        self.assertTrue(zeros > 0)
        self.assertTrue(ones > 0)
        self.assertTrue(twos > 0)

    def tearDown(self):
        self.tmp_dir_batched.cleanup()


class TestEstimationSAE(TestCase):

    def setUp(self):
        self.tmp_dir_batched = tempfile.TemporaryDirectory()
        self.batch_size = 2560
        create_batched_dataset(dataset_path_gdb,
                               dest_path=self.tmp_dir_batched.name,
                               shuffle=True,
                               splits={'training': 1.0},
                               batch_size=self.batch_size,
                               shuffle_seed=12345,
                               include_properties=('energies', 'species', 'coordinates'))
        self.direct_cache = create_batched_dataset(dataset_path_gdb,
                                                   shuffle=True,
                                                   splits={'training': 1.0},
                                                   batch_size=self.batch_size,
                                                   shuffle_seed=12345,
                                                   include_properties=('energies', 'species', 'coordinates'),
                                                   direct_cache=True)
        self.train = ANIBatchedDataset(self.tmp_dir_batched.name, split='training')

    def testExactSAE(self):
        self._testExactSAE(direct=False)

    def testStochasticSAE(self):
        self. _testStochasticSAE(direct=False)

    def testExactSAEDirect(self):
        self._testExactSAE(direct=True)

    def testStochasticSAEDirect(self):
        self. _testStochasticSAE(direct=True)

    def _testExactSAE(self, direct: bool = False):
        if direct:
            ds = self.direct_cache['training']
        else:
            ds = self.train
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore',
                                    message="Using all batches to estimate SAE, this may take up a lot of memory.")
            saes, _ = calculate_saes(ds, ('H', 'C', 'N', 'O'), mode='exact')
            torch.set_printoptions(precision=10)
        self.assertEqual(saes,
                         torch.tensor([-0.5983182192, -38.0726242065, -54.6750144958, -75.1433029175], dtype=torch.float),
                         atol=2.5e-3, rtol=2.5e-3)

    def _testStochasticSAE(self, direct: bool = False):
        if direct:
            ds = self.direct_cache['training']
        else:
            ds = self.train
        saes, _ = calculate_saes(ds, ('H', 'C', 'N', 'O'), mode='sgd')
        # in this specific case the sae difference is very large because it is a
        # very small sample, but for the full sample this imlementation is correct
        self.assertEqual(saes, torch.tensor([-20.4466, -0.3910, -8.8793, -11.4184], dtype=torch.float),
                         atol=0.2, rtol=0.2)

    def tearDown(self):
        self.tmp_dir_batched.cleanup()


class TestTransforms(TestCase):

    def setUp(self):
        self.elements = ('H', 'C', 'N', 'O')
        coordinates = torch.randn((2, 7, 3), dtype=torch.float)
        self.input_ = {'species': torch.tensor([[-1, 1, 1, 6, 1, 7, 8], [1, 1, 1, 1, 1, 1, 6]], dtype=torch.long),
                       'energies': torch.tensor([0.0, 1.0], dtype=torch.float),
                       'coordinates': coordinates}
        self.tmp_dir_batched = tempfile.TemporaryDirectory()
        self.tmp_dir_batched2 = tempfile.TemporaryDirectory()

    def testAtomicNumbersToIndices(self):
        numbers_to_indices = AtomicNumbersToIndices(self.elements)
        expect = {k: v.clone() for k, v in self.input_.items()}
        expect['species'] = torch.tensor([[-1, 0, 0, 1, 0, 2, 3], [0, 0, 0, 0, 0, 0, 1]], dtype=torch.long)
        out = numbers_to_indices(self.input_)
        for k, v in out.items():
            self.assertEqual(v, expect[k])

    def testSubtractSAE(self):
        subtract_sae = SubtractSAE(self.elements, [0.0, 1.0, 0.0, 1.0])
        expect = {k: v.clone() for k, v in self.input_.items()}
        self.input_['species'] = torch.tensor([[-1, 0, 0, 1, 0, 2, 3], [0, 0, 0, 0, 0, 0, 1]], dtype=torch.long)
        expect['energies'] = torch.tensor([-2.0, 0.0], dtype=torch.float)
        expect['species'] = torch.tensor([[-1, 0, 0, 1, 0, 2, 3], [0, 0, 0, 0, 0, 0, 1]], dtype=torch.long)
        out = subtract_sae(self.input_)
        for k, v in out.items():
            self.assertEqual(v, expect[k])

    def testCompose(self):
        subtract_sae = SubtractSAE(self.elements, [0.0, 1.0, 0.0, 1.0])
        numbers_to_indices = AtomicNumbersToIndices(self.elements)
        compose = Compose([numbers_to_indices, subtract_sae])
        expect = {k: v.clone() for k, v in self.input_.items()}
        expect['energies'] = torch.tensor([-2.0, 0.0], dtype=torch.float)
        expect['species'] = torch.tensor([[-1, 0, 0, 1, 0, 2, 3], [0, 0, 0, 0, 0, 0, 1]], dtype=torch.long)
        out = compose(self.input_)
        for k, v in out.items():
            self.assertEqual(v, expect[k])

    def testInplaceTransform(self):
        subtract_sae = SubtractSAE(self.elements, [0.0, 1.0, 0.0, 1.0])
        numbers_to_indices = AtomicNumbersToIndices(self.elements)
        compose = Compose([numbers_to_indices, subtract_sae])

        with warnings.catch_warnings():
            ignore_unshuffled_warning()
            create_batched_dataset(dataset_path, dest_path=self.tmp_dir_batched.name, shuffle=False,
                    splits={'training': 0.5, 'validation': 0.5}, batch_size=2560, inplace_transform=compose)
            create_batched_dataset(dataset_path, dest_path=self.tmp_dir_batched2.name, shuffle=False,
                    splits={'training': 0.5, 'validation': 0.5}, batch_size=2560)
        train_inplace = ANIBatchedDataset(self.tmp_dir_batched.name, split='training')
        train = ANIBatchedDataset(self.tmp_dir_batched2.name, transform=compose, split='training')
        for b, inplace_b in zip(train, train_inplace):
            for k in b.keys():
                self.assertEqual(b[k], inplace_b[k])

    def tearDown(self):
        self.tmp_dir_batched.cleanup()
        self.tmp_dir_batched2.cleanup()


class TestANIBatchedDataset(TestCase):

    def setUp(self):
        self.tmp_dir_batched = tempfile.TemporaryDirectory()
        self.tmp_dir_batched2 = tempfile.TemporaryDirectory()
        self.tmp_dir_batched_shuffled = tempfile.TemporaryDirectory()
        self.tmp_dir_batched_path = Path(self.tmp_dir_batched.name).resolve()
        self.tmp_dir_batched_path2 = Path(self.tmp_dir_batched2.name).resolve()
        self.tmp_dir_batched_path_shuffled = Path(self.tmp_dir_batched_shuffled.name).resolve()

        self.batch_size = 2560
        with warnings.catch_warnings():
            ignore_unshuffled_warning()
            create_batched_dataset(dataset_path, dest_path=self.tmp_dir_batched.name, shuffle=False,
                                   splits={'training': 0.5, 'validation': 0.5}, batch_size=self.batch_size,
                                   include_properties=('species', 'coordinates', 'energies'))
        self.train = ANIBatchedDataset(self.tmp_dir_batched.name, split='training')
        self.valid = ANIBatchedDataset(self.tmp_dir_batched.name, split='validation')

    def testInit(self):
        self.assertTrue(self.train.split == 'training')
        self.assertTrue(self.valid.split == 'validation')
        self.assertEqual(len(self.train), 3)
        self.assertEqual(len(self.valid), 3)
        self.assertEqual(self.train.batch_size(0), self.batch_size)
        self.assertEqual(self.valid.batch_size(0), self.batch_size)
        # transform does nothing if no transform was passed
        self.assertTrue(self.train.transform(None) is None)

    def testDropLast(self):
        train_drop_last = ANIBatchedDataset(self.tmp_dir_batched.name, split='training', drop_last=True)
        valid_drop_last = ANIBatchedDataset(self.tmp_dir_batched.name, split='validation', drop_last=True)
        self.assertEqual(len(train_drop_last), 2)
        self.assertEqual(len(valid_drop_last), 2)
        self.assertEqual(train_drop_last.batch_size(-1), self.batch_size)
        self.assertEqual(valid_drop_last.batch_size(-1), self.batch_size)
        for b in train_drop_last:
            self.assertTrue(len(b['coordinates']), self.batch_size)
        for b in valid_drop_last:
            self.assertTrue(len(b['coordinates']), self.batch_size)

    def testKeys(self):
        for batch in self.train:
            self.assertTrue(set(batch.keys()) == {'species', 'coordinates', 'energies'})
        for batch in self.valid:
            self.assertTrue(set(batch.keys()) == {'species', 'coordinates', 'energies'})

    def testNumConformers(self):
        # check that the number of conformers is consistent
        h5 = ANIDataset(dataset_path)
        num_conformers_batched = [len(b['species']) for b in self.train] + [len(b['species']) for b in self.valid]
        num_conformers_batched = sum(num_conformers_batched)
        self.assertEqual(h5.num_conformers, num_conformers_batched)

    def testShuffle(self):
        # thest that shuffling at creation time mixes up conformers a lot
        create_batched_dataset(dataset_path, dest_path=self.tmp_dir_batched_shuffled.name, shuffle=True,
                shuffle_seed=12345,
                splits={'training': 0.5, 'validation': 0.5}, batch_size=self.batch_size)
        train = ANIBatchedDataset(self.tmp_dir_batched_shuffled.name, split='training')
        valid = ANIBatchedDataset(self.tmp_dir_batched_shuffled.name, split='validation')
        # shuffling mixes the conformers a lot, so all batches have pads with -1
        for batch in train:
            self.assertTrue((batch['species'] == -1).any())
        for batch in valid:
            self.assertTrue((batch['species'] == -1).any())

        for batch_ref, batch in zip(self.train, train):
            # as long as the mixing is good enough this should be true
            self.assertTrue(batch_ref['coordinates'].shape != batch['coordinates'].shape)
            self.assertTrue(batch_ref['species'].shape != batch['species'].shape)
            # as long as the permutation is not the identity this should be true
            self.assertTrue((batch_ref['energies'] != batch['energies']).any())

    def testDataLoader(self):
        # check that yielding from the dataloader is equal

        with warnings.catch_warnings():
            ignore_unshuffled_warning()
            train_dataloader = torch.utils.data.DataLoader(self.train, shuffle=False, batch_size=None)
        for batch_ref, batch in zip(self.train, train_dataloader):
            for k_ref in batch_ref:
                self.assertEqual(batch_ref[k_ref], batch[k_ref])

    def testCache(self):
        # check that yielding from the cache is equal to non cache
        with warnings.catch_warnings():
            ignore_unshuffled_warning()
            train_non_cache = torch.utils.data.DataLoader(self.train,
                                                          shuffle=False,
                                                          batch_size=None)
            train_cache = torch.utils.data.DataLoader(deepcopy(self.train).cache(pin_memory=False),
                                                      shuffle=False,
                                                      batch_size=None)
        for batch_ref, batch in zip(train_non_cache, train_cache):
            for k_ref in batch_ref:
                self.assertEqual(batch_ref[k_ref], batch[k_ref])

    def testDataLoaderShuffle(self):
        # check that shuffling with dataloader mixes batches
        generator = torch.manual_seed(5521)
        train_dataloader = torch.utils.data.DataLoader(self.train, shuffle=True, batch_size=None, generator=generator)
        different_batches = 0
        for batch_ref, batch in zip(self.train, train_dataloader):
            for k_ref in batch_ref:
                if batch_ref['energies'].shape == batch['energies'].shape:
                    if (batch_ref['energies'] != batch['energies']).any():
                        different_batches += 1
                else:
                    different_batches += 1
        self.assertTrue(different_batches > 0)

    def testFileFormats(self):
        # check that batches created with all file formats are equal
        for ff in ANIBatchedDataset._SUFFIXES_AND_FORMATS.values():
            self.tmp_dir_batched2 = tempfile.TemporaryDirectory()
            with warnings.catch_warnings():
                ignore_unshuffled_warning()
                create_batched_dataset(dataset_path,
                        dest_path=self.tmp_dir_batched2.name, shuffle=False,
                        splits={'training': 0.5, 'validation': 0.5}, batch_size=self.batch_size)
            train = ANIBatchedDataset(self.tmp_dir_batched2.name, split='training')
            valid = ANIBatchedDataset(self.tmp_dir_batched2.name, split='validation')
            for batch_ref, batch in zip(self.train, train):
                for k_ref in batch_ref:
                    self.assertEqual(batch_ref[k_ref], batch[k_ref])

            for batch_ref, batch in zip(self.valid, valid):
                for k_ref in batch_ref:
                    self.assertEqual(batch_ref[k_ref], batch[k_ref])
            self.tmp_dir_batched2.cleanup()

    def tearDown(self):
        self.tmp_dir_batched.cleanup()
        self.tmp_dir_batched2.cleanup()
        self.tmp_dir_batched_shuffled.cleanup()


class TestANIDataset(TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.num_conformers = [7, 5, 8]
        numpy_conformers = {'HCNN': {'species': np.array(['H', 'C', 'N', 'N'], dtype='S'),
                                     'coordinates': self.rng.standard_normal((self.num_conformers[0], 4, 3)),
                                     'energies': self.rng.standard_normal((self.num_conformers[0],))},
                            'HOO': {'species': np.array(['H', 'O', 'O'], dtype='S'),
                                    'coordinates': self.rng.standard_normal((self.num_conformers[1], 3, 3)),
                                    'energies': self.rng.standard_normal((self.num_conformers[1],))},
                            'HCHHH': {'species': np.array(['H', 'C', 'H', 'H', 'H'], dtype='S'),
                                      'coordinates': self.rng.standard_normal((self.num_conformers[2], 5, 3)),
                                      'energies': self.rng.standard_normal((self.num_conformers[2],))}}

        # extra groups for appending
        self.torch_conformers = {'H6': {'species': torch.ones((5, 6), dtype=torch.long),
                                        'coordinates': torch.randn((5, 6, 3)),
                                        'energies': torch.randn((5,))},
                                 'C6': {'species': torch.full((5, 6), fill_value=6, dtype=torch.long),
                                        'coordinates': torch.randn((5, 6, 3)),
                                        'energies': torch.randn((5,))},
                                 'O6': {'species': torch.full((5, 6), fill_value=8, dtype=torch.long),
                                        'coordinates': torch.randn((5, 6, 3)),
                                        'energies': torch.randn((5,))}}
        self._make_random_test_data(numpy_conformers)

    def _make_random_test_data(self, numpy_conformers):
        # create two HDF5 databases, one with 3 groups and one with one
        # group, and fill them with some random data
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_store_one_group = tempfile.NamedTemporaryFile(suffix='.h5')
        self.tmp_store_three_groups = tempfile.NamedTemporaryFile(suffix='.h5')
        self.new_store_name = self.tmp_dir.name / Path('new.h5')

        with h5py.File(self.tmp_store_one_group, 'r+') as f1,\
             h5py.File(self.tmp_store_three_groups, 'r+') as f3:
            for j, (k, g) in enumerate(numpy_conformers.items()):
                f3.create_group(''.join(k))
                for p, v in g.items():
                    f3[k].create_dataset(p, data=v)
                if j == 0:
                    f1.create_group(''.join(k))
                    for p, v in g.items():
                        f1[k].create_dataset(p, data=v)

    def _make_new_dataset(self):
        return ANIDataset(self.new_store_name, create=True)

    def tearDown(self):
        self.tmp_dir.cleanup()
        self.tmp_store_one_group.close()
        self.tmp_store_three_groups.close()

    def testPresentElements(self):
        ds = self._make_new_dataset()
        new_groups = deepcopy(self.torch_conformers)
        for k in ('H6', 'O6', 'C6'):
            ds.append_conformers(k, new_groups[k])
        self.assertTrue(ds.present_elements(chem_symbols=True), ['C', 'H', 'O'])
        with self.assertRaisesRegex(ValueError, 'Either species or numbers'):
            ds.delete_properties({'species'})
            ds.present_elements()

    def testGetConformers(self):
        ds = ANIDataset(self.tmp_store_three_groups.name)

        # general getter of all conformers
        self.assertEqual(ds.get_conformers('HOO')['coordinates'], ds['HOO']['coordinates'].numpy())
        # test getting 1, 2, ... with a list
        idxs = [1, 2, 4]
        conformers = ds.get_conformers('HCHHH', idxs)
        self.assertEqual(conformers['coordinates'], ds['HCHHH']['coordinates'][torch.tensor(idxs)])
        self.assertEqual(conformers['energies'], ds['HCHHH']['energies'][torch.tensor(idxs)])

        # same with a tensor
        conformers = ds.get_conformers('HCHHH', torch.tensor(idxs))
        self.assertEqual(conformers['coordinates'], ds['HCHHH']['coordinates'][torch.tensor(idxs)])
        self.assertEqual(conformers['energies'], ds['HCHHH']['energies'][torch.tensor(idxs)])

        # same with a ndarray
        conformers = ds.get_conformers('HCHHH', np.array(idxs))
        self.assertEqual(conformers['coordinates'], ds['HCHHH']['coordinates'][torch.tensor(idxs)])
        self.assertEqual(conformers['energies'], ds['HCHHH']['energies'][torch.tensor(idxs)])

        # indices in decreasing order
        conformers = ds.get_conformers('HCHHH', list(reversed(idxs)))
        self.assertEqual(conformers['coordinates'], ds['HCHHH']['coordinates'][torch.tensor(list(reversed(idxs)))])

        # getting some equal conformers
        conformers = ds.get_conformers('HCHHH', torch.tensor(idxs + idxs))
        self.assertEqual(conformers['coordinates'], ds['HCHHH']['coordinates'][torch.tensor(idxs + idxs)])

        # getting just the energies
        conformers = ds.get_conformers('HCHHH', idxs, properties='energies')
        self.assertEqual(conformers['energies'], ds['HCHHH']['energies'][torch.tensor(idxs)])
        self.assertTrue(conformers.get('species', None) is None)
        self.assertTrue(conformers.get('coordinates', None) is None)

    def testAppendAndDeleteConformers(self):
        # tests delitem and setitem analogues for the dataset
        ds = self._make_new_dataset()

        # check creation
        new_groups = deepcopy(self.torch_conformers)
        for k in ('H6', 'C6', 'O6'):
            ds.append_conformers(k, new_groups[k])

        for k, v in ds.items():
            self.assertEqual(v, new_groups[k])
        for k in ('H6', 'C6', 'O6'):
            ds.delete_conformers(k)
        self.assertTrue(len(ds.items()) == 0)

        # check appending
        new_lengths = dict()
        for k in ds.keys():
            new_lengths[k] = len(new_groups[k]['energies']) * 2
            ds.append_conformers(k, new_groups[k])
        for k in ds.keys():
            self.assertEqual(len(ds.get_conformers(k)['energies']), new_lengths[k])
            self.assertEqual(len(ds.get_conformers(k)['species']), len(new_groups['O6']['species']))
        for k in deepcopy(ds.keys()):
            ds.delete_conformers(k)

        # rebuild dataset
        for k in ('H6', 'C6', 'O6'):
            ds.append_conformers(k, new_groups[k])

        with self.assertRaisesRegex(ValueError, 'Character "/" not supported'):
            ds.append_conformers('O/6', new_groups['O6'])
        with self.assertRaisesRegex(ValueError, 'Expected .* but got .*'):
            new_groups_copy = deepcopy(new_groups['O6'])
            del new_groups_copy['energies']
            ds.append_conformers('O6', new_groups_copy)
        with self.assertRaisesRegex(ValueError, 'All appended conformers'):
            new_groups_copy = deepcopy(new_groups['O6'])
            new_groups_copy['species'] = torch.randint(size=(5, 6), low=1, high=5, dtype=torch.long)
            ds.append_conformers('O6', new_groups_copy)
        with self.assertRaisesRegex(ValueError, 'Species needs to have two'):
            new_groups_copy = deepcopy(new_groups['O6'])
            new_groups_copy['species'] = torch.ones((5, 6, 1), dtype=torch.long)
            ds.append_conformers('O6', new_groups_copy)

    def testChunkedIteration(self):
        ds = self._make_new_dataset()
        # first we build numpy conformers with ints and str as species (both
        # allowed)
        conformers = dict()
        for gn in self.torch_conformers.keys():
            conformers[gn] = {k: v.detach().cpu().numpy()
                                  for k, v in self.torch_conformers[gn].items()}

        # Build the dataset using conformers
        for k, v in conformers.items():
            ds.append_conformers(k, v)

        keys = {}
        coords = []
        for k, _, v in ds.chunked_numpy_items(max_size=10):
            coords.append(torch.from_numpy(v['coordinates']))
            keys.update({k})

        keys_large = {}
        coords_large = []
        for k, _, v in ds.chunked_numpy_items(max_size=100000):
            coords_large.append(torch.from_numpy(v['coordinates']))
            keys_large.update({k})

        keys_expect = {}
        coords_expect = []
        for k, v in ds.numpy_items():
            coords_expect.append(torch.from_numpy(v['coordinates']))
            keys_expect.update({k})
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
            conformers_int[gn] = {k: v.detach().cpu().numpy()
                                  for k, v in self.torch_conformers[gn].items()}
            conformers_str[gn] = deepcopy(conformers_int[gn])
            conformers_str[gn]['species'] = _numbers_to_symbols(conformers_int[gn]['species'])

        # Build the dataset using conformers
        for k, v in conformers_str.items():
            ds.append_conformers(k, v)
        # Check that getters give the same result as what was input
        for k, v in ds.numpy_items(chem_symbols=True):
            self.assertEqual(v, conformers_str[k])
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
        initial_len = len(new_groups['C6']['coordinates'])
        for k in ('H6', 'C6', 'O6'):
            ds.append_conformers(k, new_groups[k])
        ds.create_full_property('spin_multiplicities', fill_value=1)
        ds.create_full_property('charges', fill_value=0)
        self.assertEqual(len(ds['H6'].keys()), 5)
        self.assertEqual(ds.properties, {'species', 'energies', 'coordinates', 'charges', 'spin_multiplicities'})
        self.assertEqual(ds['H6']['spin_multiplicities'], torch.ones(initial_len, dtype=torch.long))
        self.assertEqual(ds['C6']['charges'], torch.zeros(initial_len, dtype=torch.long))

    def testRegroupFormulas(self):
        ds = self._make_new_dataset()
        new_groups = deepcopy(self.torch_conformers)
        for j, k in enumerate(('H6', 'C6', 'O6')):
            ds.append_conformers(f'group{j}', new_groups[k])
        ds.regroup_by_formula()
        for k, v in ds.items():
            self.assertEqual(v, new_groups[k])

    def testMetadata(self):
        ds = ANIDataset(locations=self.tmp_dir.name / Path('new.h5'), names="newfile", create=True)
        meta = {'newfile': {'some metadata': 'metadata string', 'other metadata': 'other string'}}
        ds.set_metadata(meta)
        self.assertEqual(ds.metadata, meta)

    def testRegroupNumAtoms(self):
        ds = self._make_new_dataset()
        new_groups = deepcopy(self.torch_conformers)
        for j, k in enumerate(('H6', 'C6', 'O6')):
            ds.append_conformers(f'group{j}', new_groups[k])
        ds.regroup_by_num_atoms()
        self.assertEqual(len(ds.items()), 1)
        self.assertEqual(len(ds['006']['coordinates']), 15)
        self.assertEqual(len(ds['006']['species']), 15)
        self.assertEqual(len(ds['006']['energies']), 15)

    def testDeleteProperty(self):
        ds = self._make_new_dataset()
        new_groups = deepcopy(self.torch_conformers)
        for k in ('H6', 'C6', 'O6'):
            ds.append_conformers(k, new_groups[k])
        ds.delete_properties({'energies'})
        for k, v in ds.items():
            self.assertEqual(set(v.keys()), {'species', 'coordinates'})
            self.assertEqual(v['species'], new_groups[k]['species'])
            self.assertEqual(v['coordinates'], new_groups[k]['coordinates'])
        # deletion of everything kills all groups
        ds.delete_properties({'species', 'coordinates'})
        self.assertEqual(len(ds.items()), 0)

    def testRenameProperty(self):
        ds = self._make_new_dataset()
        new_groups = deepcopy(self.torch_conformers)
        for k in ('H6', 'C6', 'O6'):
            ds.append_conformers(k, new_groups[k])
        ds.rename_properties({'energies': 'renamed_energies'})
        for k, v in ds.items():
            self.assertEqual(set(v.keys()), {'species', 'coordinates', 'renamed_energies'})
        with self.assertRaisesRegex(ValueError, "Some of the properties requested"):
            ds.rename_properties({'null0': 'null1'})
        with self.assertRaisesRegex(ValueError, "Some of the properties requested"):
            ds.rename_properties({'species': 'renamed_energies'})

    def testCreation(self):
        with self.assertRaisesRegex(FileNotFoundError, "The store in .* could not be found"):
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
        self.assertEqual(keys, {'HOO', 'HCNN', 'HCHHH'})
        self.assertEqual(len(ds.keys()), 3)

    def testValues(self):
        ds = ANIDataset(self.tmp_store_three_groups.name)
        for d in ds.values():
            self.assertEqual(set(d.keys()), {'species', 'coordinates', 'energies'})
            self.assertEqual(d['coordinates'].shape[-1], 3)
            self.assertEqual(d['coordinates'].shape[0], d['energies'].shape[0])
        self.assertEqual(len(ds.values()), 3)

    def testItems(self):
        ds = ANIDataset(self.tmp_store_three_groups.name)
        for k, v in ds.items():
            self.assertTrue(isinstance(k, str))
            self.assertTrue(isinstance(v, dict))
            self.assertEqual(set(v.keys()), {'species', 'coordinates', 'energies'})
        self.assertEqual(len(ds.items()), 3)

    def testNumpyItems(self):
        ds = ANIDataset(self.tmp_store_three_groups.name)
        for k, v in ds.numpy_items():
            self.assertTrue(isinstance(k, str))
            self.assertTrue(isinstance(v, dict))
            self.assertEqual(set(v.keys()), {'species', 'coordinates', 'energies'})
        self.assertEqual(len(ds.items()), 3)

    def testDummyPropertiesAlreadyPresent(self):
        # creating dummy properties in a dataset that already has them does nothing
        ds = ANIDataset(self.tmp_store_three_groups.name)
        for k, v in ds.numpy_items(limit=1):
            expect_coords = v['coordinates']
        ds = ANIDataset(self.tmp_store_three_groups.name, dummy_properties={'coordinates': {'fill_value': 0}})
        for k, v in ds.numpy_items(limit=1):
            self.assertEqual(v['coordinates'], expect_coords)

    def testDummyPropertiesRegroup(self):
        # creating dummy properties in a dataset that already has them does nothing
        ANIDataset(self.tmp_store_three_groups.name).regroup_by_num_atoms()
        ds = ANIDataset(self.tmp_store_three_groups.name, dummy_properties={'charges': dict(), 'dipoles': dict()})
        self.assertEqual(ds.properties, {'charges', 'dipoles', 'species', 'coordinates', 'energies'})
        ds = ANIDataset(self.tmp_store_three_groups.name)
        self.assertEqual(ds.properties, {'species', 'coordinates', 'energies'})

    def testDummyPropertiesAppend(self):
        # creating dummy properties in a dataset that already has them does nothing
        ANIDataset(self.tmp_store_three_groups.name).regroup_by_num_atoms()
        ds = ANIDataset(self.tmp_store_three_groups.name, dummy_properties={'charges': dict()})
        ds.append_conformers("003", {'species': np.asarray([[1, 1, 1]], dtype=np.int64),
                                     'energies': np.asarray([10.0], dtype=np.float64),
                                     'coordinates': np.random.standard_normal((1, 3, 3)).astype(np.float32),
                                     'charges': np.asarray([1], dtype=np.int64)})
        ds = ANIDataset(self.tmp_store_three_groups.name)
        self.assertEqual(ds.properties, {'species', 'coordinates', 'energies', 'charges'})

    def testDummyPropertiesNotPresent(self):
        charges_params = {'fill_value': 0, 'dtype': np.int64, 'extra_dims': tuple(), 'is_atomic': False}
        dipoles_params = {'fill_value': 1, 'dtype': np.float64, 'extra_dims': (3,), 'is_atomic': True}
        ANIDataset(self.tmp_store_three_groups.name).regroup_by_num_atoms()
        ds = ANIDataset(self.tmp_store_three_groups.name, dummy_properties={'charges': charges_params, 'atomic_dipoles': dipoles_params})
        self.assertEqual(ds.properties, {'species', 'coordinates', 'energies', 'charges', 'atomic_dipoles'})
        for k, v in ds.numpy_items():
            self.assertTrue((v['charges'] == 0).all())
            self.assertTrue(v['charges'].dtype == np.int64)
            self.assertEqual(v['charges'].shape, (v['species'].shape[0],))
            self.assertTrue((v['atomic_dipoles'] == 1.0).all())
            self.assertTrue(v['atomic_dipoles'].dtype == np.float64)
            self.assertEqual(v['atomic_dipoles'].shape, v['species'].shape + (3,))
            self.assertEqual(set(v.keys()), {'species', 'coordinates', 'energies', 'charges', 'atomic_dipoles'})

        # renaming works as expected
        ds.rename_properties({'charges': 'other_charges', 'atomic_dipoles': 'other_atomic_dipoles'})
        self.assertEqual(ds.properties, {'species', 'coordinates', 'energies', 'other_charges', 'other_atomic_dipoles'})
        for k, v in ds.numpy_items():
            self.assertTrue((v['other_charges'] == 0).all())
            self.assertTrue((v['other_atomic_dipoles'] == 1.0).all())
            self.assertEqual(set(v.keys()), {'species', 'coordinates', 'energies', 'other_charges', 'other_atomic_dipoles'})

        # deleting works as expected
        ds.delete_properties(('other_charges', 'other_atomic_dipoles'))
        self.assertEqual(ds.properties, {'species', 'coordinates', 'energies'})
        for k, v in ds.numpy_items():
            self.assertEqual(set(v.keys()), {'species', 'coordinates', 'energies'})

    def testIterConformers(self):
        ds = ANIDataset(self.tmp_store_three_groups.name)
        confs = []
        for c in ds.iter_conformers():
            self.assertTrue(isinstance(c, dict))
            confs.append(c)
        self.assertEqual(len(confs), ds.num_conformers)


@unittest.skipIf(not ZARR_AVAILABLE, 'Zarr not installed')
class TestANIDatasetZarr(TestANIDataset):
    def _make_random_test_data(self, numpy_conformers):
        for j, (k, v) in enumerate(deepcopy(numpy_conformers).items()):
            # Zarr does not support legacy format, so we tile the species and add
            # a "grouping" attribute
            numpy_conformers[k]['species'] = np.tile(numpy_conformers[k]['species'].reshape(1, -1), (self.num_conformers[j], 1))
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_store_one_group = tempfile.TemporaryDirectory(suffix='.zarr', dir=self.tmp_dir.name)
        self.tmp_store_three_groups = tempfile.TemporaryDirectory(suffix='.zarr', dir=self.tmp_dir.name)
        self.new_store_name = self.tmp_dir.name / Path('new.zarr')

        store1 = zarr.DirectoryStore(self.tmp_store_one_group.name)
        store3 = zarr.DirectoryStore(self.tmp_store_three_groups.name)
        with zarr.hierarchy.open_group(store1, mode='w') as f1,\
             zarr.hierarchy.open_group(store3, mode='w') as f3:
            f3.attrs['grouping'] = 'by_formula'
            f1.attrs['grouping'] = 'by_formula'
            for j, (k, g) in enumerate(numpy_conformers.items()):
                f3.create_group(''.join(k))
                for p, v in g.items():
                    f3[k].create_dataset(p, data=v)
                if j == 0:
                    f1.create_group(''.join(k))
                    for p, v in g.items():
                        f1[k].create_dataset(p, data=v)

    def tearDown(self):
        self.tmp_store_one_group.cleanup()
        self.tmp_store_three_groups.cleanup()
        self.tmp_dir.cleanup()

    def testConvert(self):
        self._testConvert('zarr')

    def _testConvert(self, backend):
        ds = ANIDataset(self.tmp_store_three_groups.name)
        ds.to_backend('h5py', inplace=True)
        for d in ds.values():
            self.assertEqual(set(d.keys()), {'species', 'coordinates', 'energies'})
            self.assertEqual(d['coordinates'].shape[-1], 3)
            self.assertEqual(d['coordinates'].shape[0], d['energies'].shape[0])
        self.assertEqual(len(ds.values()), 3)
        ds.to_backend(backend, inplace=True)
        for d in ds.values():
            self.assertEqual(set(d.keys()), {'species', 'coordinates', 'energies'})
            self.assertEqual(d['coordinates'].shape[-1], 3)
            self.assertEqual(d['coordinates'].shape[0], d['energies'].shape[0])
        self.assertEqual(len(ds.values()), 3)


@unittest.skipIf(not PANDAS_AVAILABLE, 'pandas not installed')
class TestANIDatasetPandas(TestANIDatasetZarr):
    def _make_random_test_data(self, numpy_conformers):
        for j, (k, v) in enumerate(deepcopy(numpy_conformers).items()):
            # Parquet does not support legacy format, so we tile the species and add
            # a "grouping" attribute
            numpy_conformers[k]['species'] = np.tile(numpy_conformers[k]['species'].reshape(1, -1), (self.num_conformers[j], 1))
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_store_one_group = tempfile.TemporaryDirectory(suffix='.pqdir', dir=self.tmp_dir.name)
        self.tmp_store_three_groups = tempfile.TemporaryDirectory(suffix='.pqdir', dir=self.tmp_dir.name)
        self.new_store_name = self.tmp_dir.name / Path('new.pqdir')

        f1 = pandas.DataFrame()
        f3 = pandas.DataFrame()
        meta = {'grouping': 'by_formula',
                'extra_dims': {'coordinates': (3,)},
                'dtypes': {'coordinates': np.dtype(np.float32).name, 'species': np.dtype(np.int64).name, 'energies': np.dtype(np.float64).name}}
        with open(Path(self.tmp_store_one_group.name) / Path(self.tmp_store_one_group.name).with_suffix('.json').name, 'x') as f:
            json.dump(meta, f)

        with open(Path(self.tmp_store_three_groups.name) / Path(self.tmp_store_three_groups.name).with_suffix('.json').name, 'x') as f:
            json.dump(meta, f)

        frames = []
        for j, (k, g) in enumerate(numpy_conformers.items()):
            num_conformations = g['species'].shape[0]
            tmp_df = pandas.DataFrame()
            tmp_df['group'] = pandas.Series([k] * num_conformations)
            tmp_df['species'] = pandas.Series(np.vectorize(lambda x: ATOMIC_NUMBERS[x])(g['species'].astype(str)).tolist())
            tmp_df['energies'] = pandas.Series(g['energies'])
            tmp_df['coordinates'] = pandas.Series(g['coordinates'].reshape(num_conformations, -1).tolist())
            frames.append(tmp_df)
        f3 = pandas.concat(frames)
        f1 = frames[0]
        f1.to_parquet(Path(self.tmp_store_one_group.name) / Path(self.tmp_store_one_group.name).with_suffix('.pq').name)
        f3.to_parquet(Path(self.tmp_store_three_groups.name) / Path(self.tmp_store_three_groups.name).with_suffix('.pq').name)

    def testConvert(self):
        self._testConvert('pq')


class TestData(TestCase):

    def testTensorShape(self):
        ds = torchani.data.load(dataset_path).subtract_self_energies(ani1x_sae_dict).species_to_indices().shuffle().collate(batch_size).cache()
        for d in ds:
            species = d['species']
            coordinates = d['coordinates']
            energies = d['energies']
            self.assertEqual(len(species.shape), 2)
            self.assertLessEqual(species.shape[0], batch_size)
            self.assertEqual(len(coordinates.shape), 3)
            self.assertEqual(coordinates.shape[2], 3)
            self.assertEqual(coordinates.shape[:2], species.shape[:2])
            self.assertEqual(len(energies.shape), 1)
            self.assertEqual(coordinates.shape[0], energies.shape[0])

    def testNoUnnecessaryPadding(self):
        ds = torchani.data.load(dataset_path).subtract_self_energies(ani1x_sae_dict).species_to_indices().shuffle().collate(batch_size).cache()
        for d in ds:
            species = d['species']
            non_padding = (species >= 0)[:, -1].nonzero()
            self.assertGreater(non_padding.numel(), 0)

    def testReEnter(self):
        # make sure that a dataset can be iterated multiple times
        ds = torchani.data.load(dataset_path)
        for _ in ds:
            pass
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.subtract_self_energies(ani1x_sae_dict)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.species_to_indices()
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.shuffle()
        entered = False
        for d in ds:
            entered = True
            pass
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.collate(batch_size)
        entered = False
        for d in ds:
            entered = True
            pass
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.cache()
        entered = False
        for d in ds:
            entered = True
            pass
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

    def testShapeInference(self):
        shifter = torchani.EnergyShifter(None)
        ds = torchani.data.load(dataset_path).subtract_self_energies(shifter)
        len(ds)
        ds = ds.species_to_indices()
        len(ds)
        ds = ds.shuffle()
        len(ds)
        ds = ds.collate(batch_size)
        len(ds)

    def testSAE(self):
        shifter = torchani.EnergyShifter(None)
        torchani.data.load(dataset_path).subtract_self_energies(shifter)
        true_self_energies = torch.tensor([-19.354171758844188,
                                           -19.354171758844046,
                                           -54.712238523648587,
                                           -75.162829556770987], dtype=torch.float64)
        self.assertEqual(true_self_energies, shifter.self_energies)

    def testDataloader(self):
        shifter = torchani.EnergyShifter(None)
        dataset = list(torchani.data.load(dataset_path).subtract_self_energies(shifter).species_to_indices().shuffle())
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=torchani.data.collate_fn, num_workers=2)
        for _ in loader:
            pass


if __name__ == '__main__':
    unittest.main()
