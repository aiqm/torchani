import os
from pathlib import Path
import h5py
import numpy as np
import torch
import torchani
import unittest
import tempfile
import shutil
import warnings
from copy import deepcopy
from torchani.transforms import AtomicNumbersToIndices, SubtractSAE, Compose, calculate_saes
from torchani.testing import TestCase
from torchani.datasets import AniH5Dataset, AniBatchedDataset, create_batched_dataset

path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(path, '../dataset/ani-1x/sample.h5')
dataset_path_gdb = os.path.join(path, '../dataset/ani1-up_to_gdb4/ani_gdb_s02.h5')
batch_size = 256
ani1x_sae_dict = {'H': -0.60095298, 'C': -38.08316124, 'N': -54.7077577, 'O': -75.19446356}


def ignore_unshuffled_warning():
    warnings.filterwarnings(action='ignore',
                            message="Dataset will not be shuffled, this should only be used for debugging")


class TestFineGrainedShuffle(TestCase):

    def testShuffleMixesManyH5(self):
        # test that shuffling correctly mixes multiple h5 files
        num_groups = 10
        num_conformers_per_group = 12
        self._create_dummy_controlled_dataset(num_groups, num_conformers_per_group, use_energy_ranges=False)

        self.train = AniBatchedDataset(self.batched_path, split='training')
        self.valid = AniBatchedDataset(self.batched_path, split='validation')
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
            train = AniBatchedDataset(self.batched_path, split=f'training{j}')
            valid = AniBatchedDataset(self.batched_path, split=f'validation{j}')
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

    def _check_disjoint_and_nonduplicates(self, name1, name2):
        train = AniBatchedDataset(self.batched_path, split=name1)
        valid = AniBatchedDataset(self.batched_path, split=name2)
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

    def _create_dummy_controlled_dataset(self, num_groups, num_conformers_per_group, use_energy_ranges=False, folds=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.batched_path = Path('./tmp_dataset').resolve()
            self.dummy_h50 = tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.h5')
            self.dummy_h51 = tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.h5')
            self.dummy_h52 = tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.h5')
            self.rng = np.random.default_rng(12345)
            # each file will have 120 conformers, total 360 conformers
            num_groups = 10
            num_conformers_per_group = 12
            properties = ['species', 'coordinates', 'energies']
            if use_energy_ranges:
                ranges = [0, num_groups * num_conformers_per_group, 2 * num_groups * num_conformers_per_group]
            else:
                ranges = [None, None, None]
            self._create_dummy_file(self.dummy_h50, num_groups, num_conformers_per_group, 'H', 0.0, properties, ranges[0])
            self._create_dummy_file(self.dummy_h51, num_groups, num_conformers_per_group, 'C', 1.0, properties, ranges[1])
            self._create_dummy_file(self.dummy_h52, num_groups, num_conformers_per_group, 'N', 2.0, properties, ranges[2])

            self.batched_path = Path('./tmp_dataset').resolve()
            # both validation and test have 3 batches of 60 each
            if folds is None:
                create_batched_dataset(h5_path=tmpdir, dest_path=self.batched_path, shuffle=True, shuffle_seed=123456789,
                        splits={'training': 0.5, 'validation': 0.5}, batch_size=60)
            else:
                create_batched_dataset(h5_path=tmpdir, dest_path=self.batched_path, shuffle=True, shuffle_seed=123456789,
                        folds=folds, batch_size=60)

    def _create_dummy_file(self, file_, num_groups, num_conformers_per_group, element, factor, properties, range_start=None):
        with h5py.File(file_, 'r+') as f:
            for j in range(num_groups):
                f.create_group(f'group{j}')
                g = f[f'group{j}']
                for k in properties:
                    if k == 'species':
                        g.create_dataset(k, data=np.array([element, element, element], dtype='S'))
                    elif k == 'coordinates':
                        g.create_dataset(k, data=self.rng.standard_normal((num_conformers_per_group, 3, 3)))
                    elif k == 'energies':
                        if range_start is not None:
                            g.create_dataset(k, data=np.arange(range_start + j * num_conformers_per_group,
                                                               range_start + (j + 1) * num_conformers_per_group, dtype=float))
                        else:
                            g.create_dataset(k, data=factor * np.ones((num_conformers_per_group,)))

    def _test_for_batch_diversity(self, b):
        zeros = (b['energies'] == 0.0).count_nonzero()
        ones = (b['energies'] == 1.0).count_nonzero()
        twos = (b['energies'] == 2.0).count_nonzero()
        self.assertTrue(zeros > 0)
        self.assertTrue(ones > 0)
        self.assertTrue(twos > 0)

    def tearDown(self):
        try:
            shutil.rmtree(self.batched_path)
        except Exception:
            pass


class TestEstimationSAE(TestCase):

    def setUp(self):
        self.batched_path = Path('./tmp_dataset').resolve()
        self.batch_size = 2560
        create_batched_dataset(h5_path=dataset_path_gdb, dest_path=self.batched_path, shuffle=True,
                splits={'training': 1.0}, batch_size=self.batch_size, shuffle_seed=12345)
        self.train = AniBatchedDataset(self.batched_path, split='training')

    def testExactSAE(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore',
                                    message="Using all batches to estimate SAE, this may take up a lot of memory.")
            saes, _ = calculate_saes(self.train, ('H', 'C', 'N', 'O'), mode='exact')
            torch.set_printoptions(precision=10)
        self.assertEqual(saes,
                         torch.tensor([-0.5983182192, -38.0726242065, -54.6750144958, -75.1433029175], dtype=torch.float),
                         atol=2.5e-3, rtol=2.5e-3)

    def testStochasticSAE(self):
        saes, _ = calculate_saes(self.train, ('H', 'C', 'N', 'O'), mode='sgd')
        # in this specific case the sae difference is very large because it is a
        # very small sample, but for the full sample this imlementation is correct
        self.assertEqual(saes, torch.tensor([-20.4466, -0.3910, -8.8793, -11.4184], dtype=torch.float),
                         atol=0.2, rtol=0.2)

    def tearDown(self):
        shutil.rmtree(self.batched_path)


class TestTransforms(TestCase):

    def setUp(self):
        self.elements = ('H', 'C', 'N', 'O')
        coordinates = torch.randn((2, 7, 3), dtype=torch.float)
        self.input_ = {'species': torch.tensor([[-1, 1, 1, 6, 1, 7, 8], [1, 1, 1, 1, 1, 1, 6]], dtype=torch.long),
                       'energies': torch.tensor([0.0, 1.0], dtype=torch.float),
                       'coordinates': coordinates}
        self.batched_path = Path('./tmp_dataset').resolve()
        self.batched_path2 = Path('./tmp_dataset2').resolve()

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
            create_batched_dataset(h5_path=dataset_path, dest_path=self.batched_path, shuffle=False,
                    splits={'training': 0.5, 'validation': 0.5}, batch_size=2560, inplace_transform=compose)
            create_batched_dataset(h5_path=dataset_path, dest_path=self.batched_path2, shuffle=False,
                    splits={'training': 0.5, 'validation': 0.5}, batch_size=2560)
        train_inplace = AniBatchedDataset(self.batched_path, split='training')
        train = AniBatchedDataset(self.batched_path2, transform=compose, split='training')
        for b, inplace_b in zip(train, train_inplace):
            for k in b.keys():
                self.assertEqual(b[k], inplace_b[k])

    def tearDown(self):
        try:
            shutil.rmtree(self.batched_path)
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree(self.batched_path2)
        except FileNotFoundError:
            pass


class TestAniBatchedDataset(TestCase):

    def setUp(self):
        self.batched_path = Path('./tmp_dataset').resolve()
        self.batched_path2 = Path('./tmp_dataset2').resolve()
        self.batched_path_shuffled = Path('./tmp_dataset_shuffled').resolve()
        self.batch_size = 2560
        with warnings.catch_warnings():
            ignore_unshuffled_warning()
            create_batched_dataset(h5_path=dataset_path, dest_path=self.batched_path, shuffle=False,
                    splits={'training': 0.5, 'validation': 0.5}, batch_size=self.batch_size)
        self.train = AniBatchedDataset(self.batched_path, split='training')
        self.valid = AniBatchedDataset(self.batched_path, split='validation')

    def testInit(self):
        self.assertTrue(self.train.split == 'training')
        self.assertTrue(self.valid.split == 'validation')
        self.assertEqual(len(self.train), 3)
        self.assertEqual(len(self.valid), 3)
        self.assertEqual(self.train.batch_size, self.batch_size)
        self.assertEqual(self.valid.batch_size, self.batch_size)
        # transform does nothing if no transform was passed
        self.assertTrue(self.train.transform(None) is None)

    def testDropLast(self):
        with warnings.catch_warnings():
            msg = 'Recalculating batch size is necessary for drop_last and it may take considerable time if your disk is an HDD'
            warnings.filterwarnings(action='ignore',
                                    message=msg)
            train_drop_last = AniBatchedDataset(self.batched_path, split='training', drop_last=True)
            valid_drop_last = AniBatchedDataset(self.batched_path, split='validation', drop_last=True)
        self.assertEqual(len(train_drop_last), 2)
        self.assertEqual(len(valid_drop_last), 2)
        self.assertEqual(train_drop_last.batch_size, self.batch_size)
        self.assertEqual(valid_drop_last.batch_size, self.batch_size)
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
        h5 = AniH5Dataset(dataset_path)
        num_conformers_batched = [len(b['species']) for b in self.train] + [len(b['species']) for b in self.valid]
        num_conformers_batched = sum(num_conformers_batched)
        self.assertEqual(h5.num_conformers, num_conformers_batched)

    def testShuffle(self):
        # thest that shuffling at creation time mixes up conformers a lot
        create_batched_dataset(h5_path=dataset_path, dest_path=self.batched_path_shuffled, shuffle=True,
                shuffle_seed=12345,
                splits={'training': 0.5, 'validation': 0.5}, batch_size=self.batch_size)
        train = AniBatchedDataset(self.batched_path_shuffled, split='training')
        valid = AniBatchedDataset(self.batched_path_shuffled, split='validation')
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
        shutil.rmtree(self.batched_path_shuffled)

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
        for ff in AniBatchedDataset.SUPPORTED_FILE_FORMATS:

            with warnings.catch_warnings():
                ignore_unshuffled_warning()
                create_batched_dataset(h5_path=dataset_path,
                        dest_path=self.batched_path2, shuffle=False,
                        splits={'training': 0.5, 'validation': 0.5}, batch_size=self.batch_size)
            train = AniBatchedDataset(self.batched_path2, split='training')
            valid = AniBatchedDataset(self.batched_path2, split='validation')
            for batch_ref, batch in zip(self.train, train):
                for k_ref in batch_ref:
                    self.assertEqual(batch_ref[k_ref], batch[k_ref])

            for batch_ref, batch in zip(self.valid, valid):
                for k_ref in batch_ref:
                    self.assertEqual(batch_ref[k_ref], batch[k_ref])
            shutil.rmtree(self.batched_path2)

    def tearDown(self):
        shutil.rmtree(self.batched_path)
        try:
            shutil.rmtree(self.batched_path2)
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree(self.batched_path_shuffled)
        except FileNotFoundError:
            pass


class TestAniH5Dataset(TestCase):

    def setUp(self):
        # create two dummy HDF5 databases, one with 3 groups and one with one
        # group, and fill them with some data
        self.tf_one_group = tempfile.NamedTemporaryFile()
        self.tf_three_groups = tempfile.NamedTemporaryFile()
        self.rng = np.random.default_rng(12345)
        self.num_conformers1 = 7
        properties1 = {'species': np.array(['H', 'C', 'N', 'N'], dtype='S'),
                      'coordinates': self.rng.standard_normal((self.num_conformers1, 4, 3)),
                      'energies': self.rng.standard_normal((self.num_conformers1,))}
        self.num_conformers2 = 5
        properties2 = {'species': np.array(['H', 'O', 'O'], dtype='S'),
                      'coordinates': self.rng.standard_normal((self.num_conformers2, 3, 3)),
                      'energies': self.rng.standard_normal((self.num_conformers2,))}
        self.num_conformers3 = 8
        properties3 = {'species': np.array(['H', 'C', 'H', 'H', 'H'], dtype='S'),
                      'coordinates': self.rng.standard_normal((self.num_conformers3, 5, 3)),
                      'energies': self.rng.standard_normal((self.num_conformers3,))}
        with h5py.File(self.tf_one_group, 'r+') as f1:
            f1.create_group(''.join(properties1['species'].astype(str).tolist()))
        with h5py.File(self.tf_three_groups, 'r+') as f3:
            f3.create_group(''.join(properties1['species'].astype(str).tolist()))
            f3.create_group(''.join(properties2['species'].astype(str).tolist()))
            f3.create_group(''.join(properties3['species'].astype(str).tolist()))

        with h5py.File(self.tf_one_group, 'r+') as f1:
            for k, v in properties1.items():
                f1['HCNN'].create_dataset(k, data=v)
        with h5py.File(self.tf_three_groups, 'r+') as f3:
            for k, v in properties1.items():
                f3['HCNN'].create_dataset(k, data=v)
            for k, v in properties2.items():
                f3['HOO'].create_dataset(k, data=v)
            for k, v in properties3.items():
                f3['HCHHH'].create_dataset(k, data=v)

    def testSizesOneGroup(self):
        ds = AniH5Dataset(self.tf_one_group.name)
        self.assertEqual(ds.num_conformers, self.num_conformers1)
        self.assertEqual(ds.num_conformer_groups, 1)
        self.assertEqual(len(ds), ds.num_conformer_groups)

    def testSizesThreeGroups(self):
        ds = AniH5Dataset(self.tf_three_groups.name)
        self.assertEqual(ds.num_conformers, self.num_conformers1 + self.num_conformers2 + self.num_conformers3)
        self.assertEqual(ds.num_conformer_groups, 3)
        self.assertEqual(len(ds), ds.num_conformer_groups)

    def testKeys(self):
        ds = AniH5Dataset(self.tf_three_groups.name)
        keys = set()
        for k in ds.keys():
            keys.update({k})
        self.assertTrue(keys == {'/HOO', '/HCNN', '/HCHHH'})
        self.assertEqual(len(ds.keys()), 3)

    def testValues(self):
        ds = AniH5Dataset(self.tf_three_groups.name)
        for d in ds.values():
            self.assertTrue('species' in d.keys())
            self.assertTrue('coordinates' in d.keys())
            self.assertTrue('energies' in d.keys())
            self.assertEqual(d['coordinates'].shape[-1], 3)
            self.assertEqual(d['coordinates'].shape[0], d['energies'].shape[0])
        self.assertEqual(len(ds.values()), 3)

    def testItems(self):
        ds = AniH5Dataset(self.tf_three_groups.name)
        for k, v in ds.items():
            self.assertTrue(isinstance(k, str))
            self.assertTrue(isinstance(v, dict))
            self.assertTrue('species' in v.keys())
            self.assertTrue('coordinates' in v.keys())
            self.assertTrue('energies' in v.keys())
        self.assertEqual(len(ds.items()), 3)

    def testGetConformers(self):
        ds = AniH5Dataset(self.tf_three_groups.name)

        self.assertEqual(ds.get_conformers('HOO')['coordinates'], ds['HOO']['coordinates'])
        self.assertEqual(ds.get_conformers('HOO', 0)['coordinates'], ds['HOO']['coordinates'][0])
        conformers12 = ds.get_conformers('HCHHH', np.array([1, 2]))
        self.assertEqual(conformers12['coordinates'], ds['HCHHH']['coordinates'][np.array([1, 2])])
        # note that h5py does not allow this directly
        conformers12 = ds.get_conformers('HCHHH', np.array([2, 1]))
        self.assertEqual(conformers12['coordinates'], ds['HCHHH']['coordinates'][np.array([2, 1])])
        # note that h5py does not allow this directly
        conformers12 = ds.get_conformers('HCHHH', np.array([1, 1]))
        self.assertEqual(conformers12['coordinates'], ds['HCHHH']['coordinates'][np.array([1, 1])])
        conformers124 = ds.get_conformers('HCHHH', np.array([1, 2, 4]), include_properties=('energies',))
        self.assertEqual(conformers124['energies'], ds['HCHHH']['energies'][np.array([1, 2, 4])])
        self.assertTrue(conformers124.get('species', None) is None)
        self.assertTrue(conformers124.get('coordinates', None) is None)

    def testIterConformers(self):
        ds = AniH5Dataset(self.tf_three_groups.name)
        confs = []
        for c in ds.iter_conformers():
            self.assertTrue(isinstance(c, dict))
            confs.append(c)
        self.assertEqual(len(confs), ds.num_conformers)


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
