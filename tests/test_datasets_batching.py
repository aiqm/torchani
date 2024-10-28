import tempfile
from copy import deepcopy
import os
from pathlib import Path

import torch
import h5py
import numpy as np

from torchani._testing import TestCase
from torchani.datasets import (
    create_batched_dataset,
    ANIDataset,
    ANIBatchedDataset,
)

path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(path, "../dataset/ani-1x/sample.h5")


class TestANIBatchedDataset(TestCase):
    def setUp(self):
        self.tmp_dir_batched = tempfile.TemporaryDirectory()
        self.tmp_dir_batched2 = tempfile.TemporaryDirectory()
        self.tmp_dir_batched_shuffled = tempfile.TemporaryDirectory()
        self.tmp_dir_batched_path = Path(self.tmp_dir_batched.name).resolve()
        self.tmp_dir_batched_path2 = Path(self.tmp_dir_batched2.name).resolve()
        self.tmp_dir_batched_path_shuffled = Path(
            self.tmp_dir_batched_shuffled.name
        ).resolve()

        self.batch_size = 2560
        create_batched_dataset(
            dataset_path,
            dest_path=self.tmp_dir_batched.name,
            _shuffle=False,
            splits={"training": 0.5, "validation": 0.5},
            batch_size=self.batch_size,
            properties=("species", "coordinates", "energies"),
        )
        self.train = ANIBatchedDataset(self.tmp_dir_batched.name, split="training")
        self.valid = ANIBatchedDataset(self.tmp_dir_batched.name, split="validation")

    def testInit(self):
        self.assertTrue(self.train.div == "training")
        self.assertTrue(self.valid.div == "validation")
        self.assertEqual(len(self.train), 3)
        self.assertEqual(len(self.valid), 3)
        self.assertEqual(self.train._batch_size(self.train[0]), self.batch_size)
        self.assertEqual(self.valid._batch_size(self.valid[0]), self.batch_size)
        # transform does nothing if no transform was passed
        self.assertTrue(self.train.transform(None) is None)

    def testDropLast(self):
        train_drop_last = ANIBatchedDataset(
            self.tmp_dir_batched.name, split="training", drop_last=True
        )
        valid_drop_last = ANIBatchedDataset(
            self.tmp_dir_batched.name, split="validation", drop_last=True
        )
        self.assertEqual(len(train_drop_last), 2)
        self.assertEqual(len(valid_drop_last), 2)
        self.assertEqual(
            train_drop_last._batch_size(train_drop_last[-1]), self.batch_size
        )
        self.assertEqual(
            valid_drop_last._batch_size(valid_drop_last[-1]), self.batch_size
        )
        for b in train_drop_last:
            self.assertTrue(len(b["coordinates"]), self.batch_size)
        for b in valid_drop_last:
            self.assertTrue(len(b["coordinates"]), self.batch_size)

    def testKeys(self):
        for batch in self.train:
            self.assertTrue(set(batch.keys()) == {"species", "coordinates", "energies"})
        for batch in self.valid:
            self.assertTrue(set(batch.keys()) == {"species", "coordinates", "energies"})

    def testNumConformers(self):
        # check that the number of conformers is consistent
        h5 = ANIDataset(dataset_path)
        _num_conformers_batched = [len(b["species"]) for b in self.train] + [
            len(b["species"]) for b in self.valid
        ]
        num_conformers_batched = sum(_num_conformers_batched)
        self.assertEqual(h5.num_conformers, num_conformers_batched)

    def testShuffle(self):
        # thest that shuffling at creation time mixes up conformers a lot
        create_batched_dataset(
            dataset_path,
            dest_path=self.tmp_dir_batched_shuffled.name,
            divs_seed=12345,
            batch_seed=12345,
            splits={"training": 0.5, "validation": 0.5},
            batch_size=self.batch_size,
        )
        train = ANIBatchedDataset(self.tmp_dir_batched_shuffled.name, split="training")
        valid = ANIBatchedDataset(
            self.tmp_dir_batched_shuffled.name, split="validation"
        )
        # shuffling mixes the conformers a lot, so all batches have pads with -1
        for batch in train:
            self.assertTrue((batch["species"] == -1).any())
        for batch in valid:
            self.assertTrue((batch["species"] == -1).any())

        for batch_ref, batch in zip(self.train, train):
            # as long as the mixing is good enough this should be true
            self.assertTrue(
                batch_ref["coordinates"].shape != batch["coordinates"].shape
            )
            self.assertTrue(batch_ref["species"].shape != batch["species"].shape)
            # as long as the permutation is not the identity this should be true
            self.assertTrue((batch_ref["energies"] != batch["energies"]).any())

    def testDataLoader(self):
        # check that yielding from the dataloader is equal

        train_dataloader = torch.utils.data.DataLoader(
            self.train, shuffle=False, batch_size=None
        )
        for batch_ref, batch in zip(self.train, train_dataloader):
            for k_ref in batch_ref:
                self.assertEqual(batch_ref[k_ref], batch[k_ref])

    def testCache(self):
        # check that yielding from the cache is equal to non cache
        train_non_cache = torch.utils.data.DataLoader(
            self.train, shuffle=False, batch_size=None
        )
        train_cache = torch.utils.data.DataLoader(
            deepcopy(self.train).cache(pin_memory=False),
            shuffle=False,
            batch_size=None,
        )
        for batch_ref, batch in zip(train_non_cache, train_cache):
            for k_ref in batch_ref:
                self.assertEqual(batch_ref[k_ref], batch[k_ref])

    def testDataLoaderShuffle(self):
        # check that shuffling with dataloader mixes batches
        generator = torch.manual_seed(5521)
        train_dataloader = torch.utils.data.DataLoader(
            self.train, shuffle=True, batch_size=None, generator=generator
        )
        different_batches = 0
        for batch_ref, batch in zip(self.train, train_dataloader):
            for k_ref in batch_ref:
                if batch_ref["energies"].shape == batch["energies"].shape:
                    if (batch_ref["energies"] != batch["energies"]).any():
                        different_batches += 1
                else:
                    different_batches += 1
        self.assertTrue(different_batches > 0)

    def testFileFormats(self):
        # check that batches created with all file formats are equal
        self.tmp_dir_batched2 = tempfile.TemporaryDirectory()
        create_batched_dataset(
            dataset_path,
            dest_path=self.tmp_dir_batched2.name,
            splits={"training": 0.5, "validation": 0.5},
            batch_size=self.batch_size,
            _shuffle=False,
        )
        train = ANIBatchedDataset(self.tmp_dir_batched2.name, split="training")
        valid = ANIBatchedDataset(self.tmp_dir_batched2.name, split="validation")
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


class TestBatchingCorrectShuffle(TestCase):
    def setUp(self):
        self.tmp_dir_batched = tempfile.TemporaryDirectory()

    def testShuffleMixesManyH5(self):
        # test that shuffling correctly mixes multiple h5 files
        num_groups = 10
        num_conformers_per_group = 12
        self._create_dummy_dataset(
            num_groups, num_conformers_per_group, use_energy_ranges=False
        )

        self.train = ANIBatchedDataset(self.tmp_dir_batched.name, split="training")
        self.valid = ANIBatchedDataset(self.tmp_dir_batched.name, split="validation")
        for b, b_valid in zip(self.train, self.valid):
            self.assertNotEqual(b["species"], b_valid["species"])
            self.assertNotEqual(b["energies"], b_valid["energies"])
            self._test_for_batch_diversity(b)
            self._test_for_batch_diversity(b_valid)

    def testShuffleMixesManyH5Folds(self):
        # test that shuffling correctly mixes multiple h5 files
        num_groups = 10
        num_conformers_per_group = 12
        folds = 3
        self._create_dummy_dataset(
            num_groups, num_conformers_per_group, use_energy_ranges=False, folds=3
        )

        def check_train_valid(train, valid):
            for b, b_valid in zip(train, valid):
                self.assertNotEqual(b["species"], b_valid["species"])
                self.assertNotEqual(b["energies"], b_valid["energies"])
                self._test_for_batch_diversity(b)
                self._test_for_batch_diversity(b_valid)

        for j in range(folds):
            train = ANIBatchedDataset(self.tmp_dir_batched.name, split=f"training{j}")
            valid = ANIBatchedDataset(self.tmp_dir_batched.name, split=f"validation{j}")
            check_train_valid(train, valid)

    def testDisjointFolds(self):
        # test that shuffling generates disjoint train and validation, with no
        # duplicates
        num_groups = 10
        num_conformers_per_group = 12
        folds = 5
        self._create_dummy_dataset(
            num_groups, num_conformers_per_group, use_energy_ranges=True, folds=folds
        )

        for j in range(folds):
            self._check_disjoint_and_nonduplicates(f"training{j}", f"validation{j}")

        for j in range(folds):
            for k in range(j + 1, folds):
                self._check_disjoint_and_nonduplicates(
                    f"validation{j}", f"validation{k}"
                )

    def testDisjointTrainValid(self):
        # test that shuffling generates disjoint train and validation, with no
        # duplicates
        num_groups = 10
        num_conformers_per_group = 12
        self._create_dummy_dataset(
            num_groups, num_conformers_per_group, use_energy_ranges=True
        )
        self._check_disjoint_and_nonduplicates("training", "validation")

    # creates a dataset inside self.tmp_dir_batched
    def _create_dummy_dataset(
        self, num_groups, num_conformers_per_group, use_energy_ranges=False, folds=None
    ):
        def create_dummy_file(
            rng,
            file_,
            num_groups,
            num_conformers_per_group,
            element,
            factor,
            properties,
            range_start=None,
        ):
            with h5py.File(file_, "r+") as f:
                for j in range(num_groups):
                    f.create_group(f"group{j}")
                    g = f[f"group{j}"]
                    for k in properties:
                        if k == "species":
                            g.create_dataset(
                                k, data=np.array([element, element, element], dtype="S")
                            )
                        elif k == "coordinates":
                            g.create_dataset(
                                k,
                                data=rng.standard_normal(
                                    (num_conformers_per_group, 3, 3)
                                ),
                            )
                        elif k == "energies":
                            if range_start is not None:
                                g.create_dataset(
                                    k,
                                    data=np.arange(
                                        range_start + j * num_conformers_per_group,
                                        range_start
                                        + (j + 1) * num_conformers_per_group,
                                        dtype=float,
                                    ),
                                )
                            else:
                                g.create_dataset(
                                    k,
                                    data=factor * np.ones((num_conformers_per_group,)),
                                )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Each file will have 120 conformers, total 360 conformers
            num_groups = 10
            num_conformers_per_group = 12
            properties = ["species", "coordinates", "energies"]
            if use_energy_ranges:
                ranges = [
                    0,
                    num_groups * num_conformers_per_group,
                    2 * num_groups * num_conformers_per_group,
                ]
            else:
                ranges = [None, None, None]
            rng = np.random.default_rng(12345)
            with tempfile.NamedTemporaryFile(
                dir=tmpdir, suffix=".h5"
            ) as dummy_h50, tempfile.NamedTemporaryFile(
                dir=tmpdir, suffix=".h5"
            ) as dummy_h51, tempfile.NamedTemporaryFile(
                dir=tmpdir, suffix=".h5"
            ) as dummy_h52:
                create_dummy_file(
                    rng,
                    dummy_h50,
                    num_groups,
                    num_conformers_per_group,
                    "H",
                    0.0,
                    properties,
                    ranges[0],
                )
                create_dummy_file(
                    rng,
                    dummy_h51,
                    num_groups,
                    num_conformers_per_group,
                    "C",
                    1.0,
                    properties,
                    ranges[1],
                )
                create_dummy_file(
                    rng,
                    dummy_h52,
                    num_groups,
                    num_conformers_per_group,
                    "N",
                    2.0,
                    properties,
                    ranges[2],
                )

                # both validation and test have 3 batches of 60 each
                h5_dirs = sorted(Path(tmpdir).iterdir())
                if folds is None:
                    create_batched_dataset(
                        h5_dirs,
                        dest_path=self.tmp_dir_batched.name,
                        divs_seed=123456789,
                        batch_seed=123456789,
                        splits={"training": 0.5, "validation": 0.5},
                        batch_size=60,
                    )
                else:
                    create_batched_dataset(
                        h5_dirs,
                        dest_path=self.tmp_dir_batched.name,
                        divs_seed=123456789,
                        batch_seed=123456789,
                        folds=folds,
                        batch_size=60,
                    )

    def _check_disjoint_and_nonduplicates(self, name1, name2):
        train = ANIBatchedDataset(self.tmp_dir_batched.name, split=name1)
        valid = ANIBatchedDataset(self.tmp_dir_batched.name, split=name2)
        _all_train_energies = []
        _all_valid_energies = []
        for b, b_valid in zip(train, valid):
            _all_train_energies.append(b["energies"])
            _all_valid_energies.append(b_valid["energies"])
        all_train_energies = torch.cat(_all_train_energies)
        all_valid_energies = torch.cat(_all_valid_energies)

        all_train_list = [e.item() for e in all_train_energies]
        all_train_set = set(all_train_list)
        all_valid_list = [e.item() for e in all_valid_energies]
        all_valid_set = set(all_valid_list)
        # no duplicates and disjoint
        self.assertTrue(len(all_train_list) == len(all_train_set))
        self.assertTrue(len(all_valid_list) == len(all_valid_set))
        self.assertTrue(all_train_set.isdisjoint(all_valid_set))

    def _test_for_batch_diversity(self, b):
        zeros = (b["energies"] == 0.0).count_nonzero()
        ones = (b["energies"] == 1.0).count_nonzero()
        twos = (b["energies"] == 2.0).count_nonzero()
        self.assertTrue(zeros > 0)
        self.assertTrue(ones > 0)
        self.assertTrue(twos > 0)

    def tearDown(self):
        self.tmp_dir_batched.cleanup()
