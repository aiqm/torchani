import tempfile
from copy import deepcopy
import os
from pathlib import Path

import torch

from torchani.testing import TestCase
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
