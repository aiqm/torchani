import torchani
import torch
import os
import unittest
from torchani.testing import TestCase


path = os.path.dirname(os.path.realpath(__file__))
iptpath = os.path.join(path, 'test_data/inputtrain.ipt')
dspath = os.path.join(path, '../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5')


class TestNeuroChem(TestCase):

    def testNeuroChemTrainer(self):
        d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = torchani.neurochem.Trainer(iptpath, d, True, os.path.join(path, 'runs'))

        # test if loader construct correct model
        self.assertEqual(trainer.aev_computer.aev_length, 384)
        m = trainer.nn
        H = m['H']
        C = m['C']
        N = m['N']
        O = m['O']  # noqa: E741
        self.assertIsInstance(H[0], torch.nn.Linear)
        self.assertListEqual(list(H[0].weight.shape), [160, 384])
        self.assertIsInstance(H[1], torch.nn.CELU)
        self.assertIsInstance(H[2], torch.nn.Linear)
        self.assertListEqual(list(H[2].weight.shape), [128, 160])
        self.assertIsInstance(H[3], torch.nn.CELU)
        self.assertIsInstance(H[4], torch.nn.Linear)
        self.assertListEqual(list(H[4].weight.shape), [96, 128])
        self.assertIsInstance(H[5], torch.nn.CELU)
        self.assertIsInstance(H[6], torch.nn.Linear)
        self.assertListEqual(list(H[6].weight.shape), [1, 96])
        self.assertEqual(len(H), 7)

        self.assertIsInstance(C[0], torch.nn.Linear)
        self.assertListEqual(list(C[0].weight.shape), [144, 384])
        self.assertIsInstance(C[1], torch.nn.CELU)
        self.assertIsInstance(C[2], torch.nn.Linear)
        self.assertListEqual(list(C[2].weight.shape), [112, 144])
        self.assertIsInstance(C[3], torch.nn.CELU)
        self.assertIsInstance(C[4], torch.nn.Linear)
        self.assertListEqual(list(C[4].weight.shape), [96, 112])
        self.assertIsInstance(C[5], torch.nn.CELU)
        self.assertIsInstance(C[6], torch.nn.Linear)
        self.assertListEqual(list(C[6].weight.shape), [1, 96])
        self.assertEqual(len(C), 7)

        self.assertIsInstance(N[0], torch.nn.Linear)
        self.assertListEqual(list(N[0].weight.shape), [128, 384])
        self.assertIsInstance(N[1], torch.nn.CELU)
        self.assertIsInstance(N[2], torch.nn.Linear)
        self.assertListEqual(list(N[2].weight.shape), [112, 128])
        self.assertIsInstance(N[3], torch.nn.CELU)
        self.assertIsInstance(N[4], torch.nn.Linear)
        self.assertListEqual(list(N[4].weight.shape), [96, 112])
        self.assertIsInstance(N[5], torch.nn.CELU)
        self.assertIsInstance(N[6], torch.nn.Linear)
        self.assertListEqual(list(N[6].weight.shape), [1, 96])
        self.assertEqual(len(N), 7)

        self.assertIsInstance(O[0], torch.nn.Linear)
        self.assertListEqual(list(O[0].weight.shape), [128, 384])
        self.assertIsInstance(O[1], torch.nn.CELU)
        self.assertIsInstance(O[2], torch.nn.Linear)
        self.assertListEqual(list(O[2].weight.shape), [112, 128])
        self.assertIsInstance(O[3], torch.nn.CELU)
        self.assertIsInstance(O[4], torch.nn.Linear)
        self.assertListEqual(list(O[4].weight.shape), [96, 112])
        self.assertIsInstance(O[5], torch.nn.CELU)
        self.assertIsInstance(O[6], torch.nn.Linear)
        self.assertListEqual(list(O[6].weight.shape), [1, 96])
        self.assertEqual(len(O), 7)

        self.assertEqual(trainer.init_lr, 0.001)
        self.assertEqual(trainer.min_lr, 1e-5)
        self.assertEqual(trainer.max_nonimprove, 1)
        self.assertEqual(trainer.lr_decay, 0.1)

        trainer.load_data(dspath, dspath)
        trainer.run()


if __name__ == '__main__':
    unittest.main()
