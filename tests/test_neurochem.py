import torchani
import torch
import os
import unittest


path = os.path.dirname(os.path.realpath(__file__))
iptpath = os.path.join(path, 'test_data/inputtrain.ipt')
dspath = os.path.join(path, '../dataset/ani_gdb_s01.h5')


class TestNeuroChem(unittest.TestCase):

    def testNeuroChemTrainer(self):
        d = torch.device('cpu')
        trainer = torchani.neurochem.Trainer(iptpath, d, True, 'runs')
        trainer.load_data(dspath, dspath)
        trainer.run()


if __name__ == '__main__':
    unittest.main()
