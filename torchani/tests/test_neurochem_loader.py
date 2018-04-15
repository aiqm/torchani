import torch
import numpy
import torchani
import unittest
import logging

class TestNeuroChemLoader(unittest.TestCase):

    def setUp(self, dtype=torch.cuda.float32):
        self.tolerance = 1e-5
        self.ncaev = torchani.NeuroChemAEV()
        self.logger = logging.getLogger('species')

    def testLoader(self):
        nn = torchani.NeuralNetworkOnAEV(self.ncaev, from_pync=self.ncaev.network_dir)
        for i in range(len(self.ncaev.species)):
            s = self.ncaev.species[i]
            self.logger.info(s)
            for j in range(nn.layers[s]):
                linear = getattr(nn, '{}{}'.format(s, j))
                ncparams = self.ncaev.nc.getntwkparams(i,j)
                ncw = ncparams['weights']
                ncw = torch.from_numpy(ncw).type(self.ncaev.dtype)
                ncb = numpy.transpose(ncparams['biases'])
                ncb = torch.from_numpy(ncb).type(self.ncaev.dtype)
                max_wdiff = torch.max(torch.abs(ncw - linear.weight.data)).item()
                max_bdiff = torch.max(torch.abs(ncb - linear.bias.data)).item()
                self.assertEqual(max_bdiff, 0.0)
                self.assertEqual(max_wdiff, 0.0)

if __name__ == '__main__':
    unittest.main()