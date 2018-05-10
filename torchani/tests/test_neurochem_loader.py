import torch
import numpy
import torchani
import unittest
import logging


class TestNeuroChemLoader(unittest.TestCase):

    def setUp(self, dtype=torchani.default_dtype, device=torchani.default_device):
        self.tolerance = 1e-5
        self.ncaev = torchani.NeuroChemAEV(dtype=dtype, device=device)
        self.logger = logging.getLogger('species')

    def testLoader(self):
        nn = torchani.ModelOnAEV(
            self.ncaev, from_nc=self.ncaev.network_dir)
        for i in range(len(self.ncaev.species)):
            s = self.ncaev.species[i]
            model_X = getattr(nn, 'model_' + s)
            self.logger.info(s)
            for j in range(model_X.layers):
                linear = getattr(model_X, 'layer{}'.format(j))
                ncparams = self.ncaev.nc.getntwkparams(i, j)
                ncw = ncparams['weights']
                ncw = torch.from_numpy(ncw).type(
                    self.ncaev.dtype).to(self.ncaev.device)
                ncb = numpy.transpose(ncparams['biases'])
                ncb = torch.from_numpy(ncb).type(
                    self.ncaev.dtype).to(self.ncaev.device)
                max_wdiff = torch.max(
                    torch.abs(ncw - linear.weight.data)).item()
                max_bdiff = torch.max(torch.abs(ncb - linear.bias.data)).item()
                self.assertEqual(max_bdiff, 0.0)
                self.assertEqual(max_wdiff, 0.0)


if __name__ == '__main__':
    unittest.main()
