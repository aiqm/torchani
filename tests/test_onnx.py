import torchani
import unittest
import torch
import cntk
import tempfile
import os
import numpy


class TestONNX(unittest.TestCase):

    def setUp(self):
        self.tolerance = 1e-5

    def testONNX(self):  # not ready yet
        return

        # molecule structure: CH2OH
        species = ['C', 'H', 'H', 'O', 'H']
        coordinates = [
            [0, 0, 0],  # C
            [0, 0, 1],  # H
            [1, 0, 0],  # H
            [0, 1, 0],  # O
            [0, 1, 1],  # H
        ]

        # compute aev using pytorch
        aev_computer = torchani.AEV()
        coordinates = torch.FloatTensor(coordinates)
        coordinates = coordinates.unsqueeze(0)
        radial_aev, angular_aev = aev_computer(coordinates, species)
        aev = torch.cat([radial_aev, angular_aev], dim=2).numpy()

        # temp directory storing exported networks
        tmpdir = tempfile.TemporaryDirectory()
        tmpdirname = tmpdir.name

        ####################################################
        # Step 1: use pytorch to export all graphs into ONNX
        ####################################################

        # TODO: exporting AEV to ONNX is not supported yet,
        # due to lack of operators in ONNX. Add this support
        # when ONNX support this operation.

        aev_computer.export_radial_subaev_onnx(
            os.path.join(tmpdirname, 'radial.onnx'))

        # Export neural network potential to ONNX
        model = torchani.ModelOnAEV(aev_computer, from_nc=None)
        model.export_onnx(tmpdirname)

        #####################################
        # Step 2: import from ONNX using CNTK
        #####################################
        networks = {}
        for s in aev_computer.species:
            nn_onnx = os.path.join(tmpdirname, '{}.proto'.format(s))
            networks[s] = cntk.Function.load(
                nn_onnx, format=cntk.ModelFormat.ONNX)

        ###################################
        # Step 3: compute energy using CNTX
        ###################################
        energy1 = 0
        for i in range(len(species)):
            atomic_aev = aev[:, i, :]
            network = networks[species[i]]
            atomic_energy = network(atomic_aev)[0, 0, 0]
            energy1 += atomic_energy

        ###############################################
        # Test only: check the CNTK result with pytorch
        ###############################################
        energy2 = model(coordinates, species).squeeze().item()
        self.assertLessEqual(abs(energy1 - energy2), self.tolerance)


if __name__ == '__main__':
    unittest.main()
