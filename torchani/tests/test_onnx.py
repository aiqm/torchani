import torchani
import unittest
import torch
import cntk
import tempfile
import os

class TestONNX(unittest.TestCase):

    def _testONNX(self):  # not ready yet

        # molecule structure: CH2OH
        self.species = [ 'C', 'H', 'H', 'O', 'H' ]
        self.coordinates = [
            [0,0,0],  # C
            [0,0,1],  # H
            [1,0,0],  # H
            [0,1,0],  # O
            [0,1,1],  # H
        ]

        ####################################################
        # Step 1: use pytorch to export all graphs into ONNX
        ####################################################

        tmpdir = tempfile.TemporaryDirectory()
        tmpdirname = tmpdir.name
        radial_onnx = os.path.join(tmpdirname, 'radial.proto')

        # export graph for radial AEV
        dummy_center = torch.FloatTensor([[0,0,0]])
        dummy_neighbors = torch.FloatTensor([[[0,0,1],[0,1,0]]])
        aev_computer = torchani.NeighborAEV()
        class RadialAEV(torchani.NeighborAEV):
            def forward(self, center, neighbors):
                return self.radial_subaev(center, neighbors)
        torch.onnx.export(RadialAEV(), (dummy_center, dummy_neighbors), radial_onnx)

        #####################################
        # Step 2: import from ONNX using CNTK
        #####################################


        ###################################
        # Step 3: compute energy using CNTX
        ###################################


        ##########################################
        # Test only: check the result with pytorch
        ##########################################
        # compute energy using pytorch
        coordinates = torch.FloatTensor(self.coordinates)
        coordinates = coordinates.unsqueeze(0)
        model = torchani.ModelOnAEV(aev_computer, from_pync=None)
        energy_shifter = torchani.EnergyShifter()
        energy2 = model(coordinates, self.species)
        energy2 = energy_shifter.add_sae(energy2, self.species)
        energy2 = energy2.squeeze().item()
        # assert that 
        print(energy2)

if __name__ == '__main__':
    unittest.main()