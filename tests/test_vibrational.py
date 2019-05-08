import os
import math
import unittest
import torch
import torchani
import ase
import ase.optimize
import ase.vibrations


path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '../dataset/xyz_files/H2O.xyz')


class TestVibrational(unittest.TestCase):

    def testVibrationalWavenumbers(self):
        model = torchani.models.ANI1x().double()
        d = 0.9575
        t = math.pi / 180 * 104.51
        molecule = ase.Atoms('H2O', positions=[
            (d, 0, 0),
            (d * math.cos(t), d * math.sin(t), 0),
            (0, 0, 0),
        ], calculator=model.ase())
        opt = ase.optimize.BFGS(molecule)
        opt.run(fmax=1e-6)
        masses = torch.tensor([1.008, 12.011, 14.007, 15.999], dtype=torch.double)
        # compute vibrational frequencies by ASE
        vib = ase.vibrations.Vibrations(molecule)
        vib.run()
        freq = vib.get_frequencies()
        print(freq)
        # compute vibrational by torchani
        species = model.species_to_tensor(molecule.get_chemical_symbols()).unsqueeze(0)
        coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).requires_grad_(True)
        _, energies = model((species, coordinates))
        hessian = torchani.utils.hessian(coordinates, energies=energies)
        freq2 = torchani.utils.vibrational_analysis(masses[species], hessian)
        print(freq2)


if __name__ == '__main__':
    unittest.main()
