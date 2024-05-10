import os
import math
import unittest
import torch
import torchani
import ase
import ase.optimize
import ase.vibrations
import numpy
from torchani.testing import TestCase


path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '../dataset/xyz_files/H2O.xyz')


class TestVibrational(TestCase):

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
        # compute vibrational frequencies by ASE
        vib = ase.vibrations.Vibrations(molecule)
        vib.run()
        freq = torch.tensor([numpy.real(x) for x in vib.get_frequencies()[6:]])
        modes = []
        for j in range(6, 6 + len(freq)):
            modes.append(numpy.expand_dims(vib.get_mode(j), axis=0))
        vib.clean()
        modes = torch.tensor(numpy.concatenate(modes, axis=0))
        # compute vibrational by torchani
        species = torch.tensor(molecule.get_atomic_numbers()).unsqueeze(0)
        masses = torchani.utils.get_atomic_masses(species, dtype=torch.double)
        coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).requires_grad_(True)
        _, energies = model((species, coordinates))
        hessian = torchani.utils.hessian(coordinates, energies=energies)
        freq2, modes2, _, _ = torchani.utils.vibrational_analysis(masses, hessian)
        freq2 = freq2[6:].float()
        modes2 = modes2[6:]
        self.assertEqual(freq, freq2, atol=0, rtol=0.02, exact_dtype=False)

        diff1 = (modes - modes2).abs().max(dim=-1).values.max(dim=-1).values
        diff2 = (modes + modes2).abs().max(dim=-1).values.max(dim=-1).values
        diff = torch.where(diff1 < diff2, diff1, diff2)
        self.assertLess(diff.max(), 0.02)


if __name__ == '__main__':
    unittest.main()
