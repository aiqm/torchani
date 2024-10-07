import math
import typing as tp

import ase
import ase.vibrations
from ase.optimize import BFGS
import numpy as np
from numpy.typing import NDArray

import torchani

model = torchani.models.ANI1x().double()
d = 0.9575
t = math.pi / 180 * 104.51
molecule = ase.Atoms(
    "H2O",
    positions=[
        (d, 0, 0),
        (d * math.cos(t), d * math.sin(t), 0),
        (0, 0, 0),
    ],
    calculator=model.ase(),
)
opt = BFGS(molecule)
opt.run(fmax=1e-6)
# compute vibrational frequencies by ASE
vib = ase.vibrations.Vibrations(molecule)
vib.run()

array_freq = np.array([np.real(x) for x in vib.get_frequencies()[6:]])

modes: tp.List[NDArray[np.float64]] = []
for j in range(6, 6 + len(array_freq)):
    modes.append(np.expand_dims(vib.get_mode(j), axis=0))
vib.clean()
array_modes = np.concatenate(modes, axis=0)
species = molecule.get_atomic_numbers()
coordinates = molecule.get_positions()

np.savez(
    "./water-vib-expect.npz",
    modes=array_modes,
    freqs=array_freq,
    coordinates=coordinates,
    species=species,
)
