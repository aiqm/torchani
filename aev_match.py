import ase
import ase.io
import pyNeuroChem
import ase_interface
import numpy as np
import os
import torch

builtin_path = '/data/gaoxiang/torchani/torchani/resources/ani-1x_8x'
const_file = os.path.join(builtin_path, 'rHCNO-5.2R_16-3.5A_a4-8.params')
sae_file = os.path.join(builtin_path, 'sae_linfit.dat')
network_dir = os.path.join(builtin_path, 'train0/networks/')

nc = pyNeuroChem.molecule(const_file, sae_file, network_dir, 0)
calc = ase_interface.ANI(False)
calc.setnc(nc)
molecule = ase.io.read('/data/gaoxiang/torchani/tools/generate-unit-test-expect/others/Benzene.pdb')
na = len(molecule)
molecule.set_calculator(calc)

positions = molecule.get_positions(wrap=True)
cell = molecule.get_cell(complete=True)
cs = molecule.get_chemical_symbols()

nc2 = pyNeuroChem.molecule(const_file, sae_file, network_dir, 0)
calc2 = ase_interface.ANI(False)
calc2.setnc(nc2)

positions2 = positions + cell[0]
positions2 = np.concatenate([positions, positions2])
cell2 = cell.copy()
cell2[0, :] *= 2
cs2 = cs + cs
molecule2 = ase.Atoms(cs2, positions2, cell=cell2, pbc=True)
molecule2.set_calculator(calc2)

for i in range(na):
    print('-' * 48)
    e = molecule.get_potential_energy()
    aev = molecule.calc.nc.atomicenvironments(i)
    e2 = molecule2.get_potential_energy()
    print(e2 / e / 2)
    aev2_1 = molecule2.calc.nc.atomicenvironments(i)
    aev2_2 = molecule2.calc.nc.atomicenvironments(na + i)
    print(torch.tensor(aev - aev2_1).abs().max().item())
    print(torch.tensor(aev - aev2_2).abs().max().item())
