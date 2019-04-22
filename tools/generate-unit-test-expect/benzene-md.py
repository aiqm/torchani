import ase
import ase.io
import ase.optimize
import ase.md.velocitydistribution
import ase.md.verlet
import torchani
import torch
import numpy as np
import os
from neurochem_calculator import NeuroChem, path


neurochem = NeuroChem()

molecule = ase.io.read('Benzene.cif')

temp = 300 * ase.units.kB
stepsize = ase.units.fs
steps = int(10000 * ase.units.fs / stepsize)

ase.md.velocitydistribution.MaxwellBoltzmannDistribution(molecule, temp, force_temp=True)
ase.md.velocitydistribution.Stationary(molecule)
ase.md.velocitydistribution.ZeroRotation(molecule)

print("Initial temperature from velocities %.2f" % molecule.get_temperature())

molecule.set_calculator(torchani.models.ANI1ccx().to('cuda').ase())

dyn = ase.md.verlet.VelocityVerlet(
    molecule,
    stepsize,
    logfile='-',
)

counter = 0
data_dir = os.path.join(path, '../../tests/test_data/benzene-md/')
def dump_neurochem_data(molecule):
    filename = os.path.join(data_dir, '{}.dat'.format(counter))
    ret = neurochem(molecule)
    with open(filename, 'wb') as f:
        filename.dump(ret, f)

dyn.attach(dump_neurochem_data)
dyn.run(steps)
