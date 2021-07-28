import ase
import ase.io
import ase.optimize
import ase.md.velocitydistribution
import ase.md.verlet
import os
from neurochem_calculator import NeuroChem, path
import torchani
import pickle


neurochem = NeuroChem()

molecule = ase.io.read(os.path.join(path, 'tripeptide/tripeptide-000.ipt_optimization.xyz'))

temp = 300 * ase.units.kB
stepsize = 0.25 * ase.units.fs
steps = int(10000 * ase.units.fs / stepsize)

ase.md.velocitydistribution.MaxwellBoltzmannDistribution(molecule, temp, force_temp=True)
ase.md.velocitydistribution.Stationary(molecule)
ase.md.velocitydistribution.ZeroRotation(molecule)

print("Initial temperature from velocities %.2f" % molecule.get_temperature())

molecule.calc = torchani.models.ANI1ccx().to('cuda').ase()

dyn = ase.md.verlet.VelocityVerlet(
    molecule,
    stepsize,
    logfile='-',
)

counter = 0
data_dir = os.path.join(path, '../../tests/test_data/tripeptide-md/')


def dump_neurochem_data(molecule=molecule):
    global counter
    filename = os.path.join(data_dir, '{}.dat'.format(counter))
    ret = neurochem(molecule)
    with open(filename, 'wb') as f:
        pickle.dump(ret, f)
    counter += 1


dyn.attach(dump_neurochem_data, interval=400)
dyn.run(steps)
