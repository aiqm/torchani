import os
import ase
import pyNeuroChem
import ase_interface
import numpy
import pickle
import pyanitools

path = os.path.dirname(os.path.realpath(__file__))
builtin_path = os.path.join(path, '../torchani/resources/ani-1x_dft_x8ens/')
const_file=os.path.join(builtin_path, 'rHCNO-5.2R_16-3.5A_a4-8.params')
sae_file=os.path.join(builtin_path, 'sae_linfit.dat')
network_dir=os.path.join(builtin_path, 'train0/networks/')
radial_length = 64
conv_au_ev = 27.21138505


class NeuroChem:

    def __init__(self):
        self.nc = pyNeuroChem.molecule(const_file, sae_file, network_dir, 0)

    def _get_radial_part(self, fullaev):
        return fullaev[:, :, :radial_length]

    def _get_angular_part(self, fullaev):
        return fullaev[:, :, radial_length:]

    def _per_conformation(self, coordinates, species):
        atoms = coordinates.shape[0]
        mol = ase.Atoms(''.join(species), positions=coordinates)
        mol.set_calculator(ase_interface.ANI(False))
        mol.calc.setnc(self.nc)
        energy = mol.get_potential_energy() / conv_au_ev
        aevs = [mol.calc.nc.atomicenvironments(j) for j in range(atoms)]
        force = mol.get_forces() / conv_au_ev
        aevs = numpy.stack(aevs)
        return aevs, energy, force

    def __call__(self, coordinates, species):
        conformations = coordinates.shape[0]
        results = [self._per_conformation(
            coordinates[i], species) for i in range(conformations)]
        aevs, energies, forces = zip(*results)
        aevs = numpy.stack(aevs)
        energies = numpy.stack(energies)
        forces = numpy.stack(forces)
        return self._get_radial_part(aevs), \
            self._get_angular_part(aevs), \
            energies, forces


neurochem = NeuroChem()
mol_count = 0
all_species = ['H', 'C', 'N', 'O']

species_indices = {all_species[i]: i for i in range(len(all_species))}
for i in [1, 2, 3, 4]:
    data_file = os.path.join(
        path, '../dataset/ani_gdb_s0{}.h5'.format(i))
    adl = pyanitools.anidataloader(data_file)
    for data in adl:
        coordinates = data['coordinates'][:10, :]
        species = numpy.array([species_indices[i] for i in data['species']])
        species = species.reshape(1, -1)
        species = numpy.broadcast_to(species, (10, species.shape[1]))
        smiles = ''.join(data['smiles'])
        radial, angular, energies, forces = neurochem(coordinates, data['species'])
        pickleobj = (coordinates, species, radial, angular, energies, forces)
        dumpfile = os.path.join(
            path, '../tests/test_data/ANI1_subset/{}'.format(mol_count))
        with open(dumpfile, 'wb') as f:
            pickle.dump(pickleobj, f)
        mol_count += 1
