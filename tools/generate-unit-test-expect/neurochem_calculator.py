import os
import ase
import pyNeuroChem
import ase_interface
import numpy

path = os.path.dirname(os.path.realpath(__file__))
builtin_path = os.path.join(path, '../../torchani/resources/ani-1x_8x/')
const_file = os.path.join(builtin_path, 'rHCNO-5.2R_16-3.5A_a4-8.params')
sae_file = os.path.join(builtin_path, 'sae_linfit.dat')
network_dir = os.path.join(builtin_path, 'train0/networks/')
radial_length = 64
conv_au_ev = 27.21138505
all_species = ['H', 'C', 'N', 'O']
species_indices = {all_species[i]: i for i in range(len(all_species))}
nc = pyNeuroChem.molecule(const_file, sae_file, network_dir, 0)


def calc():
    calc = ase_interface.ANI(False)
    calc.setnc(nc)
    return calc


class NeuroChem:

    def _get_radial_part(self, fullaev):
        if len(fullaev.shape) == 3:
            return fullaev[:, :, :radial_length]
        assert len(fullaev.shape) == 2
        return fullaev[:, :radial_length]

    def _get_angular_part(self, fullaev):
        if len(fullaev.shape) == 3:
            return fullaev[:, :, radial_length:]
        assert len(fullaev.shape) == 2
        return fullaev[:, radial_length:]

    def _per_conformation(self, coordinates, species):
        atoms = coordinates.shape[0]
        mol = ase.Atoms(''.join(species), positions=coordinates)
        mol.calc = calc()
        energy = mol.get_potential_energy() / conv_au_ev
        aevs = [mol.calc.nc.atomicenvironments(j) for j in range(atoms)]
        force = mol.get_forces() / conv_au_ev
        aevs = numpy.stack(aevs)
        return aevs, energy, force

    def __call__(self, *args):
        if len(args) == 2:
            return self.from_coordinates_and_species(*args)
        return self.from_atoms_obj(args)

    def from_atoms_obj(self, mol):
        natoms = len(mol)
        mol = mol.copy()
        mol.calc = calc()
        energy = mol.get_potential_energy() / conv_au_ev
        aevs = [mol.calc.nc.atomicenvironments(j) for j in range(natoms)]
        aevs = numpy.stack(aevs)
        force = mol.get_forces() / conv_au_ev
        cell = mol.get_cell(complete=True)
        pbc = mol.get_pbc().astype(numpy.uint8)
        coordinates = mol.get_positions()
        species = numpy.array([species_indices[i] for i in mol.get_chemical_symbols()])
        return coordinates, species, self._get_radial_part(aevs), \
            self._get_angular_part(aevs), energy, force, cell, pbc

    def from_coordinates_and_species(self, coordinates, species):
        if len(coordinates.shape) == 2:
            coordinates = coordinates.reshape(1, -1, 3)
        conformations = coordinates.shape[0]
        results = [self._per_conformation(
            coordinates[i], species) for i in range(conformations)]
        aevs, energies, forces = zip(*results)
        aevs = numpy.stack(aevs)
        energies = numpy.stack(energies)
        forces = numpy.stack(forces)
        species = numpy.array([species_indices[i] for i in species])
        species = species.reshape(1, -1)
        species = numpy.broadcast_to(species,
                                     (coordinates.shape[0], species.shape[1]))
        return coordinates, species, self._get_radial_part(aevs), \
            self._get_angular_part(aevs), energies, forces
