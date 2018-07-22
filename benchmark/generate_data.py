from ase import Atoms
from ase.calculators.tip3p import TIP3P, rOH, angleHOH
from ase.md import Langevin
import ase.units as units
import numpy
import h5py
from rdkit import Chem
from rdkit.Chem import AllChem
# from asap3 import EMT
from ase.calculators.emt import EMT
from multiprocessing import Pool
from tqdm import tqdm, trange
from selected_system import mols, mol_file

conformations = 1024
T = 30
tqdm.monitor_interval = 0

fw = h5py.File("waters.hdf5", "w")
fm = h5py.File(mol_file, "w")


def save(h5file, name, species, coordinates):
    h5file[name] = coordinates
    h5file[name].attrs['species'] = ' '.join(species)


def waterbox(x, y, z, tqdmpos):
    name = '{}_waters'.format(x*y*z)
    # Set up water box at 20 deg C density
    a = angleHOH * numpy.pi / 180 / 2
    pos = [[0, 0, 0],
           [0, rOH * numpy.cos(a), rOH * numpy.sin(a)],
           [0, rOH * numpy.cos(a), -rOH * numpy.sin(a)]]
    atoms = Atoms('OH2', positions=pos)

    vol = ((18.01528 / 6.022140857e23) / (0.9982 / 1e24))**(1 / 3.)
    atoms.set_cell((vol, vol, vol))
    atoms.center()

    atoms = atoms.repeat((x, y, z))
    atoms.set_pbc(False)
    species = atoms.get_chemical_symbols()

    atoms.calc = TIP3P()
    md = Langevin(atoms, 1 * units.fs, temperature=T *
                  units.kB, friction=0.01)

    def generator(n):
        for _ in trange(n, desc=name, position=tqdmpos):
            md.run(1)
            positions = atoms.get_positions()
            yield positions

    save(fw, name, species, numpy.stack(generator(conformations)))


def compute(smiles):
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, useRandomCoords=True)
    AllChem.UFFOptimizeMolecule(m)
    pos = m.GetConformer().GetPositions()
    natoms = m.GetNumAtoms()
    species = [m.GetAtomWithIdx(j).GetSymbol() for j in range(natoms)]

    atoms = Atoms(species, positions=pos)

    atoms.calc = EMT()
    md = Langevin(atoms, 1 * units.fs, temperature=T *
                  units.kB, friction=0.01)

    def generator(n):
        for _ in range(n):
            md.run(1)
            positions = atoms.get_positions()
            yield positions

    c = numpy.stack(generator(conformations))
    return smiles.replace('/', '_'), species, c


def molecules():
    smiles = [s for atoms in mols for s in mols[atoms]]
    with Pool() as p:
        return p.map(compute, smiles)


if __name__ == '__main__':
    for i in molecules():
        save(fm, *i)
    print(list(fm.keys()))
    print('done with molecules')

    with Pool() as p:
        p.starmap(waterbox, [(10, 10, 10, 0), (20, 20, 10, 1),
                             (30, 30, 30, 2), (40, 40, 40, 3)])
        print(list(fw.keys()))
        print('done with water boxes')
