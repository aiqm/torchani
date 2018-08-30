from ase import Atoms
from ase.md import Langevin
import ase.units as units
from rdkit import Chem
from rdkit.Chem import AllChem
# from asap3 import EMT
from ase.calculators.emt import EMT
import argparse


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('smiles')
parser.add_argument('--conformations', type=int, default=2048)
parser.add_argument('--temperature', type=int, default=30)
parser = parser.parse_args()


def save(smiles, species, coordinates):
    print(len(species))
    print(smiles)
    for s, c in zip(species, coordinates):
        print(s, *c)


smiles = parser.smiles
m = Chem.MolFromSmiles(smiles)
m = Chem.AddHs(m)
AllChem.EmbedMolecule(m, useRandomCoords=True)
AllChem.UFFOptimizeMolecule(m)
pos = m.GetConformer().GetPositions()
natoms = m.GetNumAtoms()
species = [m.GetAtomWithIdx(j).GetSymbol() for j in range(natoms)]

atoms = Atoms(species, positions=pos)

atoms.calc = EMT()
md = Langevin(atoms, 1 * units.fs, temperature=parser.temperature * units.kB,
              friction=0.01)

for _ in range(parser.conformations):
    md.run(1)
    positions = atoms.get_positions()
    save(smiles, species, positions)
