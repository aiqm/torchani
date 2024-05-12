import argparse

from ase import Atoms
from ase.md import Langevin
import ase.units as units
from ase.calculators.emt import EMT
from rdkit import Chem
from rdkit.Chem import AllChem


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("smiles")
parser.add_argument("--conformations", type=int, default=2048)
parser.add_argument("--temperature", type=int, default=30)
args = parser.parse_args()


def save(smiles, species, coordinates):
    print(len(species))
    print(smiles)
    for s, c in zip(species, coordinates):
        print(s, *c)


smiles = args.smiles
m = Chem.MolFromSmiles(smiles)
m = Chem.AddHs(m)
AllChem.EmbedMolecule(m, useRandomCoords=True)
AllChem.UFFOptimizeMolecule(m)
pos = m.GetConformer().GetPositions()
natoms = m.GetNumAtoms()
species = [m.GetAtomWithIdx(j).GetSymbol() for j in range(natoms)]

atoms = Atoms(species, positions=pos)

atoms.calc = EMT()
md = Langevin(
    atoms, 1 * units.fs, temperature=args.temperature * units.kB, friction=0.01
)

for _ in range(args.conformations):
    md.run(1)
    positions = atoms.get_positions()
    save(smiles, species, positions)
