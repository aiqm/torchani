from ase import Atoms
from ase.calculators.tip3p import TIP3P, rOH, angleHOH
from ase.md import Langevin
import ase.units as units
from ase.io.trajectory import Trajectory
import numpy
import h5py
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.calculators.emt import EMT

conformations = 2

def waterbox(x, y, z):
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
    md = Langevin(atoms, 1 * units.fs, temperature=300 *
                  units.kB, friction=0.01)

    def generator(n):
        for _ in range(n):
            md.run(1)
            positions = atoms.get_positions()
            yield positions

    return name, species, numpy.stack(generator(conformations))


def molecules():
    mols = {
        '20': [
            'COC(=O)c1ccc([N+](=O)[O-])cc1',
            'O=c1nnc2ccccc2n1CO',
            'CCc1ccc([N+](=O)[O-])cc1',
            'Nc1ccc(c2cnco2)cc1',
            'COc1ccc(N)c(N)c1',
            'O=C(O)CNc1ccccc1',
            'NC(=O)NNc1ccccc1',
            'Cn1c(=O)oc(=O)c2ccccc12',
            'CC(=O)Nc1ccc(O)cc1',
            'COc1ccc(CC#N)cc1'
        ],
        '50': [
            'O=[N+]([O-])c1ccc(NN=Cc2ccc(C=NNc3ccc([N+](=O)[O-])cc3[N+](=O)[O-])cc2)c([N+](=O)[O-])c1',
            'CCCCCc1nccnc1OCC(C)(C)CC(C)C',
            'CC(C)(C)c1ccc(N(C(=O)c2ccccc2)C(C)(C)C)cc1',
            'CCCCCCCCCCCOC(=O)Nc1ccccc1',
            'CC(=O)NCC(CN1CCCC1)(c1ccccc1)c1ccccc1',
            'CCCCCc1cnc(C)c(OCC(C)(C)CCC)n1',
            'CCCCCCCCCCCCN1CCOC(=O)C1',
            'CCCCOc1ccc(C=Nc2ccc(CCCC)cc2)cc1',
            'CC1CC(C)C(=NNC(=O)N)C(C(O)CC2CC(=O)NC(=O)C2)C1',
            'CCCCCOc1ccc(C=Nc2ccc(C(=O)OCC)cc2)cc1'
        ],
        '10': [
            'N#CCC(=O)N',
            'N#CCCO',
            'O=C1NC(=O)C(=O)N1',
            'COCC#N',
            'N#CCNC=O',
            'ON=CC=NO',
            'NCC(=O)O',
            'NC(=O)CO',
            'N#Cc1ccco1',
            'C=CC(=O)N'
        ],
        '4,5,6': [
            'C',
            'C#CC#N',
            'C=C',
            'CC#N',
            'C#CC#C',
            'O=CC#C',
            'C#C'
        ],
        '100': [
            'CC(C)C[C@@H](C(=O)O)NC(=O)C[C@@H]([C@H](CC1CCCCC1)NC(=O)CC[C@@H]([C@H](Cc2ccccc2)NC(=O)OC(C)(C)C)O)O',
            'CC(C)(C)OC(=O)N[C@@H](Cc1ccccc1)[C@@H](CN[C@@H](Cc2ccccc2)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc3ccccc3)C(=O)N)O',
            'CC(C)(C)OC(=O)N[C@@H](Cc1ccccc1)[C@H](CN[C@@H](Cc2ccccc2)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc3ccccc3)C(=O)N)O',
            'CC[C@H](c1ccc(cc1)O)[C@H](c2ccc(cc2)O)C(=O)OCCCCCCCCOC(=O)C(c3ccc(cc3)O)C(CC)c4ccc(cc4)O',
            'CC/C(=C\\CC[C@H](C)C[C@@H](C)CC[C@@H]([C@H](C)C(=O)C[C@H]([C@H](C)[C@@H](C)OC(=O)C[C@H](/C(=C(\\C)/C(=O)O)/C(=O)O)O)O)O)/C=C/C(=O)O',
            'CC[C@H](C)[C@H]1C(=O)NCCCOc2ccc(cc2)C[C@@H](C(=O)N1)NC(=O)[C@@H]3Cc4ccc(cc4)OCCCCC(=O)N[C@H](C(=O)N3)C(C)C',
            'CC(C)(C)CC(C)(C)c1ccc(cc1)OCCOCCOCCOCCOCCOCCOCCOCCOCCO',
            'CCOC(=O)CC[C@H](C[C@@H]1CCNC1=O)NC(=O)[C@H](Cc2ccccc2)NC(=O)[C@H](CCC(=O)OC(C)(C)C)NC(=O)OCc3ccccc3',
            'C[C@]12CC[C@@H]3c4ccc(cc4CC[C@H]3[C@@H]1C[C@@H]([C@@H]2O)CCCCCCCCC(=O)OC[C@@H]5[C@H]([C@H]([C@@H](O5)n6cnc7c6ncnc7N)O)O)O',
            'c1cc(ccc1CCc2c[nH]c3c2C(=O)NC(=N3)N)C(=O)N[C@@H](CCC(=O)N[C@@H](CCC(=O)N[C@@H](CCC(=O)N[C@H](CCC(=O)O)C(=O)O)C(=O)O)C(=O)O)C(=O)O'
        ],
        '305': [
            '[H]/N=C(/N)\\NCCC[C@H](C(=O)N[C@H]([C@@H](C)O)C(=O)N[C@H](Cc1ccc(cc1)O)C(=O)NCCCC[C@@H](C(=O)NCCCC[C@@H](C(=O)NCC(=O)O)NC(=O)[C@H](CCCCNC(=O)[C@@H](Cc2ccc(cc2)O)NC(=O)[C@@H]([C@@H](C)O)NC(=O)[C@@H](CCCN/C(=N\\[H])/N)N)NC(=O)[C@@H](Cc3ccc(cc3)O)NC(=O)[C@@H]([C@@H](C)O)NC(=O)[C@@H](CCCN/C(=N\\[H])/N)N)NC(=O)[C@@H](Cc4ccc(cc4)O)NC(=O)[C@@H]([C@@H](C)O)NC(=O)[C@@H](CCCN/C(=N\\[H])/N)N)N'
        ]
    }

    for atoms in mols:
        for smiles in mols[atoms]:
            m = Chem.MolFromSmiles(smiles)
            m = Chem.AddHs(m)
            AllChem.EmbedMolecule(m,useRandomCoords=True)
            AllChem.UFFOptimizeMolecule(m)
            pos = m.GetConformer().GetPositions()
            natoms = m.GetNumAtoms()
            species = [m.GetAtomWithIdx(j).GetSymbol() for j in range(natoms)]
            
            atoms = Atoms(species, positions=pos)

            atoms.calc = EMT()
            md = Langevin(atoms, 1 * units.fs, temperature=300 *
                        units.kB, friction=0.01)

            def generator(n):
                for _ in range(n):
                    md.run(1)
                    positions = atoms.get_positions()
                    yield positions

            c = numpy.stack(generator(conformations))
            yield smiles.replace('/','_'), species, c

def save(h5file, name, species, coordinates):
    h5file[name] = coordinates
    h5file[name].attrs['species'] = ' '.join(species)

if __name__ == '__main__':
    with h5py.File("molecules.hdf5", "w") as f:
        for i in molecules():
            save(f, *i)

    with h5py.File("waters.hdf5", "w") as f:
        print(list(f.keys()))
        save(f, *waterbox(10, 10, 10))
        save(f, *waterbox(20, 20, 10))
        save(f, *waterbox(30, 30, 30))
        save(f, *waterbox(40, 40, 40))
