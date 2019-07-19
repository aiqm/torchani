import os
import pickle
import json
import tqdm
import random
import ase
from ase.optimize import BFGS
from neurochem_calculator import calc, path

keep_ratio = 0.01  # reduce the size of generated file by discarding
mol_count = 0
with open(os.path.join(path, 'nist-dataset/result.json')) as f:
    pickle_objects = []
    for i in tqdm.tqdm(json.load(f), desc='Optim'):
        if random.random() > keep_ratio:
            continue
        atoms = i['atoms']
        natoms = len(atoms)
        species = []
        coordinates = []
        for atype, x, y, z in atoms:
            species.append(atype)
            coordinates.append([x, y, z])
        mol = ase.Atoms(species, positions=coordinates)
        mol.set_calculator(calc())
        opt = BFGS(mol, logfile='/dev/null')
        opt.run()
        mol.set_calculator(None)
        pickle_objects.append(mol)
        mol_count += 1

    dumpfile = os.path.join(
        path, '../../tests/test_data/NeuroChemOptimized/all')
    with open(dumpfile, 'wb') as f:
        pickle.dump(pickle_objects, f)
