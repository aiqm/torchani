import os
import pickle
import numpy
from neurochem_calculator import NeuroChem, path
import json
import tqdm
import random


neurochem = NeuroChem()
keep_ratio = 0.1  # reduce the size of generated file by discarding
mol_count = 0
with open(os.path.join(path, 'nist-dataset/result.json')) as f:
    pickle_objects = []
    for i in tqdm.tqdm(json.load(f), desc='NIST'):
        if random.random() > keep_ratio:
            continue
        atoms = i['atoms']
        natoms = len(atoms)
        species = []
        coordinates = []
        for atype, x, y, z in atoms:
            species.append(atype)
            coordinates.append([x, y, z])
        pickleobj = neurochem(numpy.array(coordinates), species)
        pickle_objects.append(pickleobj)
        mol_count += 1

    dumpfile = os.path.join(
        path, '../../tests/test_data/NIST/all')
    with open(dumpfile, 'wb') as bf:
        pickle.dump(pickle_objects, bf)
