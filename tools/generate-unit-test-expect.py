import os
import pickle
import pyanitools
import numpy
from neurochem_calculator import NeuroChem, path
import json
import tqdm
import random


neurochem = NeuroChem()

# generate expect for ANI1 subset
mol_count = 0
for i in [1, 2, 3, 4]:
    data_file = os.path.join(
        path, '../dataset/ani_gdb_s0{}.h5'.format(i))
    adl = pyanitools.anidataloader(data_file)
    for data in tqdm.tqdm(adl, desc='ANI1: {} heavy atoms'.format(i)):
        coordinates = data['coordinates'][:10, :]
        pickleobj = neurochem(coordinates, data['species'])
        dumpfile = os.path.join(
            path, '../tests/test_data/ANI1_subset/{}'.format(mol_count))
        with open(dumpfile, 'wb') as f:
            pickle.dump(pickleobj, f)
        mol_count += 1


# generate expect for NIST
keep_ratio = 0.1  # reduce the size of generated file by discarding
mol_count = 0
with open(os.path.join(path, 'diverse_test_set/result.json')) as f:
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
        path, '../tests/test_data/NIST/all')
    with open(dumpfile, 'wb') as f:
        pickle.dump(pickle_objects, f)
