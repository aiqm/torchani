import os
import pickle
import pyanitools
from neurochem_calculator import NeuroChem, path
import tqdm


neurochem = NeuroChem()

# generate expect for ANI1 subset
mol_count = 0
for i in [1, 2, 3, 4]:
    data_file = os.path.join(
        path, '../../dataset/ani1-up_to_gdb4/ani_gdb_s0{}.h5'.format(i))
    adl = pyanitools.anidataloader(data_file)
    for data in tqdm.tqdm(adl, desc='ANI1: {} heavy atoms'.format(i)):
        coordinates = data['coordinates'][:10, :]
        pickleobj = neurochem(coordinates, data['species'])
        dumpfile = os.path.join(
            path, '../../tests/resources/ANI1_subset/{}'.format(mol_count))
        with open(dumpfile, 'wb') as f:
            pickle.dump(pickleobj, f)
        mol_count += 1
