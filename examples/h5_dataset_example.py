from torchani.datasets import AniH5Dataset
import numpy as np

# Example usage of the AniH5Dataset class, which supersedes the obsolete
# anidataloader

dataset = AniH5Dataset('/home/ignacio/Datasets/ani1x_release_wb97x_dz.h5')

# ############## Conformer groups:  ###########################
# To access groups of conformers we can just use the dataset as a dictionary
group = dataset['C10H10']
print(group)

# items(), values() and keys() work as expected for groups of conformers
for k, v in dataset.items():
    print(k, v)

for k in dataset.keys():
    print(k)

for v in dataset.values():
    print(v)

# To get the number of groups of conformers we can use len(), or also
# dataset.num_conformer_groups
num_groups = len(dataset)
print(num_groups)

# ############## Conformers:  ###########################
# To access individual conformers or subsets of conformers we use *_conformer
# methods, get_conformers and iter_conformers
conformer = dataset.get_conformers('C10H10', 0)
print(conformer)
conformer = dataset.get_conformers('C10H10', 1)
print(conformer)

# A numpy array can also be passed for indexing, to fetch multiple conformers
# from the same group, which is faster.
# Since I copy the data for simplicity, this allows all of numpy fancy indexing
# operations (directly indexing using h5py does not).
conformers = dataset.get_conformers('C10H10', np.array([0, 1]))
print(conformers)

# We can also access all the group, same as with [] if we don't pass an index
conformer = dataset.get_conformers('C10H10')
print(conformer)

# Finally, it is possible to specify which properties we want using 'include_properties'
conformer = dataset.get_conformers('C10H10', include_properties=('species', 'energies'))
print(conformer)

conformer = dataset.get_conformers('C10H10', np.array([0, 3]), include_properties=('species', 'energies'))
print(conformer)

# We can iterate over all conformers sequentially by calling iter_conformer,
# (this is faster than doing it manually since it caches each conformer group
# previous to starting the iteration)
for c in dataset.iter_conformers():
    print(c)

# To get the number of conformers we can use num_conformers
num_conformers = dataset.num_conformers
print(num_conformers)
