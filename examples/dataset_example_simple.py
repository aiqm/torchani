"""
Basic usage of the ANIDataset class
========================================================

This supersedes the obsolete anidataloader. There are also builtin datasets
that live in moria, and they can be directly downloaded through torchani.
"""
import shutil
from pathlib import Path
from torchani.datasets import ANIDataset

###############################################################################
# Downloading the builtin datasets performs a checksum to make sure the files
# are correct. If the function is called again and the dataset is already on
# the path, only the checksum is performed, the data is not downloaded. The
# output is an ANIDataset class
# Uncomment the following code to download (watch out, it may take some time):

# import torchani  # noqa
# ds_1x = torchani.datasets.ANI1x('./datasets/ani1x/', download=True)
# ds_comp6 = torchani.datasets.COMP6v1('./datasets/comp6v1/', download=True)
# ds_2x = torchani.datasets.ANI2x('./datasets/ani2x/', download=True)

###############################################################################
# For the purposes of this example we will copy and modify two files inside
# torchani/dataset, which can be downloaded by running the download.sh script
file1_path = Path.cwd() / 'file1.h5'
file2_path = Path.cwd() / 'file2.h5'
shutil.copy(Path.cwd() / '../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5', file1_path)
shutil.copy(Path.cwd() / '../dataset/ani1-up_to_gdb4/ani_gdb_s02.h5', file2_path)

###############################################################################
# ANIDataset accepts a path to an h5 file or a list of paths to many files
# (optionally with names)
ds = ANIDataset(locations=(file1_path, file2_path), names=('file1', 'file2'))

###############################################################################
# ANIDatasets have properties they can access. All conformers in the dataset
# have the same set of properties, lets check what properties this dataset
# holds
print(ds.properties)

###############################################################################
# When opening these files we see that we get a warning because they have some
# unsupported legacy properties, so the first thing we will do is delete them
ds.delete_properties(('coordinatesHE', 'energiesHE', 'smiles'))
print(ds.properties)

###############################################################################
# Conformer groups
# ----------------
#
# To access groups of conformers we can just use the dataset as an ordered
# dictionary
group = ds['file2/gdb11_s02/gdb11_s02-8']
print(group)

###############################################################################
# We see that we get some tensors with properties,
# but this access is not very convenient, the keys seem to have weird
# mangled names which don't say very much about what is in them.
print(list(ds.keys()))

###############################################################################
# This is because this dataset is in a legacy format, we can check that
# by querying the "grouping"
print(ds.grouping)

###############################################################################
# Before moving on, lets reformat this dataset so that it is in a more
# standarized format
ds.regroup_by_formula()
print(list(ds.keys()))

###############################################################################
# Now the dataset is organized by formulas, which makes access much easier
# (If we only had one file ds['CH4'] would have been enough)
group = ds['file1/CH4']

###############################################################################
# items(), values() and keys() work as expected for groups of conformers,
# here we print only the first 100 as a sample
for j, (k, v) in enumerate(ds.items()):
    print(k, v)
    if j == 10:
        break

for j, k in enumerate(ds.keys()):
    print(k)
    if j == 10:
        break

for j, v in enumerate(ds.values()):
    print(v)
    if j == 10:
        break

###############################################################################
# To get the number of groups of conformers we can use len(), or also
# dataset.num_conformer_groups
num_groups = len(ds)
print(num_groups)

###############################################################################
# To get the number of conformers we can use num_conformers
num_conformers = ds.num_conformers
print(num_conformers)

###############################################################################
# Conformers
# ----------
#
# To access individual conformers or subsets of conformers we use "conformer"
# methods, get_conformers and iter_conformers
conformer = ds.get_conformers('file1/CH4', 0)
print(conformer)
conformer = ds.get_conformers('file1/CH4', 1)
print(conformer)

###############################################################################
# A tensor / list / array can also be passed for indexing, to fetch multiple
# conformers from the same group, which is faster. Since we copy the data forh
# simplicity, this allows all fancy indexing operations (directly indexing
# using h5py for example does not).
conformers = ds.get_conformers('file1/CH4', [0, 1])
print(conformers)

###############################################################################
# We can also access all the group if we don't pass an index, same as normal indexing
conformer = ds.get_conformers('file1/CH4')
print(conformer)

###############################################################################
# Finally, it is possible to also specify which properties we want using 'properties'
conformer = ds.get_conformers('file1/CH4', [0, 3], properties=('species', 'energies'))
print(conformer)

###############################################################################
# If you want you can also get the conformers as numpy arrays by calling
# get_numpy_conformers.  this has an optional flag "chem_symbols" which if
# specified "True" will output the elements as strings ('C', 'H', 'H', ... etc)
conformer = ds.get_numpy_conformers('file1/CH4', [0, 1], chem_symbols=True)
print(conformer)

###############################################################################
# We can iterate over all conformers sequentially by calling iter_conformer,
# (this is faster than doing it manually since it caches each conformer group
# previous to starting the iteration), here we print the first 100 as a sample
for j, c in enumerate(ds.iter_conformers()):
    print(c)
    if j == 100:
        break

###############################################################################
# We will now delete the files we copied for cleanup purposes
file1_path.unlink()
file2_path.unlink()
