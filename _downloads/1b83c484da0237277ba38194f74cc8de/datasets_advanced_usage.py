r"""
Advanced usage of :obj:`~torchani.datasets.ANIDataset`
======================================================

Example showing more involved conformer and property manipulation.
"""
# %%
# To begin with, let's import the modules we will use:
import shutil
from pathlib import Path

import torch
import numpy as np

from torchani.datasets import ANIDataset, concatenate
from torchani.datasets.filters import filter_by_high_force
# %%
# Again for the purposes of this example we will copy and modify two files
# inside torchani/dataset, which can be downloaded by running the download.sh
# script.
file1_path = Path.cwd() / "file1.h5"
file2_path = Path.cwd() / "file2.h5"
shutil.copy(Path.cwd() / "../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5", file1_path)
shutil.copy(Path.cwd() / "../dataset/ani1-up_to_gdb4/ani_gdb_s02.h5", file2_path)
ds = ANIDataset(locations=(file1_path, file2_path), names=("file1", "file2"))
# %%
# Property deletion / renaming
# ----------------------------
#
# All of the molecules in the dataset have the same properties, energies,
# coordinates, etc. You can query which are these.
ds.properties
# %%
# It is possible to delete unwanted / unnedded properties.
ds.delete_properties(("coordinatesHE", "energiesHE", "smiles"))
ds.properties
# %%
# It is also possible to rename the properties by passing a dict of old-new names (the
# class assumes at least one of "species" or "numbers" is always present, so don't
# rename those).
ds.rename_properties({"energies": "molecular_energies", "coordinates": "coord"})
ds.properties
# %%
# Lets rename them back to their original values:
ds.rename_properties({"molecular_energies": "energies", "coord": "coordinates"})
ds.properties
# %%
# Grouping
# --------
#
# You can query whether your dataset is in a legacy format by interrogating the
# dataset grouping attribute
ds.grouping
# %%
# Legacy format is the format used by some old datasets. In the legacy format
# there can be groups arbitrarily nested in the hierarchical tree inside the h5
# files, and the "species"/"numbers" property does not have a batch dimension.
# This means all properties with an "atomic" dimension must be ordered the same
# way within a group (don't worry too much if you don't understand what this
# means, it basically means this is difficult to deal with)
#
# We can convert to a less error prone and easier to parse format by calling
# "regroup_by_formula" or "regroup_by_num_atoms"
ds = ds.regroup_by_formula()
ds.grouping
# %%
# Another possibility is to group by num atoms
ds = ds.regroup_by_num_atoms()
ds.grouping
# %%
# In these formats all of the first dimensions of all properties are the same
# in all groups, and groups can only have depth one. In other words the tree
# structure is, for "by_formula" ::
#
#    /C10H22/coordinates, shape (10, 32, 3)
#           /species, shape (10, 32)
#           /energies, shape (10,)
#    /C8H22N2/coordinates, shape (10, 32, 3)
#           /species, shape (10, 32)
#           /energies, shape (10,)
#    /C12H22/coordinates, shape (5, 34, 3)
#           /species, shape (5, 34)
#           /energies, shape (5,)
#
# and for, "by_num_atoms" ::
#
#    /032/coordinates, shape (20, 32, 3)
#         /species, shape (20, 32)
#         /energies, shape (20,)
#    /034/coordinates, shape (5, 34, 3)
#         /species, shape (5, 34)
#         /energies, shape (5,)
#
# Conformer groups can be iterated over in chunks, up to a specified maximum
# chunk size. This breaks a conformer group into mini-batches containing
# multiple inputs, allowing the dataset to be iterated over much more
# efficiently. As we regrouped the dataset by num_atoms in the previous step,
# this will iterate over conformer groups containing the same number of atoms.
with ds.keep_open("r") as read_ds:
    for group, j, conformer in read_ds.chunked_items(max_size=1500, limit=2):
        species = conformer["species"]
        coordinates = conformer["coordinates"]
        ani_input = (species, coordinates)
        print(ani_input)
# %%
# Property creation
# -----------------
#
# Sometimes it may be useful to just create one placeholder property for some
# purpose. You can make the second dimension equal to the number of atoms in
# the group by setting ``is_atomic=True``, and you can add also extra dims, for
# example, this creates a property with shape ``(N, A)``, for more examples see
# docstring of the function.
ds = ds.create_full_property(
    "new_property", is_atomic=True, fill_value=0.0, dtype=float
)
ds.properties
# %%
# We now delete the created property for cleanup
ds.delete_properties("new_property", verbose=False)
ds.properties
# %%
# Manipulating conformers
# -----------------------
#
# All of the molecules in the dataset have the same properties
# Conformers as tensors can be appended by calling ``append_conformers``.
# Here I put random numbers as species and coordinates but you should put
# something that makes sense, if you have only one store you can pass
# "group_name" directly.
conformers = {
    "species": torch.tensor([[1, 1, 6, 6], [1, 1, 6, 6]]),
    "coordinates": torch.randn(2, 4, 3),
    "energies": torch.randn(2),
}
ds.append_conformers("file1/004", conformers)
# %%
# It is also possible to append conformers as numpy arrays, in this case
# "species" can hold the chemical symbols or atomic numbers. Internally these
# will be converted to atomic numbers.
numpy_conformers = {
    "species": np.array(
        [["H", "H", "C", "N"], ["H", "H", "N", "O"], ["H", "H", "H", "H"]]
    ),
    "coordinates": np.random.standard_normal((3, 4, 3)),
    "energies": np.random.standard_normal(3),
}
ds.append_conformers("file1/004", numpy_conformers)
# %%
# Conformers can also be deleted from the dataset. Passing an index will delete
# a series of conformers, not passing anything deletes the whole group
molecules = ds.get_conformers("file1/004")
molecules
# %%
# Lets delete some conformers and try again
ds.delete_conformers("file1/004", [0, 2])
molecules = ds.get_conformers("file1/004")
# %%
# The len of the dataset has not changed
len(ds)
# %%
# Lets get rid of the whole group
ds.delete_conformers("file1/004")
len(ds)
# %%
# Currently, when appending the class checks:
#
# - That the first dimension of all your properties is the same
# - That you are appending a set of conformers with correct properties
# - That all your formulas are correct when the grouping type is "by_formula",
# - That your group name does not contain illegal "/" characters
# - That you are only appending one of "species" or "numbers"
#
# It does NOT check:
#
# - That the number of atoms is the same in all properties that are atomic
# - That the name of the group is consistent with the formula / num atoms
#
# It is the responsibility of the user to make sure of those items.
#
# Utilities
# ---------
#
# Multiple datasets can be concatenated into one h5 file, optionally deleting the
# original h5 files if the concatenation is successful.
concat_path = Path.cwd() / "concat.h5"
ds = concatenate(ds, concat_path, delete_originals=True)
# %%
# Context manager usage
# ---------------------
#
# If you need to perform a lot of read/write operations in the dataset it can
# be useful to keep all the underlying stores open, you can do this by using a
# ``keep_open`` context.
with ds.keep_open("r+") as open_ds:
    for c in open_ds.iter_conformers(limit=10):
        print(c)
# %%
# Creating a dataset from scratch
# -------------------------------
#
# It is possible to create an ANIDataset from scratch by calling: By defalt the
# grouping is "by_num_atoms". The first set of conformers you append will
# determine what properties this dataset will support.
new_path = Path.cwd() / "new_ds.h5"
new_ds = ANIDataset(new_path, grouping="by_formula")
numpy_conformers = {
    "species": np.array([["H", "H", "C", "C"], ["H", "C", "H", "C"]]),
    "coordinates": np.random.standard_normal((2, 4, 3)),
    "forces": np.random.normal(size=(2, 4, 3), scale=0.1),
    "dipoles": np.random.standard_normal((2, 3)),
    "energies": np.random.standard_normal(2),
}
new_ds.append_conformers("C2H2", numpy_conformers)
print(new_ds.properties)
for c in new_ds.iter_conformers():
    print(c)
# %%
# Another useful feature is deleting inplace all conformers with force
# magnitude above a given threshold, we will exemplify this by introducing some
# conformers with extremely large forces
bad_conformers = {
    "species": np.array([["H", "H", "N", "N"], ["H", "H", "N", "N"]]),
    "coordinates": np.random.standard_normal((2, 4, 3)),
    "forces": np.random.normal(size=(2, 4, 3), scale=100.0),
    "dipoles": np.random.standard_normal((2, 3)),
    "energies": np.random.standard_normal(2),
}
new_ds.append_conformers("C2H2", bad_conformers)
filtered_conformers_and_ids = filter_by_high_force(new_ds, delete_inplace=True)
filtered_conformers_and_ids
# %%
# Finally, lets delete the files we used for cleanup
concat_path.unlink()
new_path.unlink()
