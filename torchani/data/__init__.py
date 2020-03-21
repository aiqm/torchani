# -*- coding: utf-8 -*-
"""Tools for loading, shuffling, and batching ANI datasets

The `torchani.data.load(path)` creates an iterable of raw data,
where species are strings, and coordinates are numpy ndarrays.

You can transform these iterable by using transformations.
To do transformation, just do `it.transformation_name()`.

Available transformations are listed below:

- `species_to_indices` converts species from strings to numbers.
- `subtract_self_energies` subtracts self energies, you can pass.
    a dict of self energies, or an `EnergyShifter` to let it infer
    self energy from dataset and store the result to the given shifter.
- `remove_outliers`
- `shuffle`
- `cache` cache the result of previous transformations.
- `collate` pad the dataset, convert it to tensor, and stack them
    together to get a batch.
- `pin_memory` copy the tensor to pinned memory so that later transfer
    to cuda could be faster.

You can also use `split` to split the iterable to pieces. Use `split` as:

.. code-block:: python

    it.split(size1, size2, None)

where the None in the end indicate that we want to use all of the the rest

Example:

.. code-block:: python

    energy_shifter = torchani.utils.EnergyShifter(None)
    dataset = torchani.data.load(path).subtract_self_energies(energy_shifter).species_to_indices().shuffle()
    size = len(dataset)
    training, validation = dataset.split(int(0.8 * size), None)
    training = training.collate(batch_size).cache()
    validation = validation.collate(batch_size).cache()
"""

from os.path import join, isfile, isdir
import os
from ._pyanitools import anidataloader
import torch
from .. import utils
import importlib
import functools
import math
import random
from collections import Counter
import numpy

PKBAR_INSTALLED = importlib.util.find_spec('pkbar') is not None  # type: ignore
if PKBAR_INSTALLED:
    import pkbar

verbose = True


PROPERTIES = ('energies', 'forces')
PADDING = {
    'species': -1,
    'coordinates': 0.0,
    'forces': 0.0,
    'energies': 0.0
}


class Transformations:

    @staticmethod
    def species_to_indices(iter_, species_order=('H', 'C', 'N', 'O', 'F', 'Cl', 'S')):
        if species_order == 'periodic_table':
            species_order = utils.PERIODIC_TABLE
        idx = {k: i for i, k in enumerate(species_order)}
        for d in iter_:
            d['species'] = numpy.array([idx[s] for s in d['species']])
            yield d

    @staticmethod
    def subtract_self_energies(iter_, self_energies=None):
        intercept = 0.0
        if isinstance(self_energies, utils.EnergyShifter):
            shifter = self_energies
            self_energies = {}
            counts = {}
            Y = []
            for n, d in enumerate(iter_):
                species = d['species']
                count = Counter()
                for s in species:
                    count[s] += 1
                for s, c in count.items():
                    if s not in counts:
                        counts[s] = [0] * n
                    counts[s].append(c)
                for s in counts:
                    if len(counts[s]) != n + 1:
                        counts[s].append(0)
                Y.append(d['energies'])
            species = sorted(list(counts.keys()))
            X = [counts[s] for s in species]
            if shifter.fit_intercept:
                X.append([1] * n)
            X = numpy.array(X).transpose()
            Y = numpy.array(Y)
            sae, _, _, _ = numpy.linalg.lstsq(X, Y, rcond=None)
            sae_ = sae
            if shifter.fit_intercept:
                intercept = sae[-1]
                sae_ = sae[:-1]
            for s, e in zip(species, sae_):
                self_energies[s] = e
            shifter.__init__(sae, shifter.fit_intercept)
        for d in iter_:
            e = intercept
            for s in d['species']:
                e += self_energies[s]
            d['energies'] -= e
            yield d

    @staticmethod
    def remove_outliers(iter_, threshold1=15.0, threshold2=8.0):
        assert 'subtract_self_energies', "Transformation remove_outliers can only run after subtract_self_energies"

        # pass 1: remove everything that has per-atom energy > threshold1
        def scaled_energy(x):
            num_atoms = len(x['species'])
            return abs(x['energies']) / math.sqrt(num_atoms)
        filtered = [x for x in iter_ if scaled_energy(x) < threshold1]

        # pass 2: compute those that are outside the mean by threshold2 * std
        n = 0
        mean = 0
        std = 0
        for m in filtered:
            n += 1
            mean += m['energies']
            std += m['energies'] ** 2
        mean /= n
        std = math.sqrt(std / n - mean ** 2)

        return filter(lambda x: abs(x['energies'] - mean) < threshold2 * std, filtered)

    @staticmethod
    def shuffle(iter_):
        list_ = list(iter_)
        random.shuffle(list_)
        return list_

    @staticmethod
    def cache(iter_):
        return list(iter_)

    @staticmethod
    def collate(iter_, batch_size):
        batch = []
        i = 0
        for d in iter_:
            d = {k: torch.as_tensor(d[k]) for k in d}
            batch.append(d)
            i += 1
            if i == batch_size:
                i = 0
                yield utils.stack_with_padding(batch, PADDING)
                batch = []
        if len(batch) > 0:
            yield utils.stack_with_padding(batch, PADDING)

    @staticmethod
    def pin_memory(iter_):
        for d in iter_:
            yield {k: d[k].pin_memory() for k in d}


class TransformableIterable:
    def __init__(self, wrapped_iter, transformations=()):
        self.wrapped_iter = wrapped_iter
        self.transformations = transformations

    def __iter__(self):
        return iter(self.wrapped_iter)

    def __getattr__(self, name):
        transformation = getattr(Transformations, name)

        @functools.wraps(transformation)
        def f(*args, **kwargs):
            return TransformableIterable(
                transformation(self, *args, **kwargs),
                self.transformations + (name,))

        return f

    def split(self, *nums):
        iters = []
        self_iter = iter(self)
        for n in nums:
            list_ = []
            if n is not None:
                for _ in range(n):
                    list_.append(next(self_iter))
            else:
                for i in self_iter:
                    list_.append(i)
            iters.append(TransformableIterable(list_, self.transformations + ('split',)))
        return iters

    def __len__(self):
        return len(self.wrapped_iter)


def load(path, additional_properties=()):
    properties = PROPERTIES + additional_properties

    # https://stackoverflow.com/a/39564774
    class IterableAdapter:
        def __init__(self, iterator_factory):
            self.iterator_factory = iterator_factory

        def __iter__(self):
            return self.iterator_factory()

    def h5_files(path):
        """yield file name of all h5 files in a path"""
        if isdir(path):
            for f in os.listdir(path):
                f = join(path, f)
                yield from h5_files(f)
        elif isfile(path) and (path.endswith('.h5') or path.endswith('.hdf5')):
            yield path

    def molecules():
        for f in h5_files(path):
            anidata = anidataloader(f)
            anidata_size = anidata.group_size()
            use_pbar = PKBAR_INSTALLED and verbose
            if use_pbar:
                pbar = pkbar.Pbar('=> loading {}, total molecules: {}'.format(f, anidata_size), anidata_size)
            for i, m in enumerate(anidata):
                yield m
                if use_pbar:
                    pbar.update(i)

    def conformations():
        for m in molecules():
            species = m['species']
            coordinates = m['coordinates']
            for i in range(coordinates.shape[0]):
                ret = {'species': species, 'coordinates': coordinates[i]}
                for k in properties:
                    if k in m:
                        ret[k] = m[k][i]
                yield ret

    return TransformableIterable(IterableAdapter(lambda: conformations()))


__all__ = ['load']
