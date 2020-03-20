# -*- coding: utf-8 -*-
"""Tools for loading, shuffling, and batching ANI datasets"""

from torch.utils.data import Dataset
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

PKBAR_INSTALLED = importlib.util.find_spec('pkbar') is not None
if PKBAR_INSTALLED:
    import pkbar

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
verbose = True


class Transformations:

    @staticmethod
    def species_to_indices(iter_, species_order=('H', 'C', 'N', 'O', 'F', 'Cl', 'S')):
        if species_order == 'periodic_table':
            species_order = utils.PERIODIC_TABLE
        idx = {k: i for i, k in enumerate(species_order)}
        for d in iter_:
            d['species'] = [idx[s] for s in d['species']]
            yield d

    @staticmethod
    def subtract_self_energies(iter_, self_energies=None):
        iter_ = list(iter_)
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
            species = list(counts.keys())
            X = [counts[s] for s in species]
            if shifter.fit_intercept:
                X.append([1] * n)
            X = numpy.array(X).transpose()
            Y = numpy.array(Y)
            sae, _, _, _ = numpy.linalg.lstsq(X, Y, rcond=None)
            if shifter.fit_intercept:
                intercept = sae[-1]
                sae = sae[:-1]
            for s, e in zip(species, sae):
                self_energies[s] = e
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
        l = list(iter_)
        random.shuffle(l)
        return iter(l)

    @staticmethod
    def cache(iter_):
        return iter(list(iter_))


class TransformableIterator:
    def __init__(self, wrapped_iter, transformations=()):
        self.wrapped_iter = wrapped_iter
        self.transformations = transformations

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.wrapped_iter)

    def __getattr__(self, name):
        transformation = getattr(Transformations, name)

        @functools.wraps(transformation)
        def f(*args, **kwargs):
            return TransformableIterator(
                transformation(self, *args, **kwargs),
                self.transformations + (name,))

        return f


def load(path, additional_properties=()):
    PROPERTIES = ('energies', 'forces')
    properties = PROPERTIES + additional_properties

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
            anidata_size = anidata.size()
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

    return TransformableIterator(conformations())


__all__ = ['load']
