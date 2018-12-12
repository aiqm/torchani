# -*- coding: utf-8 -*-
"""Helpers for working with ignite."""

import torch
from . import utils
from torch.nn.modules.loss import _Loss
from ignite.metrics.metric import Metric
from ignite.metrics import RootMeanSquaredError
from ignite.contrib.metrics.regression import MaximumAbsoluteError


class Container(torch.nn.ModuleDict):
    """Each minibatch is splitted into chunks, as explained in the docstring of
    :class:`torchani.data.BatchedANIDataset`, as a result, it is impossible to
    use :class:`torchani.AEVComputer`, :class:`torchani.ANIModel` directly with
    ignite. This class is designed to solve this issue.

    Arguments:
        modules (:class:`collections.abc.Mapping`): same as the argument in
            :class:`torch.nn.ModuleDict`.
    """

    def __init__(self, modules):
        super(Container, self).__init__(modules)

    def forward(self, species_x):
        """Takes sequence of species, coordinates pair as input, and returns
        computed properties as a dictionary. Same property from different
        chunks will be concatenated to form a single tensor for a batch.
        """
        results = {k: [] for k in self}
        for sx in species_x:
            for k in self:
                _, result = self[k](sx)
                results[k].append(result)
        for k in self:
            results[k] = torch.cat(results[k])
        results['species'] = utils.pad([s for s, _ in species_x])
        return results


class DictLoss(_Loss):
    """Since :class:`Container` output dictionaries, losses defined in
    :attr:`torch.nn` needs to be wrapped before used. This class wraps losses
    that directly work on tensors with a key by calling the wrapped loss on the
    associated value of that key.
    """

    def __init__(self, key, loss):
        super(DictLoss, self).__init__()
        self.key = key
        self.loss = loss

    def forward(self, input_, other):
        return self.loss(input_[self.key], other[self.key])


class PerAtomDictLoss(DictLoss):
    """Similar to :class:`DictLoss`, but scale the loss values by the number of
    atoms for each structure. The `loss` argument must be set to not to reduce
    by the caller. Currently the only reduce operation supported is averaging.
    """

    def forward(self, input_, other):
        loss = self.loss(input_[self.key], other[self.key])
        num_atoms = (input_['species'] >= 0).sum(dim=1)
        loss /= num_atoms.to(loss.dtype).to(loss.device)
        n = loss.numel()
        return loss.sum() / n


class DictMetric(Metric):
    """Similar to :class:`DictLoss`, but this is for metric, not loss."""

    def __init__(self, key, metric):
        self.key = key
        self.metric = metric
        super(DictMetric, self).__init__()

    def reset(self):
        self.metric.reset()

    def update(self, output):
        y_pred, y = output
        self.metric.update((y_pred[self.key], y[self.key]))

    def compute(self):
        return self.metric.compute()


def MSELoss(key, per_atom=True):
    """Create MSE loss on the specified key."""
    if per_atom:
        return PerAtomDictLoss(key, torch.nn.MSELoss(reduction='none'))
    return DictLoss(key, torch.nn.MSELoss())


class TransformedLoss(_Loss):
    """Do a transformation on loss values."""

    def __init__(self, origin, transform):
        super(TransformedLoss, self).__init__()
        self.origin = origin
        self.transform = transform

    def forward(self, input_, other):
        return self.transform(self.origin(input_, other))


def RMSEMetric(key):
    """Create RMSE metric on key."""
    return DictMetric(key, RootMeanSquaredError())


def MAEMetric(key):
    """Create max absolute error metric on key."""
    return DictMetric(key, MaximumAbsoluteError())


__all__ = ['Container', 'MSELoss', 'TransformedLoss', 'RMSEMetric',
           'MAEMetric']
