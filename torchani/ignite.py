import torch
from . import utils
from torch.nn.modules.loss import _Loss
from ignite.metrics.metric import Metric
from ignite.metrics import RootMeanSquaredError


class Container(torch.nn.ModuleDict):

    def __init__(self, modules):
        super(Container, self).__init__(modules)

    def forward(self, species_coordinates):
        results = {k: [] for k in self}
        for sc in species_coordinates:
            for k in self:
                _, result = self[k](sc)
                results[k].append(result)
        for k in self:
            results[k] = torch.cat(results[k])
        results['species'], results['coordinates'] = \
            utils.pad_and_batch(species_coordinates)
        return results


class DictLoss(_Loss):

    def __init__(self, key, loss):
        super(DictLoss, self).__init__()
        self.key = key
        self.loss = loss

    def forward(self, input, other):
        return self.loss(input[self.key], other[self.key])


class _PerAtomDictLoss(DictLoss):

    def forward(self, input, other):
        loss = self.loss(input[self.key], other[self.key])
        num_atoms = (input['species'] >= 0).sum(dim=1)
        loss /= num_atoms.to(loss.dtype).to(loss.device)
        n = loss.numel()
        return loss.sum() / n


class DictMetric(Metric):

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
    if per_atom:
        return _PerAtomDictLoss(key, torch.nn.MSELoss(reduction='none'))
    else:
        return DictLoss(key, torch.nn.MSELoss())


class TransformedLoss(_Loss):

    def __init__(self, origin, transform):
        super(TransformedLoss, self).__init__()
        self.origin = origin
        self.transform = transform

    def forward(self, input, other):
        return self.transform(self.origin(input, other))


def RMSEMetric(key):
    return DictMetric(key, RootMeanSquaredError())
