from torch.nn.modules.loss import _Loss
from ignite.metrics.metric import Metric
from ignite.metrics import RootMeanSquaredError
import torch


class DictLoss(_Loss):

    def __init__(self, key, loss):
        super(DictLoss, self).__init__()
        self.key = key
        self.loss = loss

    def forward(self, input, other):
        return self.loss(input[self.key], other[self.key])


class _PerAtomDictLoss(DictLoss):

    @staticmethod
    def num_atoms(input):
        ret = []
        for s, c in zip(input['species'], input['coordinates']):
            ret.append(torch.full((c.shape[0],), len(s),
                       dtype=c.dtype, device=c.device))
        return torch.cat(ret)

    def forward(self, input, other):
        loss = self.loss(input[self.key], other[self.key])
        loss /= self.num_atoms(input)
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
        return _PerAtomDictLoss(key, torch.nn.MSELoss(reduce=False))
    else:
        return DictLoss(key, torch.nn.MSELoss())


def RMSEMetric(key):
    return DictMetric(key, RootMeanSquaredError())
