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


energy_mse_loss = DictLoss('energies', torch.nn.MSELoss())
energy_rmse_metric = DictMetric('energies', RootMeanSquaredError())
