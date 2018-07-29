import torch
from ..models import BatchModel


class Container(torch.nn.Module):

    def __init__(self, models):
        super(Container, self).__init__()
        self.keys = models.keys()
        for i in models:
            if not isinstance(models[i], BatchModel):
                raise ValueError('Container must contain batch models')
            setattr(self, 'model_' + i, models[i])

    def forward(self, batch):
        output = {}
        for i in self.keys:
            model = getattr(self, 'model_' + i)
            output[i] = model(batch)
        return output
