import torch
from ..data import collate


class Container(torch.nn.Module):

    def __init__(self, models):
        super(Container, self).__init__()
        self.keys = models.keys()
        for i in models:
            setattr(self, 'model_' + i, models[i])

    def forward(self, batch):
        all_results = []
        for i in zip(batch['species'], batch['coordinates']):
            results = {}
            for k in self.keys:
                model = getattr(self, 'model_' + k)
                _, results[k] = model(i)
                all_results.append(results)
        batch.update(collate(all_results))
        return batch
