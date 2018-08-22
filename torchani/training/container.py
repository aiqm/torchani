import torch
from .. import utils


class Container(torch.nn.Module):

    def __init__(self, models):
        super(Container, self).__init__()
        self.keys = models.keys()
        for i in models:
            setattr(self, 'model_' + i, models[i])

    def forward(self, species_coordinates):
        results = {k: [] for k in self.keys}
        for sc in species_coordinates:
            for k in self.keys:
                model = getattr(self, 'model_' + k)
                _, result = model(sc)
                results[k].append(result)
        results['species'], results['coordinates'] = \
            utils.pad_and_batch(species_coordinates)
        for k in self.keys:
            results[k] = torch.cat(results[k])
        return results
