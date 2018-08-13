import torch


class Container(torch.nn.Module):

    def __init__(self, models):
        super(Container, self).__init__()
        self.keys = models.keys()
        for i in models:
            setattr(self, 'model_' + i, models[i])

    def forward(self, species_coordinates):
        species, coordinates = species_coordinates
        results = {
            'species': species,
            'coordinates': coordinates,
        }
        for k in self.keys:
            model = getattr(self, 'model_' + k)
            _, results[k] = model((species, coordinates))
        return results
