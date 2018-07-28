import torch


class BatchModel(torch.nn.Module):

    def __init__(self, model):
        super(BatchModel, self).__init__()
        self.model = model

    def forward(self, batch):
        results = []
        for i in batch:
            coordinates = i['coordinates']
            species = i['species']
            results.append(self.model(coordinates, species))
        return torch.cat(results)
