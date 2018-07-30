import os
import sys
import torch
import ignite
import torchani
import timeit
import model

chunk_size = 256
batch_chunks = 4
dataset_path = sys.argv[1]
dataset = torchani.data.ANIDataset(dataset_path, chunk_size)
dataloader = torchani.data.dataloader(dataset, batch_chunks)
nnp = model.get_or_create_model('/tmp/model.pt', True)

class Flatten(torch.nn.Module):

    def __init__(self, model):
        super(Flatten, self).__init__()
        self.model = model

    def forward(self, *input):
        return self.model(*input).flatten()
batch_nnp = torchani.models.BatchModel(Flatten(nnp))
container = torchani.ignite.Container({'energies': batch_nnp})

loss = torchani.ignite.DictLosses({'energies': torch.nn.MSELoss()})
optimizer = torch.optim.Adam(nnp.parameters())

trainer = ignite.engine.create_supervised_trainer(container, optimizer, loss)

start = timeit.default_timer()
trainer.run(dataloader, max_epochs=1)
elapsed = round(timeit.default_timer() - start, 2)
print('Radial terms:', nnp.aev_computer.timers['radial terms'])
print('Angular terms:', nnp.aev_computer.timers['angular terms'])
print('Terms and indices:', nnp.aev_computer.timers['terms and indices'])
print('Combinations:', nnp.aev_computer.timers['combinations'])
print('Mask R:', nnp.aev_computer.timers['mask_r'])
print('Mask A:', nnp.aev_computer.timers['mask_a'])
print('Assemble:', nnp.aev_computer.timers['assemble'])
print('Total AEV:', nnp.aev_computer.timers['total'])
print('NN:', nnp.timers['nn'])
print('Total Forward:', nnp.timers['forward'])
print('Epoch time:', elapsed)