import sys
import torch
import torchani
import configs
import torchani.data
from tqdm import tqdm
import pickle
from common import get_or_create_model, Averager, evaluate

device = configs.device
if len(sys.argv) >= 2:
    device = torch.device(sys.argv[1])

ds = torchani.data.load_dataset(configs.data_path)
model = get_or_create_model('/tmp/model.pt', device=device)
# just to conveniently zero grads
optimizer = torch.optim.Adam(model.parameters())


def grad_or_zero(parameter):
    if parameter.grad is not None:
        return parameter.grad.reshape(-1)
    else:
        return torch.zeros_like(parameter.reshape(-1))


def batch_gradient(batch):
    a = Averager()
    for molecule_id in batch:
        _species = ds.species[molecule_id]
        coordinates, energies = batch[molecule_id]
        coordinates = coordinates.to(model.aev_computer.device)
        energies = energies.to(model.aev_computer.device)
        a.add(*evaluate(coordinates, energies, _species))
    mse = a.avg()
    optimizer.zero_grad()
    mse.backward()
    grads = [grad_or_zero(p) for p in model.parameters()]
    grads = torch.cat(grads)
    return grads


def compute(chunk_size, batch_chunks):
    sampler = torchani.data.BatchSampler(ds, chunk_size, batch_chunks)
    dataloader = torch.utils.data.DataLoader(
        ds, batch_sampler=sampler, collate_fn=torchani.data.collate)

    model_file = 'data/model.pt'
    model.load_state_dict(torch.load(
        model_file, map_location=lambda storage, loc: storage))

    ag = Averager()  # avg(grad)
    agsqr = Averager()  # avg(grad^2)
    for batch in tqdm(dataloader, total=len(sampler)):
        g = batch_gradient(batch)
        ag.add(1, g)
        agsqr.add(1, g**2)
    ag = ag.avg()
    agsqr = agsqr.avg()
    filename = 'data/avg-{}-{}.dat'.format(chunk_size, batch_chunks)
    with open(filename, 'wb') as f:
        pickle.dump((ag, agsqr), f)


chunk_size = int(sys.argv[2])
batch_chunks = int(sys.argv[3])
compute(chunk_size, batch_chunks)
# for chunk_size, batch_chunks in hyperparams:
#     compute(chunk_size, batch_chunks)
