import sys
import torch
import torchani
import configs
import torchani.data
import math
from tqdm import tqdm
import itertools
import os
import pickle


if len(sys.argv)>=2:
    configs.device = torch.device(sys.argv[1])
from common import *

ds = torchani.data.load_dataset(configs.data_path)
optimizer = torch.optim.Adam(model.parameters())  # just to conveniently zero grads

hyperparams = [  # (chunk size, batch chunks)
    (64, 4),
    (64, 8),
    (64, 16),
    (64, 32),
    (128, 2),
    (128, 4),
    (128, 8),
    (128, 16),
    (256, 1),
    (256, 2),
    (256, 4),
    (256, 8),
    (512, 1),
    (512, 2),
    (512, 4),
    (1024, 1),
    (1024, 2),
    (2048, 1),
]

def batch_gradient(batch):
    a = Averager()
    for molecule_id in batch:
        _species = ds.species[molecule_id]
        coordinates, energies = batch[molecule_id]
        coordinates = coordinates.to(aev_computer.device)
        energies = energies.to(aev_computer.device)
        a.add(*evaluate(coordinates, energies, _species))
    mse = a.avg()
    optimizer.zero_grad()
    mse.backward()
    grads = [p.grad.reshape(-1) for p in model.parameters()]
    grads = torch.cat(grads)
    return grads

def compute(chunk_size, batch_chunks):
    sampler = torchani.data.BatchSampler(ds, chunk_size, batch_chunks)
    dataloader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=torchani.data.collate)

    model_file = 'data/model.pt'
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    ag = Averager()  # avg(grad)
    agsqr = Averager()  # avg(grad^2)
    for batch in tqdm(dataloader, total=len(sampler)):
        g = batch_gradient(batch)
        ag.add(1, g)
        agsqr.add(1, g**2)
    ag = ag.avg()
    agsqr = agsqr.avg()
    with open('data/avg-{}-{}.dat'.format(chunk_size, batch_chunks), 'wb') as f:
        pickle.dump((ag, agsqr), f)

chunk_size = int(sys.argv[2])
batch_chunks = int(sys.argv[3])
compute(chunk_size, batch_chunks)
# for chunk_size, batch_chunks in hyperparams:
#     compute(chunk_size, batch_chunks)