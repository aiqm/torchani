import torch
import torchani
import torchani.data
import math
import timeit
import pickle
from tensorboardX import SummaryWriter
from tqdm import tqdm
from common import get_or_create_model, Averager, evaluate

chunk_size = 256
batch_chunks = 1024 // chunk_size

with open('data/dataset.dat', 'rb') as f:
    training, validation, testing = pickle.load(f)

    training_sampler = torchani.data.BatchSampler(
        training, chunk_size, batch_chunks)
    validation_sampler = torchani.data.BatchSampler(
        validation, chunk_size, batch_chunks)
    testing_sampler = torchani.data.BatchSampler(
        testing, chunk_size, batch_chunks)

    training_dataloader = torch.utils.data.DataLoader(
        training, batch_sampler=training_sampler,
        collate_fn=torchani.data.collate)
    validation_dataloader = torch.utils.data.DataLoader(
        validation, batch_sampler=validation_sampler,
        collate_fn=torchani.data.collate)
    testing_dataloader = torch.utils.data.DataLoader(
        testing, batch_sampler=testing_sampler,
        collate_fn=torchani.data.collate)

writer = SummaryWriter()

checkpoint = 'checkpoint.pt'
model = get_or_create_model(checkpoint)
optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
step = 0
epoch = 0


def subset_rmse(subset_dataloader):
    a = Averager()
    for batch in subset_dataloader:
        for molecule_id in batch:
            _species = subset_dataloader.dataset.species[molecule_id]
            coordinates, energies = batch[molecule_id]
            coordinates = coordinates.to(model.aev_computer.device)
            energies = energies.to(model.aev_computer.device)
            count, squared_error = evaluate(coordinates, energies, _species)
            squared_error = squared_error.item()
            a.add(count, squared_error)
    mse = a.avg()
    rmse = math.sqrt(mse) * 627.509
    return rmse


def optimize_step(a):
    mse = a.avg()
    rmse = math.sqrt(mse.item()) * 627.509
    writer.add_scalar('training_rmse_vs_step', rmse, step)
    loss = mse if epoch < 10 else 0.5 * torch.exp(2 * mse)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


best_validation_rmse = math.inf
best_epoch = 0
start = timeit.default_timer()
while True:
    for batch in tqdm(training_dataloader, desc='epoch {}'.format(epoch),
                      total=len(training_sampler)):
        a = Averager()
        for molecule_id in batch:
            _species = training.species[molecule_id]
            coordinates, energies = batch[molecule_id]
            coordinates = coordinates.to(model.aev_computer.device)
            energies = energies.to(model.aev_computer.device)
            count, squared_error = evaluate(
                model, coordinates, energies, _species)
            a.add(count, squared_error / len(_species))
        optimize_step(a)
        step += 1

    validation_rmse = subset_rmse(validation_dataloader)
    elapsed = round(timeit.default_timer() - start, 2)
    print('Epoch:', epoch, 'time:', elapsed,
          'validation rmse:', validation_rmse)
    writer.add_scalar('validation_rmse_vs_epoch', validation_rmse, epoch)
    writer.add_scalar('epoch_vs_step', epoch, step)
    writer.add_scalar('time_vs_epoch', elapsed, epoch)

    if validation_rmse < best_validation_rmse:
        best_validation_rmse = validation_rmse
        best_epoch = epoch
        writer.add_scalar('best_validation_rmse_vs_epoch',
                          best_validation_rmse, best_epoch)
        torch.save(model.state_dict(), checkpoint)
    elif epoch - best_epoch > 1000:
        print('Stop at best validation rmse:', best_validation_rmse)
        break

    epoch += 1

testing_rmse = subset_rmse(testing_dataloader)
print('Test rmse:', validation_rmse)
