import torch
import torchani
import torchani.data
import tqdm
import timeit
import configs
import functools
from common import get_or_create_model, Averager, evaluate

ds = torchani.data.load_dataset(configs.data_path)
sampler = torchani.data.BatchSampler(ds, 256, 4)
dataloader = torch.utils.data.DataLoader(
    ds, batch_sampler=sampler,
    collate_fn=torchani.data.collate, num_workers=20)
model = get_or_create_model('/tmp/model.pt', True)
optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)


def benchmark(timer, index):
    def wrapper(fun):
        @functools.wraps(fun)
        def wrapped(*args, **kwargs):
            start = timeit.default_timer()
            ret = fun(*args, **kwargs)
            end = timeit.default_timer()
            timer[index] += end - start
            return ret
        return wrapped
    return wrapper


timer = {'backward': 0}


@benchmark(timer, 'backward')
def optimize_step(a):
    mse = a.avg()
    optimizer.zero_grad()
    mse.backward()
    optimizer.step()


start = timeit.default_timer()
for batch in tqdm.tqdm(dataloader, total=len(sampler)):
    a = Averager()
    for molecule_id in batch:
        _species = ds.species[molecule_id]
        coordinates, energies = batch[molecule_id]
        coordinates = coordinates.to(model.aev_computer.device)
        energies = energies.to(model.aev_computer.device)
        a.add(*evaluate(model, coordinates, energies, _species))
    optimize_step(a)

elapsed = round(timeit.default_timer() - start, 2)
print('Radial terms:', model.aev_computer.timers['radial terms'])
print('Angular terms:', model.aev_computer.timers['angular terms'])
print('Terms and indices:', model.aev_computer.timers['terms and indices'])
print('Combinations:', model.aev_computer.timers['combinations'])
print('Mask R:', model.aev_computer.timers['mask_r'])
print('Mask A:', model.aev_computer.timers['mask_a'])
print('Assemble:', model.aev_computer.timers['assemble'])
print('Total AEV:', model.aev_computer.timers['total'])
print('NN:', model.timers['nn'])
print('Total Forward:', model.timers['forward'])
print('Total Backward:', timer['backward'])
print('Epoch time:', elapsed)
