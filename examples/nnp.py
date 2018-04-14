import torchani
import pyanitools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import sys

aev = torchani.AEV()
shift_energy = torchani.EnergyShifter()


net = torchani.NeuralNetworkOnAEV(aev, [128,128,1])
net.reset_parameters()
print(net)

optimizer = optim.Adam(net.parameters(), amsgrad=True)
batch_size = 1024
global_steps = 0

def step(energies, species, coordinates, batch_size, training, epoch, do_print, fileindex):
    label = torch.from_numpy(shift_energy(energies, species)).type(aev.dtype)

    pred = net(coordinates, species)
    squared_error = torch.sum((label - pred) ** 2)
    mse = squared_error  / batch_size
    rmse_kcalmol = 627.509 * torch.sqrt(mse)
    if do_print:
        print('[epoch={},file={}] rmse:'.format(epoch, fileindex), rmse_kcalmol.item(), 'kcal/mol')
    if training:
        loss = 0.5 * torch.exp(2 * mse) if epoch > 10 else mse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def visit_file(filename, training, epoch):
    global global_steps
    print(filename)
    fileindex = filename[-5:-3]
    adl = pyanitools.anidataloader(filename)
    for data in adl:
        global_steps += 1
        remaining_coordinates = torch.from_numpy(data['coordinates']).type(aev.dtype)
        remaining_energies = data['energies']
        remaining_conformations = remaining_coordinates.shape[0]
        species = data['species']
        do_print = global_steps % 100 == 0 or not training

        while remaining_conformations >= 2 * batch_size:
            coordinates = remaining_coordinates[:batch_size]
            remaining_coordinates = remaining_coordinates[batch_size:]
            energies = remaining_energies[:batch_size]
            remaining_energies = remaining_energies[batch_size:]
            remaining_conformations -= batch_size

            step(energies, species, coordinates, batch_size, training, epoch, do_print, fileindex)

        if remaining_conformations > 0:
            step(remaining_energies, species, remaining_coordinates, remaining_conformations, training, epoch, do_print, fileindex)

    adl.cleanup()


for epoch in range(10000):
    start = time.time()
    print('epoch:', epoch)
    print('training')
    for filename in ['data/ani_gdb_s0{}.h5'.format(i) for i in range(1, 8)]:
        visit_file(filename, True, epoch)
    end = time.time()
    print('train epoch time:', end - start)
    if epoch % 100 == 50:
        print('testing')
        start = time.time()
        for filename in ['data/ani_gdb_s0{}.h5'.format(i) for i in range(8, 9)]:
            visit_file(filename, False, epoch)
        end = time.time()
        print('test epoch time:', end - start)

        
            

       
        
