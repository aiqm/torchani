import torch
import torchani
from torchani.data import new_data
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Rcr = 5.2000e+00
Rca = 3.5000e+00
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([8.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)
num_species = 4
aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)

dspath = '/home/richard/dev/torchani/dataset/ani-1x/ANI-1x_complete.h5'

energy_shifter = torchani.utils.EnergyShifter(None)
species_to_tensor = torchani.utils.ChemicalSymbolsToInts('HCNO')

batch_size = 2560

H_network = torch.nn.Sequential(
    torch.nn.Linear(384, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(384, 144),
    torch.nn.CELU(0.1),
    torch.nn.Linear(144, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

N_network = torch.nn.Sequential(
    torch.nn.Linear(384, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(384, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

nn = torchani.ANIModel([H_network, C_network, N_network, O_network])

model = torch.nn.Sequential(aev_computer, nn).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)


def hartree2kcal(x):
    return 627.509 * x


mse = torch.nn.MSELoss(reduction='none')

max_epochs = 2
test1 = True
test2 = True

if test1:
    print('1. using original dataset API')
    print('=> loading dataset...')

    start = time.time()
    training = torchani.data.load_ani_dataset(
        dspath, species_to_tensor, batch_size, device=device,
        transform=[energy_shifter.subtract_from_dataset])
    stop = time.time()
    print('=> loaded - {:.1f}s'.format(stop - start))

    print('=> the first batch is ([chunk1, chunk2, ...], {"energies", "force", ...}) in which chunk1=(species, coordinates)')

    chunks, properties = training[0]
    for i, chunk in enumerate(chunks):
        print('chunk{}'.format(i + 1), list(chunk[0].size()), list(chunk[1].size()))

    print('energies', list(properties['energies'].size()))

    print('=> start training')
    for epoch in range(0, max_epochs):

        print('Epoch: %d/%d' % (epoch, max_epochs))
        progbar = new_data.Progbar(target=len(training) - 1, width=8)

        for i, (batch_x, batch_y) in enumerate(training):

            true_energies = batch_y['energies'].to('cuda')
            predicted_energies = []
            num_atoms = []

            for chunk_species, chunk_coordinates in batch_x:
                chunk_species = chunk_species.to('cuda')
                chunk_coordinates = chunk_coordinates.to('cuda')
                num_atoms.append((chunk_species >= 0).to(true_energies.dtype).sum(dim=1))
                _, chunk_energies = model((chunk_species, chunk_coordinates))
                predicted_energies.append(chunk_energies)

            num_atoms = torch.cat(num_atoms)
            predicted_energies = torch.cat(predicted_energies)
            loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
            rmse = hartree2kcal((mse(predicted_energies, true_energies)).mean()).detach().cpu().numpy()
            loss.backward()
            optimizer.step()

            progbar.update(i, values=[("rmse", rmse)])

if test2:
    print('2. using shuffled dataset')
    shuffledataset = new_data.ShuffleDataset(dspath, batch_size=batch_size, num_workers=2, bar=2)

    print('=> the first batch is ([chunk1, chunk2, ...], {"energies", "force", ...}) in which chunk1=(species, coordinates)')

    chunks, properties = iter(shuffledataset).next()
    for i, chunk in enumerate(chunks):
        print('chunk{}'.format(i + 1), list(chunk[0].size()), list(chunk[1].size()))

    for key, value in properties.items():
        print(key, list(value.size()))

    print('=> start training')
    for epoch in range(0, max_epochs):
        # rmse = validate()

        print('Epoch: %d/%d' % (epoch, max_epochs))
        progbar = new_data.Progbar(target=len(shuffledataset) - 1, width=8)

        for i, (batch_x, batch_y) in enumerate(shuffledataset):

            true_energies = batch_y['energies'].to('cuda')
            predicted_energies = []
            num_atoms = []

            for chunk_species, chunk_coordinates in batch_x:
                chunk_species = chunk_species.to('cuda')
                chunk_coordinates = chunk_coordinates.to('cuda')
                num_atoms.append((chunk_species >= 0).to(true_energies.dtype).sum(dim=1))
                _, chunk_energies = model((chunk_species, chunk_coordinates))
                predicted_energies.append(chunk_energies)

            num_atoms = torch.cat(num_atoms)
            predicted_energies = torch.cat(predicted_energies)
            loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
            rmse = hartree2kcal((mse(predicted_energies, true_energies)).mean()).detach().cpu().numpy()
            loss.backward()
            optimizer.step()

            progbar.update(i, values=[("rmse", rmse)])
