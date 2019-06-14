import os
import torch
import torchani
import ignite
import pickle
import argparse


ani1x = torchani.models.ANI1x()

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset_path',
                    help='Path of the dataset. The path can be a hdf5 file or \
                    a directory containing hdf5 files. It can also be a file \
                    dumped by pickle.')
parser.add_argument('-d', '--device',
                    help='Device of modules and tensors',
                    default=('cuda' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--batch_size',
                    help='Number of conformations of each batch',
                    default=1024, type=int)
parser.add_argument('--const_file',
                    help='File storing constants',
                    default=ani1x.const_file)
parser.add_argument('--sae_file',
                    help='File storing self atomic energies',
                    default=ani1x.sae_file)
parser.add_argument('--network_dir',
                    help='Directory or prefix of directories storing networks',
                    default=ani1x.ensemble_prefix + '0/networks')
parser.add_argument('--compare_with',
                    help='The TorchANI model to compare with', default=None)
parser = parser.parse_args()

# load modules and datasets
device = torch.device(parser.device)
consts = torchani.neurochem.Constants(parser.const_file)
shift_energy = torchani.neurochem.load_sae(parser.sae_file)
aev_computer = torchani.AEVComputer(**consts)
nn = torchani.neurochem.load_model(consts.species, parser.network_dir)
model = torch.nn.Sequential(aev_computer, nn)
container = torchani.ignite.Container({'energies': model})
container = container.to(device)

# load datasets
if parser.dataset_path.endswith('.h5') or \
   parser.dataset_path.endswith('.hdf5') or \
   os.path.isdir(parser.dataset_path):
    dataset = torchani.data.BatchedANIDataset(
        parser.dataset_path, consts.species_to_tensor, parser.batch_size,
        device=device, transform=[shift_energy.subtract_from_dataset])
    datasets = [dataset]
else:
    with open(parser.dataset_path, 'rb') as f:
        datasets = pickle.load(f)
        if not isinstance(datasets, list) and not isinstance(datasets, tuple):
            datasets = [datasets]


# prepare evaluator
def hartree2kcal(x):
    return 627.509 * x


def evaluate(dataset, container):
    evaluator = ignite.engine.create_supervised_evaluator(container, metrics={
        'RMSE': torchani.ignite.RMSEMetric('energies')
    })
    evaluator.run(dataset)
    metrics = evaluator.state.metrics
    rmse = hartree2kcal(metrics['RMSE'])
    print(rmse, 'kcal/mol')


for dataset in datasets:
    evaluate(dataset, container)


if parser.compare_with is not None:
    nn.load_state_dict(torch.load(parser.compare_with))
    print('TorchANI results:')
    for dataset in datasets:
        evaluate(dataset, container)
