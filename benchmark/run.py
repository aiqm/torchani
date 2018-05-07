import h5py
import torch
from selected_system import mols, mol_file
from ani_benchmark import NeighborBenchmark,FreeNeighborBenchmark,NoNeighborBenchmark
import pandas

torch.set_num_threads(1)

fm = h5py.File(mol_file, "r")
benchmarks = {
    'N,C': NeighborBenchmark(device=torch.device("cpu")),
    'N,G': NeighborBenchmark(device=torch.device("cuda")),
    'FN,C': FreeNeighborBenchmark(device=torch.device("cpu")),
    'FN,G': FreeNeighborBenchmark(device=torch.device("cuda")),
    'NN,C': NoNeighborBenchmark(device=torch.device("cpu")),
    'NN,G': NoNeighborBenchmark(device=torch.device("cuda")),
}

for i in mols:
    print('number of atoms:', i)
    smiles = mols[i]
    for s in smiles:
        print('Running benchmark on molecule', s)
        key = s.replace('/', '_')
        coordinates = torch.from_numpy(fm[key][()])
        species = fm[key].attrs['species'].split()
        results = {}
        for b in benchmarks:
            bench = benchmarks[b]
            coordinates = coordinates.type(bench.aev_computer.dtype)
            try:
                result = bench.oneByOne(coordinates, species)
            except RuntimeError:
                result = { 'aev': None, 'energy': None, 'force': None }
            results[b + ',1'] = result
            try:
                result = bench.inBatch(coordinates, species)
            except RuntimeError:
                result = { 'aev': None, 'energy': None, 'force': None }
            results[b + ',B'] = result
        df = pandas.DataFrame(results)
        print(df)
        break
