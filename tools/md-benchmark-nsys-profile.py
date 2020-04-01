import ase
import ase.io
import ase.md
import argparse
import torchani
import autonvtx
import torch

parser = argparse.ArgumentParser()
parser.add_argument('filename', help="file for the molecule")
args = parser.parse_args()

molecule = ase.io.read(args.filename)
model = torchani.models.ANI1x()[0].cuda()
calculator = model.ase()
molecule.set_calculator(calculator)
dyn = ase.md.verlet.VelocityVerlet(molecule, timestep=1 * ase.units.fs)

dyn.run(1000)  # warm up


def time_func(key, func):

    def wrapper(*args, **kwargs):
        torch.cuda.nvtx.range_push(key)
        ret = func(*args, **kwargs)
        torch.cuda.nvtx.range_pop()
        return ret

    return wrapper


torchani.aev.cutoff_cosine = time_func('cutoff_cosine', torchani.aev.cutoff_cosine)
torchani.aev.radial_terms = time_func('radial_terms', torchani.aev.radial_terms)
torchani.aev.angular_terms = time_func('angular_terms', torchani.aev.angular_terms)
torchani.aev.compute_shifts = time_func('compute_shifts', torchani.aev.compute_shifts)
torchani.aev.neighbor_pairs = time_func('neighbor_pairs', torchani.aev.neighbor_pairs)
torchani.aev.neighbor_pairs_nopbc = time_func('neighbor_pairs_nopbc', torchani.aev.neighbor_pairs_nopbc)
torchani.aev.triu_index = time_func('triu_index', torchani.aev.triu_index)
torchani.aev.cumsum_from_zero = time_func('cumsum_from_zero', torchani.aev.cumsum_from_zero)
torchani.aev.triple_by_molecule = time_func('triple_by_molecule', torchani.aev.triple_by_molecule)
torchani.aev.compute_aev = time_func('compute_aev', torchani.aev.compute_aev)

torch.cuda.cudart().cudaProfilerStart()
autonvtx(model)
with torch.autograd.profiler.emit_nvtx(record_shapes=True):
    dyn.run(10)
