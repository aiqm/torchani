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

torch.cuda.cudart().cudaProfilerStart()
autonvtx(model)
with torch.autograd.profiler.emit_nvtx(record_shapes=True):
    dyn.run(10)
