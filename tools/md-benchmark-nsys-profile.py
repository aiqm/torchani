import ase
import ase.io
import ase.md
import argparse
import torchani
import autonvtx
import torch
from tool_utils import time_functions_in_model

parser = argparse.ArgumentParser()
parser.add_argument('filename', help="file for the molecule")
args = parser.parse_args()

molecule = ase.io.read(args.filename)
model = torchani.models.ANI1x(model_index=0).cuda()
calculator = model.ase()
molecule.calc = calculator
dyn = ase.md.verlet.VelocityVerlet(molecule, timestep=1 * ase.units.fs)

dyn.run(1000)  # warm up

# enable timers
fn_to_time_aev = ['_compute_radial_aev', '_compute_angular_aev',
                         '_compute_aev', '_triple_by_molecule']
fn_to_time_neighborlist = ['forward']

aev_computer = model.aev_computer
time_functions_in_model(aev_computer, fn_to_time_aev, nvtx=True)
time_functions_in_model(aev_computer.neighborlist, fn_to_time_neighborlist, nvtx=True)

torch.cuda.cudart().cudaProfilerStart()
autonvtx(model)
with torch.autograd.profiler.emit_nvtx(record_shapes=True):
    dyn.run(10)
