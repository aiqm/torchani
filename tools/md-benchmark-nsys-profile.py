import argparse

import torch
import ase
import ase.io
import ase.md

from torchani.models import ANI1x
from tool_utils import time_functions


def patch(model, name=None):
    if name is None:
        name = type(model).__name__
    else:
        name = name + ': ' + type(model).__name__

    def push(*args, _name=name, **kwargs):
        torch.cuda.nvtx.range_push(_name)

    def pop(*args, **kwargs):
        torch.cuda.nvtx.range_pop()

    model.register_forward_pre_hook(push)
    model.register_forward_hook(pop)

    for name, child in model.named_children():
        patch(child, name)

    return model


parser = argparse.ArgumentParser()
parser.add_argument("filename", help="file for the molecule")
args = parser.parse_args()

molecule = ase.io.read(args.filename)
model = ANI1x(model_index=0).cuda()
molecule.calc = model.ase()
dyn = ase.md.verlet.VelocityVerlet(molecule, timestep=1 * ase.units.fs)

dyn.run(1000)  # warm up
time_functions(
    [
        ("forward", model.aev_computer.neighborlist),
        ("forward", model.aev_computer.angular_terms),
        ("forward", model.aev_computer.radial_terms),
        (
            (
                "_compute_radial_aev",
                "_compute_angular_aev",
                "_compute_aev",
                "_triple_by_molecule",
                "forward",
            ),
            model.aev_computer,
        ),
        ("forward", model.neural_networks),
        ("forward", model.energy_shifter),
    ],
    timers={},
    sync=True,
    nvtx=True,
)
torch.cuda.cudart().cudaProfilerStart()
patch(model)
with torch.autograd.profiler.emit_nvtx(record_shapes=True):
    dyn.run(10)  # profile
