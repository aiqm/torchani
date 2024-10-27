import os

import torch
from torchani.aev import AEVComputer

aev = AEVComputer.like_1x()

# This test only runs as a conda test, and only if there is a cuda device
if torch.cuda.is_available() and (os.environ.get("CONDA_BUILD_STATE", None) == "TEST"):
    coordinates = torch.tensor(
        [
            [
                [0.03192167, 0.00638559, 0.01301679],
                [-0.83140486, 0.39370209, -0.26395324],
                [-0.66518241, -0.84461308, 0.20759389],
                [0.45554739, 0.54289633, 0.81170881],
                [0.66091919, -0.16799635, -0.91037834],
            ]
        ],
        device="cuda",
    )
    species = torch.tensor([[1, 0, 0, 0, 0]], device="cuda")
    aev = aev.cuda()
    cuaev = AEVComputer.like_1x(strategy="cuaev-fused").cuda()
    aevs_cu = cuaev((species, coordinates)).aevs
    aevs_py = aev((species, coordinates)).aevs
    assert ((aevs_cu - aevs_py) < 1e-4).all()
