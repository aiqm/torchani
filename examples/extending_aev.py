r"""
Extending the local atomic features: AEVs with custom terms and cutoffs
=======================================================================

TorchANI allows for simple customization of the AEV features. This is an important
extension point of the library. Here we explain how to implement your own custom AEV
terms and cutoff functions.
"""

# To begin with, let's first import the modules and setup devices we will use:
import math

import torch
from torch import Tensor

from torchani.cutoffs import Cutoff
from torchani.aev import ANIRadial, AEVComputer, AngularTerm
from torchani.utils import linspace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# We will use these coordinates and atomic nums throughout this example
coords = torch.tensor(
    [
        [
            [0.03192167, 0.00638559, 0.01301679],
            [-0.83140486, 0.39370209, -0.26395324],
            [-0.66518241, -0.84461308, 0.20759389],
            [0.45554739, 0.54289633, 0.81170881],
            [0.66091919, -0.16799635, -0.91037834],
        ]
    ],
    device=device,
)
species = torch.tensor([[1, 0, 0, 0, 0]], device=device)
# %%
# First lets create an AEVComputer in the 2x style, for comparison:
aevcomp = AEVComputer.like_2x().to(device)
aevs = aevcomp(species, coords)
radial_len = aevcomp.radial_len
# %%
# Now we calculate some AEVs
aevs = aevcomp(species, coords)
# %%
# The first 5 radial terms of the first atom:
aevs[0, 0, :5].tolist()
# %%
# The first 5 angular terms of the first atom:
radial_len = aevcomp.radial_len
aevs[0, 0, radial_len:radial_len + 5].tolist()
# %%
# Suppose we want to make one that is essentially the same, but using a different cutoff
# function, such as a smooth cutoff
#
# .. warning::
#
#   Be very careful, if a model has not been trained using this cutoff function
#   then using this aev computer with it will give nonsensical results
#
aevcomp_smooth = AEVComputer.like_1x(cutoff_fn="smooth").to(device)
radial_len = aevcomp_smooth.radial_len
aevs = aevcomp_smooth(species, coords)
radial_len = aevcomp_smooth.radial_len
# %%
# Now we calculate some AEVs
aevs = aevcomp_smooth(species, coords)
# %%
# The first 5 radial terms of the first atom:
aevs[0, 0, :5].tolist()
# %%
# The first 5 angular terms of the first atom:
radial_len = aevcomp_smooth.radial_len
aevs[0, 0, radial_len:radial_len + 5].tolist()
# %%
# Lets say now we want to experiment with a different cutoff function, such as a
# biweight cutoff.
#
# .. warning::
#
#   biweight does not have a continuous second derivative at the cutoff value, this may
#   not be appropriate for your model
#
#
# Since biweight is not coded in TorchANI we can code it ourselves and pass it
# to the AEVComputer, as long as the forward method has this form, it will work!
#
# The same cutoff function will be used for both radial and angular terms


class CutoffBiweight(Cutoff):
    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        # Assuming all elements in distances are smaller than cutoff
        return (cutoff**2 - distances**2) ** 2 / cutoff**4


cutoff_fn_biw = CutoffBiweight()
aevcomp_biw = AEVComputer.like_1x(cutoff_fn=cutoff_fn_biw).to(device)
radial_len = aevcomp_biw.radial_len
# %%
# Now we calculate some AEVs
aevs = aevcomp_biw(species, coords)
# %%
# The first 5 radial terms of the first atom:
aevs[0, 0, :5].tolist()
# %%
# The first 5 angular terms of the first atom:
radial_len = aevcomp_biw.radial_len
aevs[0, 0, radial_len:radial_len + 5].tolist()
# %%
# Lets try something a bit more complicated. Lets experiment with different angular
# terms that have a form of ``exp(-gamma * (cos(theta) - cos(theta0))**2)``. How can we
# do that?
#
# We can pass a custom module to the ``AEVComputer``. As long as it has the necessary
# attributes and methods (it has to have a *dim_out*, a *cutoff*, a *cutoff_fn* and a
# *compute_terms*)


class CosAngular(AngularTerm):
    def __init__(self, eta, shifts, gamma, sections, cutoff, cutoff_fn="cosine"):
        super().__init__(cutoff=cutoff, cutoff_fn=cutoff_fn)  # *Must* be called
        assert len(sections) == len(gamma)
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("eta", torch.tensor([eta]))
        self.register_buffer("shifts", torch.tensor(shifts))
        self.register_buffer("sections", torch.tensor(sections))
        self.num_feats = len(shifts) * len(sections)  # *Must* have an num_feats

    # The inputs are two tensors, of shapes (triples,) and (triples, 3)
    def compute_terms(self, triple_distances: Tensor, triple_vectors: Tensor) -> Tensor:
        triple_vectors = triple_vectors.view(2, -1, 3, 1, 1)
        triple_distances = triple_distances.view(2, -1, 1, 1)
        cos_angles = triple_vectors.prod(0).sum(1) / torch.clamp(
            triple_distances.prod(0), min=1e-10
        )
        factor1 = triple_distances.sum(0) / 2 - self.shifts.view(-1, 1)
        factor2 = cos_angles - torch.cos(self.sections.view(1, -1))
        exponent = self.eta * factor1**2 + self.gamma.view(1, -1) * factor2**2
        # *Must* output with ``(triples, self.num_feats)``
        return (4 * torch.exp(-exponent)).view(-1, self.num_feats)


# %%
# Now lets initialize the module, since we will use our custom terms in some AEVs
eta = 8.0
cutoff = 3.5
shifts = [0.9000, 1.5500, 2.2000, 2.8500]
sections = linspace(0.0, math.pi, 9)
gamma = [1023.0, 146.5, 36.0, 18.6, 15.5, 18.6, 36.0, 146.5, 1023.0]
cos_angular = CosAngular(eta, shifts, gamma, sections, cutoff, cutoff_fn="smooth")
# %%
# For the radial part we use the standard ANI-1x terms, with the same cutoff function
ani_radial = ANIRadial.like_1x(cutoff_fn="smooth")
aevcomp_cos = AEVComputer(radial=ani_radial, angular=cos_angular, num_species=4).to(
    device
)
# %%
# Now we calculate some AEVs
aevs = aevcomp_cos(species, coords)
# %%
# The first 5 radial terms of the first atom:
aevs[0, 0, :5].tolist()
# %%
# The first 5 angular terms of the first atom:
radial_len = aevcomp_cos.radial_len
aevs[0, 0, radial_len:radial_len + 5].tolist()
