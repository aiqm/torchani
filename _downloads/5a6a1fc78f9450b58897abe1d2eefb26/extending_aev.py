r"""
Extending the local atomic features: AEVs with custom terms and cutoffs
=======================================================================

TorchANI allows for simple customization of the AEV features. This is an important
extension point of the library. Here we explain how to implement your own custom AEV
terms and cutoff functions.
"""

# To begin with, let's first import the modules and setup devices we will use:
import torch

from torchani.cutoffs import Cutoff
from torchani.aev import AEVComputer, Angular, Radial

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
# biweight cutoff. Since biweight is not coded in TorchANI we can code it ourselves and
# pass it to the AEVComputer, as long as the forward method has this form, it will work!
#
# The same cutoff function will be used for both radial and angular terms


class CutoffBiweight(Cutoff):
    def forward(self, distances, cutoff):
        # The calculation can assume all distances passed to be smaller than the cutoff
        return (cutoff**2 - distances**2) ** 2 / cutoff**4


biweight = CutoffBiweight()
aevcomp_biweight = AEVComputer.like_1x(cutoff_fn=biweight).to(device)
radial_len = aevcomp_biweight.radial_len
# %%
# Now we calculate some AEVs
aevs = aevcomp_biweight(species, coords)
# %%
# The first 5 radial terms of the first atom:
aevs[0, 0, :5].tolist()
# %%
# The first 5 angular terms of the first atom:
radial_len = aevcomp_biweight.radial_len
aevs[0, 0, radial_len:radial_len + 5].tolist()
# %%
# Lets try something a bit more complicated. Lets experiment with different 2-body and
# 3-body terms. Our 3-body terms will include a term ``exp(-eta_a * (cos(theta) -
# cos_phi)**2)``, and our 2-body terms will be lorentzians, with the form ``1 / (1 +
# x**2)``, where ``x = ((r - shifts) / fwhm)``. How can we do that?
#
# We can pass custom modules to the ``AEVComputer``. The easiest
# way to code custom modules is, for the 2-body part, by subclassing ``Radial``,
# which can be used to calculate terms of the form ``R(r_ij) * fcut(r_ij)``, where
# ``i, j`` is a pair of neighbors.


class Lorentzian(Radial):
    tensors = ["shifts", "fwhm"]  # Tensors we will use. fwhm = Full Width at Half Max

    def compute(self, distances):
        x = 2 * (distances - self.shifts) / self.fwhm
        return 1 / (1 + x**2)


# %%
# And for the 3-body part, by subclassing ``Angular``, which calculates terms of the
# form ``R(r_ij, r_ik) * A(cos(theta_ijk)) * fcut(r_ij) * fcut(r_ik)``,
# where ``i, j, k`` is a triple consisting on two pairs of neighbors
# that share one atom in common.


class ExpCosine(Angular):
    angles_tensors = ["cos_phi", "eta_a"]  # Tensors we will use in A(cos(theta_ijk))
    radial_tensors = ["shifts", "eta_r"]  # Tensors we will use in R(r_ij, r_ik)

    def compute_cos_angles(self, cos_angles):
        return 2 * torch.exp(-self.eta_a * (cos_angles - self.cos_phi) ** 2)

    def compute_radial(self, distances_ji, distances_jk):
        mean_dists = (distances_ji + distances_jk) / 2
        return 2 * torch.exp(-self.eta_r * (mean_dists - self.shifts) ** 2)


# %%
# Now lets initialize the angular module with constants
custom_3body = ExpCosine(
    eta_r=8.0,
    shifts=[0.9000, 1.5500, 2.2000, 2.8500],
    eta_a=[1023.0, 146.5, 36.0, 18.6, 15.5, 18.6, 36.0, 146.5, 1023.0],
    cos_phi=[1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0],
    cutoff=3.5,
    cutoff_fn="smooth",
)
# %%
# For the 3-body module, we want to make the shifts trainable, which is supported.
# if we wanted to make both ``fwhm`` and ``shifts`` trainable we could use
# ``trainable=["shifts", "fwhm"]``
custom_2body = Lorentzian(
    fwhm=1.5,
    shifts=[0.0, 1.0, 2.0, 3.0, 4.0],
    trainable="shifts",
    cutoff=5.2,
    cutoff_fn="smooth",
)
# %%
# Finally we create our custom AEVComputer, which will use the specified terms
custom_aev = AEVComputer(radial=custom_2body, angular=custom_3body, num_species=4).to(
    device
)
# %%
# Now we calculate some AEVs
aevs = custom_aev(species, coords)
# %%
# The first 5 radial terms of the first atom:
aevs[0, 0, :5].tolist()
# %%
# The first 5 angular terms of the first atom:
radial_len = custom_aev.radial_len
aevs[0, 0, radial_len:radial_len + 5].tolist()
