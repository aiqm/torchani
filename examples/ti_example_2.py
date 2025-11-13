r"""Example: ANI with TI (2)"""
import matplotlib.pyplot as plt
import torch

from torchani.arch import ANIForThermoIntegration
from torchani.models import ANI2x


ani = ANI2x(model_index=0)
ani_for_ti = ANIForThermoIntegration.from_ani_model(ani)
sd = ani.state_dict()

for k, v in ani_for_ti.state_dict().items():
    assert (v == sd[k]).all()

ani.set_enabled("energy_shifter", False)
ani_for_ti.set_enabled("energy_shifter", False)

# Conversion of phenol to aniline
species = torch.tensor([[6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 7, 1, 1, 8]])
coords = torch.tensor(
    [
        [
            [0.1408100333, -1.2275420474, 0.0160044789],
            [-1.2353142293, -1.1510573197, 0.1470239006],
            [-1.8484321051, 0.1027855831, 0.1754650033],
            [-1.1225300501, 1.2742044472, 0.0766157782],
            [0.2441355629, 1.1743221169, -0.0529162220],
            [0.8735301726, -0.0476325800, -0.0837116321],
            [0.6249152770, -2.2059735227, -0.0067353961],
            [-1.8223970925, -2.0306071606, 0.2253027469],
            [-2.9408946708, 0.1939139324, 0.2787325748],
            [-1.6529026367, 2.2275638569, 0.1044388535],
            [0.8234554405, 2.0972194106, -0.1315109278],
            [2.2809722426, -0.1180800932, -0.2179042604],
            [2.9043523517, -0.0572384387, 0.6332709107],
            [2.7302997038, -0.2318781849, -1.1640758085],
            [2.2809722426, -0.1180800932, -0.2179042604],
        ]
    ]
)

disappearing_idxs = torch.tensor([[11, 12]])
appearing_idxs = torch.tensor([[14]])

energies = []
atomic_energies_list = []
factors = torch.linspace(0.0, 1.0, 100)
for ti_factor in factors:
    energy = ani_for_ti.forward_for_ti(
        (species, coords), ti_factor, appearing_idxs, disappearing_idxs
    ).energies
    _atomic_energies = ani_for_ti.forward_for_ti(
        (species, coords),
        ti_factor,
        appearing_idxs,
        disappearing_idxs,
        atomic=True,
    ).energies
    atomic_energies_list.append(_atomic_energies)
    energies.append(energy.item())

atomic_energies = torch.cat(atomic_energies_list)
initial_idxs = [
    i for i in range(coords.size(1)) if i not in appearing_idxs.view(-1).tolist()
]
initial_energy = ani(
    (species[:, initial_idxs], coords[:, initial_idxs, :])
).energies.item()
final_idxs = [
    i for i in range(coords.size(1)) if i not in disappearing_idxs.view(-1).tolist()
]
final_energy = ani((species[:, final_idxs], coords[:, final_idxs, :])).energies.item()

fig, ax = plt.subplots()
ax.plot(factors.cpu().numpy(), energies)
for e, znum in zip(atomic_energies.T, species.view(-1).tolist()):
    ax.plot(factors.cpu().numpy(), e.cpu().numpy(), label=f"Atomic number: {znum}")

ax.legend()
ax.axhline(initial_energy, color="k", linestyle="dashed")
ax.axhline(final_energy, color="k", linestyle="dashed")
ax.set_xlabel(r"TI $\lambda$")
ax.set_ylabel(r"Energy (Ha)")
plt.show()
