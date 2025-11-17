r"""Example: ANI with TI (1)"""
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

species = torch.tensor([[6, 8, 1, 1, 17]])
coords = torch.tensor(
    [
        [
            [-0.0398460080, -0.0054042378, -0.0006821430],
            [1.1699090417, -0.1651235922, 0.0390627287],
            [-0.7099690966, -0.8354632204, 0.0312009563],
            [-0.4200939371, 1.0059910505, -0.0695815420],
            [-0.4200939371, 1.0059910505, -0.0695815420],
        ]
    ]
)

disappearing_idxs = torch.tensor([[3]])
appearing_idxs = torch.tensor([[4]])
energies = []
atomic_energies_list = []
factors = torch.linspace(0.0, 1.0, 100)
for ti_factor in factors:
    energy = ani_for_ti.forward_for_ti(
        (species, coords), ti_factor, appearing_idxs, disappearing_idxs
    ).energies
    _atomic_energies = ani_for_ti.forward_for_ti(
        (species, coords), ti_factor, appearing_idxs, disappearing_idxs, atomic=True,
    ).energies
    atomic_energies_list.append(_atomic_energies)
    energies.append(energy.item())

atomic_energies = torch.cat(atomic_energies_list)
initial_energy = ani((species[:, :-1], coords[:, :-1, :])).energies.item()
final_energy = ani(
    (species[:, [0, 1, 2, 4]], coords[:, [0, 1, 2, 4], :])
).energies.item()

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
