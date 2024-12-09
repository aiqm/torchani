import math
import itertools
import typing as tp

import torch

try:
    import matplotlib.pyplot as plt

    _MPL_AVAIL = True
except ImportError:
    _MPL_AVAIL = False

from torchani.units import HARTREE_TO_EV, HARTREE_TO_KCALPERMOL, ANGSTROM_TO_BOHR
from torchani.constants import ATOMIC_NUMBER
from torchani.potentials.core import PairPotential


def plot(
    pot: PairPotential,
    title: str = "",
    symbol_pairs: tp.Sequence[tp.Tuple[str, str]] = (),
    xmin: float = 0.1,
    xmax: tp.Optional[float] = None,
    ymin: tp.Optional[float] = None,
    ymax: tp.Optional[float] = None,
    steps: int = 1000,
    force: bool = False,
    eunits: str = "hartree",
    runits: str = "angstrom",
    ylog: bool = False,
    block: bool = True,
) -> None:
    if not _MPL_AVAIL:
        raise RuntimeError("Please install matplotlib to plot this potential")
    efactors = {
        "ev": HARTREE_TO_EV,
        "kcalpermol": HARTREE_TO_KCALPERMOL,
        "hartree": 1,
    }
    rfactors = {
        "angstrom": 1,
        "bohr": ANGSTROM_TO_BOHR,
    }
    efactor = efactors.get(eunits.lower(), None)
    rfactor = rfactors.get(runits.lower(), None)
    if not title:
        title = pot.__class__.__name__
    if efactor is None:
        raise ValueError(
            f"Unsupported unit {eunits}. Supported are {set(efactors.keys())}"
        )
    if rfactor is None:
        raise ValueError(
            f"Unsupported unit {runits}. Supported are {set(rfactors.keys())}"
        )
    if not symbol_pairs:
        symbol_pairs = tuple(itertools.combinations_with_replacement(pot.symbols, 2))
    fig, ax = plt.subplots()
    if xmax is None:
        xmax = pot.cutoff if pot.cutoff != math.inf else 10.0
    r = torch.linspace(xmin, xmax, steps) * rfactor
    for pair in symbol_pairs:
        atomic_nums = torch.zeros((steps, 2), dtype=torch.long)
        atomic_nums[:, 0] = ATOMIC_NUMBER[pair[0]]
        atomic_nums[:, 1] = ATOMIC_NUMBER[pair[1]]
        coords = torch.zeros((steps, 2, 3))
        if force:
            r = r.detach().requires_grad_(True)
        coords[:, 0, 0] = r
        energies = pot(atomic_nums, coords) * efactor
        if force:
            forces = -torch.autograd.grad(energies.sum(), r)[0]
            r.detach_()
            energies.detach_()
            ax.plot(r, forces, label=f"{pair[0]}-{pair[1]}")
        else:
            ax.plot(r, energies, label=f"{pair[0]}-{pair[1]}")
    ax.legend()
    if title != "no":
        ax.set_title(title)
    runit_sym = {
        "angstrom": r"\AA",
        "bohr": r"a_0",
    }[runits.lower()]
    ax.set_xlabel(r"Inter atomic distance, $\left("f"{runit_sym}"r"\right)$")
    eunit_sym = {
        "hartree": r"E_h",
        "ev": r"\mathrm{eV}",
        "kcalpermol": r"\text{kcal}/\text{mol}",
    }[eunits.lower()]
    ax.set_ylabel(r"Energy, $\left("f"{eunit_sym}"r"\right)$")
    if force:
        ax.set_ylabel(r"Force, $\left("f"{eunit_sym}/{runit_sym}"r"\right)$")
    if ylog:
        ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    plt.show(block=block)
