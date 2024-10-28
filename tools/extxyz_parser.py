r"""This tool parsers files in the extxyz format, like the ones in the 3BPA dataset
and converts them into .h5 files

Currently it is fairly specific for 3BPA but can be easily modified.
"""

import numpy as np
from pathlib import Path

from torchani.units import HARTREE_TO_EV
from torchani.datasets import ANIDataset


# All of the molecules are simulated with no pbc, so the lattice doesn't matter
keys = ["Lattice", "Properties", "energy", "dihedrals", "pbc"]

xyz_dir = Path("./3BPA")

for fpath in sorted(xyz_dir.rglob("*.xyz")):
    print("Processing", fpath.name)
    ds = ANIDataset(locations=f"./{fpath.name}.h5", create=True)
    with open(fpath, mode="rt", encoding="utf-8") as f:
        iterfile = iter(f)
        line = next(iterfile, None)
        if line is None:
            raise ValueError("File should have at least one line")
        while line is not None:
            atoms_num = int(line.strip())
            line = next(iterfile, None)
            if line is None:
                break
            parts = []
            parts_raw = line.split("=")
            for p in parts_raw:
                if '"' in p:
                    parts.extend(p.split('"'))
                else:
                    parts.extend(p.split())
            parts = [p for p in parts if p.strip()]
            kv_dict = {k.strip(): v for k, v in zip(parts[0::2], parts[1::2])}
            energy = float(kv_dict["energy"]) / HARTREE_TO_EV
            dihedrals = None
            forces = None
            if "dihedrals" in kv_dict:
                dihedrals_str = kv_dict["dihedrals"].split("[")[-1].replace("]", "")
                dihedrals = [float(d) for d in dihedrals_str.split(",")]
                assert len(dihedrals) == 3, "There should be 3 dihedrals if present"
            coordinates = []
            species = []
            for _ in range(atoms_num):
                line_parts = next(iterfile).split()
                assert len(line_parts) in [4, 7], "There should be 4 or 7 fields / line"
                species.append(line_parts[0])
                coordinates.append([float(c) for c in line_parts[1:4]])
                if "forces" in kv_dict["Properties"]:
                    forces = [float(frc) / HARTREE_TO_EV for frc in line_parts[5:8]]
            line = next(iterfile, None)
            num_atoms = str(len(species)).zfill(3)
            conformer = {
                "species": np.array([species]),
                "coordinates": np.array([coordinates]),
                "energies": np.array([energy]),
            }

            if forces is not None:
                conformer["forces"] = np.array([forces])

            if dihedrals is not None:
                conformer["dihedrals"] = np.array([dihedrals])
            ds.append_conformers(num_atoms, conformer)
