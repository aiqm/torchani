r"""Some utilities for extracting information from neurochem files"""
import torch
import typing_extensions as tpx
from dataclasses import dataclass
import typing as tp
import zipfile
import shutil
from pathlib import Path

from torchani.aev import AEVComputer
from torchani.nn import Ensemble, ANIModel
from torchani.models import BuiltinModel
from torchani.storage import NEUROCHEM_DIR
from torchani.potentials import EnergyAdder
from torchani.neurochem.utils import model_dir_from_prefix
from torchani.neurochem.neurochem import (
    load_aev_computer_and_symbols,
    load_model_ensemble,
    load_model,
    load_energy_adder,
)


__all__ = [
    "modules_from_builtin_name",
    "modules_from_info_file",
    "download_model_parameters",
    "load_builtin_from_name",
    "load_builtin_from_info_file",
]


SUPPORTED_MODELS = {"ani1x", "ani2x", "ani1ccx"}

NN = tp.Union[ANIModel, Ensemble]


@dataclass
class NeurochemInfo:
    sae: Path
    const: Path
    ensemble_prefix: Path
    ensemble_size: int

    @classmethod
    def from_info_file(cls, info_file_path: Path) -> tpx.Self:
        with open(info_file_path, mode="rt", encoding="utf-8") as f:
            lines: tp.List[str] = [x.strip() for x in f.readlines()][:4]
            _const_file, _sae_file, _ensemble_prefix, _ensemble_size = lines

            ensemble_size: int = int(_ensemble_size)
            const_file_reldir, const_file_name = _const_file.split("/")
            sae_file_reldir, sae_file_name = _sae_file.split("/")
            ensemble_prefix_reldir, ensemble_prefix_name = _ensemble_prefix.split("/")

            const_file_path: Path = (
                NEUROCHEM_DIR / const_file_reldir
            ) / const_file_name
            sae_file_path: Path = (NEUROCHEM_DIR / sae_file_reldir) / sae_file_name
            ensemble_prefix: Path = (
                NEUROCHEM_DIR / ensemble_prefix_reldir
            ) / ensemble_prefix_name
        return cls(sae_file_path, const_file_path, ensemble_prefix, ensemble_size)

    @classmethod
    def from_builtin_name(cls, model_name: str) -> tpx.Self:
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Neurochem model {model_name} not supported,"
                f" supported models are: {SUPPORTED_MODELS}",
            )
        suffix = model_name.replace("ani", "")
        info_file_path = NEUROCHEM_DIR / f"ani-{suffix}_8x.info"
        if not info_file_path.is_file():
            download_model_parameters()
        info = cls.from_info_file(info_file_path)
        return info


def download_model_parameters(
    root: tp.Optional[Path] = None, verbose: bool = True
) -> None:
    if root is None:
        root = NEUROCHEM_DIR
    zip_path = root / "neurochem-builtins.zip"
    if any(root.iterdir()):
        if verbose:
            print("Found existing files in directory, assuming params already present")
        return
    repo = "ani-model-zoo"
    tag = "ani-2x"
    extracted_dirname = f"{repo}-{tag}"
    url = f"https://github.com/aiqm/{repo}/archive/{tag}.zip"
    if verbose:
        print("Downloading ANI model parameters ...")
    torch.hub.download_url_to_file(url, str(zip_path), progress=verbose)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    zip_path.unlink()
    extracted_dir = Path(root) / extracted_dirname
    for f in (extracted_dir / "resources").iterdir():
        shutil.move(str(f), root / f.name)
    shutil.rmtree(extracted_dir)


def modules_from_info(
    info: NeurochemInfo,
    model_index: tp.Optional[int] = None,
    use_cuda_extension: bool = False,
    use_cuaev_interface: bool = False,
) -> tp.Tuple[AEVComputer, NN, EnergyAdder, tp.Sequence[str]]:
    aev_computer, symbols = load_aev_computer_and_symbols(
        info.const,
        use_cuda_extension=use_cuda_extension,
        use_cuaev_interface=use_cuaev_interface,
    )
    adder = load_energy_adder(info.sae)

    neural_networks: NN
    if model_index is None:
        neural_networks = load_model_ensemble(
            symbols, info.ensemble_prefix, info.ensemble_size
        )
    else:
        if model_index >= info.ensemble_size:
            raise ValueError(
                f"Model index {model_index} should be <= {info.ensemble_size}"
            )
        neural_networks = load_model(
            symbols,
            model_dir_from_prefix(
                info.ensemble_prefix,
                model_index,
            ),
        )
    return aev_computer, neural_networks, adder, symbols


def modules_from_builtin_name(
    model_name: str,
    model_index: tp.Optional[int] = None,
    use_cuda_extension: bool = False,
    use_cuaev_interface: bool = False,
) -> tp.Tuple[AEVComputer, NN, EnergyAdder, tp.Sequence[str]]:
    r"""
    Creates the necessary modules to generate a pretrained builtin model,
    parsing the data from legacy neurochem files. and optional arguments to
    modify some modules.
    """
    return modules_from_info(
        NeurochemInfo.from_builtin_name(model_name),
        model_index,
        use_cuda_extension,
        use_cuaev_interface,
    )


def modules_from_info_file(
    info_file: Path,
    model_index: tp.Optional[int] = None,
    use_cuda_extension: bool = False,
    use_cuaev_interface: bool = False,
) -> tp.Tuple[AEVComputer, NN, EnergyAdder, tp.Sequence[str]]:
    r"""
    Creates the necessary modules to generate a pretrained neurochem model,
    parsing the data from legacy neurochem files. and optional arguments to
    modify some modules.
    """
    return modules_from_info(
        NeurochemInfo.from_info_file(info_file),
        model_index,
        use_cuda_extension,
        use_cuaev_interface,
    )


def load_builtin_from_info_file(
    info_file: tp.Union[Path, str],
    model_index: tp.Optional[int] = None,
    use_cuda_extension: bool = False,
    use_cuaev_interface: bool = False,
    periodic_table_index: bool = True,
) -> BuiltinModel:
    info_file = Path(info_file).resolve()
    components = modules_from_info_file(
        info_file,
        model_index,
        use_cuda_extension,
        use_cuaev_interface,
    )
    aev_computer, neural_networks, energy_adder, elements = components
    return BuiltinModel(
        aev_computer,
        neural_networks,
        energy_adder,
        elements,
        periodic_table_index=periodic_table_index,
    )


def load_builtin_from_name(
    model_name: str,
    model_index: tp.Optional[int] = None,
    use_cuda_extension: bool = False,
    use_cuaev_interface: bool = False,
    periodic_table_index: bool = True,
) -> BuiltinModel:
    components = modules_from_builtin_name(
        model_name,
        model_index,
        use_cuda_extension,
        use_cuaev_interface,
    )
    aev_computer, neural_networks, energy_adder, elements = components
    return BuiltinModel(
        aev_computer,
        neural_networks,
        energy_adder,
        elements,
        periodic_table_index=periodic_table_index,
    )
