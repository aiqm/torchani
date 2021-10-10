r"""Some utilities for extracting information from neurochem files"""
import os
import io
import requests
import zipfile
from typing import Optional, Sequence, Tuple, Union, Dict, Any
from distutils import dir_util
from pathlib import Path
from ..aev import AEVComputer
from ..utils import EnergyShifter
from ..nn import Ensemble, ANIModel
from .neurochem import Constants, load_model_ensemble, load_model, load_sae


__all__ = ['parse_neurochem_resources']


SUPPORTED_INFO_FILES = ['ani-1ccx_8x.info', 'ani-1x_8x.info', 'ani-2x_8x.info']

NN = Union[ANIModel, Ensemble]


def parse_neurochem_resources(info_file_path):
    torchani_dir = Path(__file__).resolve().parent.parent.as_posix()
    resource_path = os.path.join(torchani_dir, 'resources/')
    print(resource_path)
    local_dir = os.path.expanduser('~/.local/torchani/')

    resource_info = os.path.join(resource_path, info_file_path)

    if os.path.isfile(resource_info) and os.stat(resource_info).st_size > 0:
        # No action needed if the info file can be located in the default path
        pass

    elif os.path.isfile(os.path.join(local_dir, info_file_path)):
        # if the info file is not located in the default path, ~/.local/torchani
        # is tried as an alternative
        resource_path = local_dir

    else:
        # if all else fails files are downloaded and extracted ONLY if a
        # correct info file path is passed, otherwise an error is raised
        if info_file_path in SUPPORTED_INFO_FILES:
            repo_name = "ani-model-zoo"
            tag_name = "ani-2x"
            extracted_name = '{}-{}'.format(repo_name, tag_name)
            url = "https://github.com/aiqm/{}/archive/{}.zip".format(repo_name, tag_name)

            print('Downloading ANI model parameters ...')
            resource_res = requests.get(url)
            resource_zip = zipfile.ZipFile(io.BytesIO(resource_res.content))
            try:
                resource_zip.extractall(resource_path)
            except (PermissionError, OSError):
                resource_zip.extractall(local_dir)
                resource_path = local_dir
            source = os.path.join(resource_path, extracted_name, "resources")
            dir_util.copy_tree(source, resource_path)
            dir_util.remove_tree(os.path.join(resource_path, extracted_name))

        else:
            raise ValueError(f'File {info_file_path} could not be found either in {resource_path} or {local_dir}\n'
                             'It is also not one of the supported builtin info files:'
                             ' {SUPPORTED_INFO_FILES}')

    return _get_resources(resource_path, info_file_path)


def _get_resources(resource_path, info_file):
    with open(os.path.join(resource_path, info_file)) as f:
        # const_file: Path to the file with the builtin constants.
        # sae_file: Path to the file with the Self Atomic Energies.
        # ensemble_prefix: Prefix of the neurochem resource directories.
        lines = [x.strip() for x in f.readlines()][:4]
        const_file_path, sae_file_path, ensemble_prefix_path, ensemble_size = lines
        const_file = os.path.join(resource_path, const_file_path)
        sae_file = os.path.join(resource_path, sae_file_path)
        ensemble_prefix = os.path.join(resource_path, ensemble_prefix_path)
        ensemble_size = int(ensemble_size)
    return const_file, sae_file, ensemble_prefix, ensemble_size


def _get_component_modules(info_file: str,
                           model_index: Optional[int] = None,
                           aev_computer_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[AEVComputer, NN, EnergyShifter, Sequence[str]]:
    # this creates modules from a neurochem info path,
    # since for neurochem architecture and parameters are kind of mixed up,
    # this doesn't support non pretrained models, it directly outputs a pretrained module
    if aev_computer_kwargs is None:
        aev_computer_kwargs = dict()
    const_file, sae_file, ensemble_prefix, ensemble_size = parse_neurochem_resources(info_file)
    consts = Constants(const_file)
    elements = consts.species
    aev_computer = AEVComputer(**consts, **aev_computer_kwargs)

    if model_index is None:
        neural_networks = load_model_ensemble(elements, ensemble_prefix, ensemble_size)
    else:
        if (model_index >= ensemble_size):
            raise ValueError(f"The ensemble size is only {ensemble_size}, model {model_index} can't be loaded")
        network_dir = os.path.join(f'{ensemble_prefix}{model_index}', 'networks')
        neural_networks = load_model(elements, network_dir)
    return aev_computer, neural_networks, load_sae(sae_file), elements
