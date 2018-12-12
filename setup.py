from setuptools import setup, find_packages

setup_attrs = {
    'name': 'torchani',
    'description': 'PyTorch implementation of ANI',
    'url': 'https://github.com/zasdfgbnm/torchani',
    'author': 'Xiang Gao',
    'author_email': 'qasdfgtyuiop@ufl.edu',
    'license': 'MIT',
    'packages': find_packages(),
    'include_package_data': True,
    'use_scm_version': True,
    'setup_requires': ['setuptools_scm'],
    'install_requires': [
        'torch-nightly',
        'pytorch-ignite-nightly',
        'lark-parser',
        'h5py',
    ],
    'test_suite': 'nose.collector',
    'tests_require': [
        'nose',
        'tensorboardX',
        'tqdm',
        'ase',
        'coverage',
    ],
}

setup(**setup_attrs)
