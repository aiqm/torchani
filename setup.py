from setuptools import setup, find_packages
import sys

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
        'lark-parser',
    ],
    'test_suite': 'nose.collector',
    'tests_require': [
        'nose',
        'tb-nightly',
        'tqdm',
        'ase',
        'coverage',
        'h5py',
        'pytorch-ignite-nightly',
        'pillow',
    ],
}

if sys.version_info[0] < 3:
    setup_attrs['install_requires'].append('typing')

setup(**setup_attrs)
