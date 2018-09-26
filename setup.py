from setuptools import setup, find_packages

setup_attrs = {
    'name': 'torchani',
    'version': '0.1',
    'description': 'PyTorch implementation of ANI',
    'url': 'https://github.com/zasdfgbnm/torchani',
    'author': 'Xiang Gao',
    'author_email': 'qasdfgtyuiop@ufl.edu',
    'license': 'MIT',
    'packages': find_packages(),
    'include_package_data': True,
    'install_requires': [
        'torch',
        'pytorch-ignite',
        'lark-parser',
        'h5py',
    ],
    'test_suite': 'nose.collector',
    'tests_require': [
        'nose',
        'tensorboardX',
        'tqdm',
    ],
    'extras_require': {
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
            'sphinx-gallery',
            'pillow',
            'matplotlib',
            'sphinx_rtd_theme'
            'tqdm',
            'tensorboardX',
        ],
    },
}

setup(**setup_attrs)
