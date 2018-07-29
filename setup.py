from setuptools import setup
from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}

setup(name='torchani',
      version='0.1',
      description='PyTorch implementation of ANI',
      url='https://github.com/zasdfgbnm/torchani',
      author='Xiang Gao',
      author_email='qasdfgtyuiop@ufl.edu',
      license='MIT',
      packages=['torchani'],
      include_package_data=True,
      install_requires=[
          'torch',
          'pytorch-ignite',
          'lark-parser',
          'h5py',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      cmdclass=cmdclass,
      )
