from setuptools import setup
from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}

setup(name='torchani',
      version='0.1',
      description='ANI based on pytorch',
      url='https://github.com/zasdfgbnm/torchani',
      author='Xiang Gao',
      author_email='qasdfgtyuiop@ufl.edu',
      license='MIT',
      packages=['torchani'],
      include_package_data=True,
      install_requires=[
          'torch',
          'lark-parser',
          'h5py',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'cntk'],
      cmdclass=cmdclass,
      )
