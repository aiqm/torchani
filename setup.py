from setuptools import setup

setup(name='torchani',
      version='0.1',
      description='ANI based on pytorch',
      url='https://github.com/zasdfgbnm/torchani',
      author='Xiang Gao',
      author_email='qasdfgtyuiop@ufl.edu',
      license='MIT',
      packages=['torchani'],
      install_requires=[
          'pytorch',
          'lark-parser',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
