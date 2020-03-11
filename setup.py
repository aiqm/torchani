from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


with open("README.md", "r") as fh:
    long_description = fh.read()


class _PrebuildInstall(install):
    # Custom setup command to prebuild models
    # This command prebuilds all builtin ANI models when the package is
    # installed  It is run only in non develop installations

    def run(self):
        super().run()
        import torchani
        try:
            torchani.models.prebuild_models()
        except Exception as e:
            print(e)
            # if pickling fails we don't do anything
            # pip swallows output but this gets printed with
            # a raw `python setup.py install` call


class _PrebuildDevelop(develop):
    # Custom setup command to prebuild models
    # This command prebuilds all builtin ANI models when the package is
    # installed  It is run only in develop installations

    def initialize_options(self):
        super().initialize_options()
        self.no_compilation = False

    def run(self):
        super().run()
        import torchani
        try:
            torchani.models.prebuild_models()
        except Exception as e:
            print(e)
            # if pickling fails we don't do anything
            # pip swallows output but this gets printed with
            # a raw `python setup.py develop` call


setup_attrs = {
    'cmdclass': {'install': _PrebuildInstall, 'develop': _PrebuildDevelop},
    'name': 'torchani',
    'description': 'PyTorch implementation of ANI',
    'long_description': long_description,
    'long_description_content_type': "text/markdown",
    'url': 'https://github.com/aiqm/torchani',
    'author': 'Xiang Gao',
    'author_email': 'qasdfgtyuiop@gmail.com',
    'license': 'MIT',
    'packages': find_packages(),
    'include_package_data': True,
    'use_scm_version': True,
    'setup_requires': ['setuptools_scm'],
    'install_requires': [
        'torch',
        'lark-parser',
    ],
}

setup(**setup_attrs)
