from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


with open("README.md", "r") as fh:
    long_description = fh.read()

class _JITCompileInstall(install):
    # Custom setup command to JIT compile models
    # This command JIT-compiles all builtin ANI models when the package is
    # installed  It is run only in non develop installations

    user_options = install.user_options + [('no-compilation', None, "Installs without compiling the builtin models")]

    def initialize_options(self):
        super().initialize_options()
        self.no_compilation = False

    def run(self):
        if self.no_compilation:
            return
        install.run(self)
        import torchani
        torchani.models.prebuild_models()


class _JITCompileDevelop(develop):
    # Custom setup command to JIT compile models
    # This command JIT-compiles all builtin ANI models when the package is
    # installed  It is run only in develop installations

    user_options = develop.user_options + [('no-compilation', None, "Installs without compiling the builtin models")]

    def initialize_options(self):
        super().initialize_options()
        self.no_compilation = False

    def run(self):
        develop.run(self)
        if self.no_compilation:
            return
        import torchani
        torchani.models.prebuild_models()


setup_attrs = {
    'cmdclass': {'install': _JITCompileInstall, 'develop': _JITCompileDevelop},
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
