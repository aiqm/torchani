import os
import subprocess
from setuptools import setup, find_packages

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    raise RuntimeError("Please install PyTorch before installing TorchANI.")

with open("README.md", "r") as fh:
    long_description = fh.read()

def cuda_extension():
    nvcc_args = ["-gencode=arch=compute_50,code=sm_50", "-gencode=arch=compute_60,code=sm_60",
                 "-gencode=arch=compute_61,code=sm_61", "-gencode=arch=compute_70,code=sm_70",
                 "-Xptxas=-v", '--expt-extended-lambda', '-use_fast_math']
    include_dirs = []
    cuda_version = float(torch.version.cuda)
    if cuda_version >= 10:
        nvcc_args.append("-gencode=arch=compute_75,code=sm_75")
    if cuda_version >= 11:
        nvcc_args.append("-gencode=arch=compute_80,code=sm_80")
    if cuda_version >= 11.1:
        nvcc_args.append("-gencode=arch=compute_86,code=sm_86")
    if cuda_version < 11:
        # if no cub, download it to include dir from github
        if not os.path.isdir('./include/cub'):
            if not os.path.exists('./include'):
                os.makedirs('include')
            commands = """
            echo "Downloading CUB library";
            wget -q https://github.com/NVIDIA/cub/archive/main.zip;
            unzip -q main.zip -d include;
            mv include/cub-main/cub include;
            echo "Removing unnecessary files";
            rm main.zip;
            rm -rf include/cub-main;
            """
            subprocess.run(commands, shell=True, check=True, universal_newlines=True)
        include_dirs.append(os.path.abspath("./include"))
    return CUDAExtension(
            name='torchani.cuaev._real_cuaev',
            sources=['torchani/cuaev/aev.cu'],
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-std=c++14'],
                                'nvcc': nvcc_args})

setup(
    name='torchani',
    description='PyTorch implementation of ANI',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/aiqm/torchani',
    author='Xiang Gao',
    author_email='qasdfgtyuiop@gmail.com',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[
        'torch',
        'lark-parser',
        'requests'
    ],
    ext_modules=[
        cuda_extension()
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
