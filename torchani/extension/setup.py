import os
import torch
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc_args = ["-gencode=arch=compute_50,code=sm_50", "-gencode=arch=compute_60,code=sm_60",
             "-gencode=arch=compute_61,code=sm_61", "-gencode=arch=compute_70,code=sm_70",
             "-Xptxas=-v", '--expt-extended-lambda', '-use_fast_math']
include_dirs = []
cuda_version = float(torch.version.cuda)
if cuda_version >= 10:
    nvcc_args.append("-gencode=arch=compute_75,code=sm_75")
elif cuda_version >= 11:
    nvcc_args.append("-gencode=arch=compute_80,code=sm_80")
elif cuda_version >= 11.1:
    nvcc_args.append("-gencode=arch=compute_86,code=sm_86")
else:
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

setup(
    name='cuaev',
    version='0.1',
    ext_modules=[
        CUDAExtension(
            name='cuaev',
            sources=['aev.cu'],
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-std=c++11'],
                                'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
