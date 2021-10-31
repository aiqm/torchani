import os
import subprocess
from setuptools import setup, find_packages
from distutils import log
import sys

BUILD_CUAEV_ALL_SM = '--cuaev-all-sms' in sys.argv
if BUILD_CUAEV_ALL_SM:
    sys.argv.remove('--cuaev-all-sms')

FAST_BUILD_CUAEV = '--cuaev' in sys.argv
if FAST_BUILD_CUAEV:
    sys.argv.remove('--cuaev')

# Use along with --cuaev for CI test to reduce compilation time on Non-GPUs system
ONLY_BUILD_SM80 = '--only-sm80' in sys.argv
if ONLY_BUILD_SM80:
    sys.argv.remove('--only-sm80')

if not BUILD_CUAEV_ALL_SM and not FAST_BUILD_CUAEV:
    log.warn("Will not install cuaev")  # type: ignore

with open("README.md", "r") as fh:
    long_description = fh.read()


def maybe_download_cub():
    import torch
    dirs = torch.utils.cpp_extension.include_paths(cuda=True)
    for d in dirs:
        cubdir = os.path.join(d, 'cub')
        log.info(f'Searching for cub at {cubdir}...')
        if os.path.isdir(cubdir):
            log.info(f'Found cub in {cubdir}')
            return []
    # if no cub, download it to include dir from github
    if not os.path.isdir('./include/cub'):
        if not os.path.exists('./include'):
            os.makedirs('include')
        commands = """
        echo "Downloading CUB library";
        wget -q https://github.com/NVIDIA/cub/archive/refs/tags/1.11.0.zip;
        unzip -q 1.11.0.zip -d include;
        mv include/cub-1.11.0/cub include;
        echo "Removing unnecessary files";
        rm 1.11.0.zip;
        rm -rf include/cub-1.11.0;
        """
        subprocess.run(commands, shell=True, check=True, universal_newlines=True)
    return [os.path.abspath("./include")]


def cuda_extension(build_all=False):
    import torch
    from torch.utils.cpp_extension import CUDAExtension
    SMs = []
    print('-' * 75)
    if not build_all:
        devices = torch.cuda.device_count()
        print('FAST_BUILD_CUAEV: ON')
        print('This build will only support the following devices or the devices with same cuda capability: ')
        for i in range(devices):
            d = 'cuda:{}'.format(i)
            sm = torch.cuda.get_device_capability(i)
            sm = int(f'{sm[0]}{sm[1]}')
            if sm >= 50:
                print('{}: {}'.format(i, torch.cuda.get_device_name(d)))
                print('   {}'.format(torch.cuda.get_device_properties(i)))
            if sm not in SMs and sm >= 50:
                SMs.append(sm)

    nvcc_args = ["-Xptxas=-v", '--expt-extended-lambda', '-use_fast_math']
    if SMs and not ONLY_BUILD_SM80:
        for sm in SMs:
            nvcc_args.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")
    elif ONLY_BUILD_SM80:  # --cuaev --only-sm80
        nvcc_args.append("-gencode=arch=compute_80,code=sm_80")
    else:  # no gpu detected
        print('Will build for all SMs')
        nvcc_args.append("-gencode=arch=compute_60,code=sm_60")
        nvcc_args.append("-gencode=arch=compute_61,code=sm_61")
        nvcc_args.append("-gencode=arch=compute_70,code=sm_70")
        cuda_version = float(torch.version.cuda)
        if cuda_version >= 10:
            nvcc_args.append("-gencode=arch=compute_75,code=sm_75")
        if cuda_version >= 11:
            nvcc_args.append("-gencode=arch=compute_80,code=sm_80")
        if cuda_version >= 11.1:
            nvcc_args.append("-gencode=arch=compute_86,code=sm_86")
    print("nvcc_args: ", nvcc_args)
    print('-' * 75)
    include_dirs = [*maybe_download_cub(), os.path.abspath("torchani/cuaev/")]
    return CUDAExtension(
        name='torchani.cuaev',
        pkg='torchani.cuaev',
        sources=["torchani/cuaev/cuaev.cpp", "torchani/cuaev/aev.cu"],
        include_dirs=include_dirs,
        extra_compile_args={'cxx': ['-std=c++14'], 'nvcc': nvcc_args})


def cuaev_kwargs():
    if not BUILD_CUAEV_ALL_SM and not FAST_BUILD_CUAEV:
        return dict(
            provides=['torchani']
        )
    from torch.utils.cpp_extension import BuildExtension
    kwargs = dict(
        provides=[
            'torchani',
            'torchani.cuaev',
        ],
        ext_modules=[
            cuda_extension(BUILD_CUAEV_ALL_SM)
        ],
        cmdclass={
            'build_ext': BuildExtension,
        })
    return kwargs


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
        'requests',
        'importlib_metadata',
    ],
    **cuaev_kwargs()
)
