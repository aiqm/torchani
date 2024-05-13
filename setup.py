import os
import subprocess
import sys
import logging

from setuptools import setup, find_packages

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("setup")


BUILD_EXT_ALL_SM = '--ext-all-sms' in sys.argv
if '--ext-all-sms' in sys.argv:
    sys.argv.remove('--ext-all-sms')

FAST_BUILD_EXT = '--ext' in sys.argv
if '--ext' in sys.argv:
    sys.argv.remove('--ext')

# compile cuaev with DEBUG infomation
CUAEV_DEBUG = '--cuaev-debug' in sys.argv
if CUAEV_DEBUG:
    sys.argv.remove('--cuaev-debug')

# compile cuaev with optimizations: e.g. intrinsics functions and use_fast_math flag
# CUAEV_OPT = '--cuaev-opt' in sys.argv
# if CUAEV_OPT:
#     sys.argv.remove('--cuaev-opt')
CUAEV_OPT = True

if not BUILD_EXT_ALL_SM and not FAST_BUILD_EXT:
    log.warning("Will not install cuaev")

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
        wget -q https://github.com/NVIDIA/cub/archive/main.zip;
        unzip -q main.zip -d include;
        mv include/cub-main/cub include;
        echo "Removing unnecessary files";
        rm main.zip;
        rm -rf include/cub-main;
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
        print('FAST_BUILD_EXT: ON')
        print(
            'This build will only support the following devices'
            ' (or devices with same cuda capability): '
        )
        for i in range(devices):
            d = 'cuda:{}'.format(i)
            _sm = torch.cuda.get_device_capability(i)
            sm = int(f'{_sm[0]}{_sm[1]}')
            if sm >= 50:
                print('{}: {}'.format(i, torch.cuda.get_device_name(d)))
                print('   {}'.format(torch.cuda.get_device_properties(i)))
            if sm not in SMs and sm >= 50:
                SMs.append(sm)

    nvcc_args = ['--expt-extended-lambda']
    if CUAEV_OPT:
        nvcc_args.append('-use_fast_math')
    # nvcc_args.append('-Xptxas=-v')

    # use cub in a safe manner, see:
    # https://github.com/pytorch/pytorch/pull/55292
    # https://github.com/pytorch/pytorch/pull/66219
    nvcc_args += [
        '-DCUB_NS_QUALIFIER=::cuaev::cub',
        '-DCUB_NS_PREFIX=namespace cuaev {',
        '-DCUB_NS_POSTFIX=}',
    ]
    if SMs:
        for sm in SMs:
            nvcc_args.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")
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
    if CUAEV_DEBUG:
        nvcc_args.append('-DTORCHANI_DEBUG')
    if CUAEV_OPT:
        nvcc_args.append('-DTORCHANI_OPT')
    print("nvcc_args: ", nvcc_args)
    print('-' * 75)
    include_dirs = [*maybe_download_cub(), os.path.abspath("torchani/csrc/")]
    return CUDAExtension(
        name='torchani.cuaev',
        sources=["torchani/csrc/cuaev.cpp", "torchani/csrc/aev.cu"],
        include_dirs=include_dirs,
        extra_compile_args={'cxx': ['-std=c++14'], 'nvcc': nvcc_args})


def mnp_extension():
    from torch.utils.cpp_extension import CUDAExtension
    cxx_args = ['-std=c++14', '-fopenmp']
    if CUAEV_DEBUG:
        cxx_args.append('-DTORCHANI_DEBUG')
    return CUDAExtension(
        name='torchani.mnp',
        sources=["torchani/csrc/mnp.cpp"],
        extra_compile_args={'cxx': cxx_args})


def ext_kwargs():
    if not BUILD_EXT_ALL_SM and not FAST_BUILD_EXT:
        return dict(
            provides=['torchani']
        )
    from torch.utils.cpp_extension import BuildExtension
    kwargs = dict(
        provides=[
            'torchani',
            'torchani.cuaev',
            'torchani.mnp',
        ],
        ext_modules=[
            cuda_extension(BUILD_EXT_ALL_SM),
            mnp_extension()
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
    setup_requires=["setuptools>=61", 'setuptools_scm>=8'],
    install_requires=[
        'typing_extensions>=4.0.0',
        'torch',
        'numpy',
        'lark-parser',
        'requests',
        'h5py',
        'pyyaml',
        'tqdm',
    ],
    entry_points={
        'console_scripts': ['torchani = torchani.cli:main'],
    },
    **ext_kwargs()
)
