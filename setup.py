import os
import subprocess
from setuptools import setup, find_packages
from distutils import log
import sys
import warnings


def alert(text):
    return('\033[91m{}\33[0m'.format(text))  # red


BUILD_EXT_ALL_SM = '--cuaev-all-sms' in sys.argv
if BUILD_EXT_ALL_SM:
    warnings.warn(alert("--cuaev-all-sms flag is deprecated, please use --ext-all-sms instead."))
    sys.argv.remove('--cuaev-all-sms')

FAST_BUILD_EXT = '--cuaev' in sys.argv
if FAST_BUILD_EXT:
    warnings.warn(alert("--cuaev flag is deprecated, please use --ext instead."))
    sys.argv.remove('--cuaev')

BUILD_EXT_ALL_SM = BUILD_EXT_ALL_SM or '--ext-all-sms' in sys.argv
if '--ext-all-sms' in sys.argv:
    sys.argv.remove('--ext-all-sms')

FAST_BUILD_EXT = FAST_BUILD_EXT or '--ext' in sys.argv
if '--ext' in sys.argv:
    sys.argv.remove('--ext')

# Use along with --cuaev for CI test to reduce compilation time on Non-GPUs system
ONLY_BUILD_SM80 = '--only-sm80' in sys.argv
if ONLY_BUILD_SM80:
    sys.argv.remove('--only-sm80')

# DEBUG infomation for cuaev
DEBUG_EXT = '--debug' in sys.argv
if DEBUG_EXT:
    sys.argv.remove('--debug')

if not BUILD_EXT_ALL_SM and not FAST_BUILD_EXT:
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
    if DEBUG_EXT:
        nvcc_args.append('-DTORCHANI_DEBUG')
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
    if DEBUG_EXT:
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
    setup_requires=['setuptools_scm'],
    install_requires=[
        'torch',
        'lark-parser',
        'requests',
        'importlib_metadata',
    ],
    entry_points={
        'console_scripts': ['torchani = torchani.cli:main'],
    },
    **ext_kwargs()
)
