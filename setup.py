import typing as tp
import os
import subprocess
import sys
import logging

from setuptools import setup, find_packages

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("setup")


def collect_all_sms() -> tp.Set[str]:
    import torch

    print("-" * 75)
    print("Will add all SMs torch supports")
    sms = {"60", "61", "70"}
    _torch_cuda = torch.version.cuda
    cuda_version = float(_torch_cuda) if _torch_cuda is not None else 0
    if cuda_version >= 10:
        sms.add("75")
    if cuda_version >= 11:
        sms.add("80")
    if cuda_version >= 11.1:
        sms.add("86")
    print("-" * 75)
    print()
    return sms


def collect_compatible_sms() -> tp.Set[str]:
    import torch

    print("-" * 75)
    print("Will try to find compatible CUDA devices visible to torch")
    devices = torch.cuda.device_count()
    sms: tp.Set[str] = set()
    for i in range(devices):
        sm_tuple = torch.cuda.get_device_capability(i)
        if sm_tuple >= (5, 0):
            print("Found compatible device:")
            print(f'{i}: {torch.cuda.get_device_name(f"cuda:{i}")}')
            print(f"   {torch.cuda.get_device_properties(i)}")
            sms.add(f"{sm_tuple[0]}{sm_tuple[1]}")
    if not sms:
        print("No compatible devices found")
        sms = collect_all_sms()
    print("-" * 75)
    print()

    return sms


def maybe_download_cub():
    import torch

    dirs = torch.utils.cpp_extension.include_paths(cuda=True)
    for d in dirs:
        cubdir = os.path.join(d, "cub")
        log.info(f"Searching for cub at {cubdir}...")
        if os.path.isdir(cubdir):
            log.info(f"Found cub in {cubdir}")
            return []
    # if no cub, download it to include dir from github
    if not os.path.isdir("./include/cub"):
        if not os.path.exists("./include"):
            os.makedirs("include")
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


def cuda_extension(sms: tp.Set[str], debug: bool, opt: bool):
    from torch.utils.cpp_extension import CUDAExtension

    print("-" * 75)
    print(f"Will build cuAEV with support for SMs: {', '.join(sms)}")

    nvcc_args = ["--expt-extended-lambda"]
    # The following defs are required to use CUB safely. For details see:
    # https://github.com/pytorch/pytorch/pull/55292
    # https://github.com/pytorch/pytorch/pull/66219
    nvcc_args.extend(
        [
            "-DCUB_NS_QUALIFIER=::cuaev::cub",
            "-DCUB_NS_PREFIX=namespace cuaev {",
            "-DCUB_NS_POSTFIX=}",
        ]
    )
    if debug:
        nvcc_args.append("-DTORCHANI_DEBUG")
    if opt:
        nvcc_args.extend(["-DTORCHANI_OPT", "-use_fast_math"])
    nvcc_args.extend([f"-gencode=arch=compute_{sm},code=sm_{sm}" for sm in sms])
    print("NVCC compiler args:")
    for arg in nvcc_args:
        print(f"    {arg}")

    cxx_args = ["-std=c++17"]
    print("C++ compiler args:")
    for arg in cxx_args:
        print(f"    {arg}")

    print("-" * 75)
    print()
    include_dirs = [*maybe_download_cub(), os.path.abspath("torchani/csrc/")]
    return CUDAExtension(
        name="torchani.cuaev",
        sources=["torchani/csrc/cuaev.cpp", "torchani/csrc/aev.cu"],
        include_dirs=include_dirs,
        extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
    )


def mnp_extension(debug: bool):
    print("-" * 75)
    print("Will build MNP")

    # This extension doesn't need nvcc to be compiled, but it still uses torch
    # CUDA libraries. This means CUDAExtension is needed
    from torch.utils.cpp_extension import CUDAExtension

    cxx_args = ["-std=c++17", "-fopenmp"]
    if debug:
        cxx_args.append("-DTORCHANI_DEBUG")

    print("C++ compiler args:")
    for arg in cxx_args:
        print(f"    {arg}")
    print("-" * 75)
    print()
    return CUDAExtension(
        name="torchani.mnp",
        sources=["torchani/csrc/mnp.cpp"],
        extra_compile_args={"cxx": cxx_args},
    )


# Flags for requiresting specific SMs
sms: tp.Set[str] = set()
for sm in {"60", "61", "70", "75", "80", "86"}:
    if f"--ext-sm{sm}" in sys.argv:
        sys.argv.remove(f"--ext-sm{sm}")
        sms.add(sm)
# Flag for requesting compatible SMs detection
if "--ext" in sys.argv:
    sys.argv.remove("--ext")
    sms.update(collect_compatible_sms())
# Flag for requesting all sms
if "--ext-all-sms" in sys.argv:
    sys.argv.remove("--ext-all-sms")
    sms.update(collect_all_sms())

# Compile extensions with DEBUG infomation
debug = False
if "--debug" in sys.argv:
    sys.argv.remove("--debug")
    debug = True

# Compile optimized extensions (intrinsic math fns and -use_fast_math nvcc flag)
opt = True
if "--no-opt" in sys.argv:
    sys.argv.remove("--no-opt")
    opt = False


with open("README.md", "r") as fh:
    long_description = fh.read()


# At least 1 sm is always added to this set if extensions need to be built
if not sms:
    log.warning("Will not install TorchANI extensions")
    ext_kwargs = dict()
else:
    from torch.utils.cpp_extension import BuildExtension

    ext_kwargs = {
        "ext_modules": [
            cuda_extension(sms, debug=debug, opt=opt),
            mnp_extension(debug=debug),
        ],
        "cmdclass": {
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        },
    }


setup(
    name="torchani",
    description="PyTorch implementation of ANI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aiqm/torchani",
    author="Xiang Gao",
    author_email="qasdfgtyuiop@gmail.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    use_scm_version={"fallback_version": "0.0.0"},
    setup_requires=["setuptools>=61", "setuptools_scm>=8"],
    install_requires=[
        "numpy>=1.24",
        "torch==2.3",
        "typing_extensions>=4.0.0",
        "h5py",
        "tqdm",
    ],
    entry_points={
        "console_scripts": ["torchani = torchani.cli:main"],
    },
    **ext_kwargs,
)
