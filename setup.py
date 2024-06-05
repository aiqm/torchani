from copy import deepcopy
import typing as tp
import textwrap
import os
import subprocess
import sys

from setuptools import setup


def maybe_download_cub(torch_include_dirs: tp.Iterable[str]) -> str:
    print("-" * 75)
    print("The CUB library is needed to build the cuAEV extension")
    for d in torch_include_dirs:
        cubdir = os.path.join(d, "cub")
        print(f"Searching for CUB at {cubdir}...")
        if os.path.isdir(cubdir):
            print(f"Found CUB in {cubdir}")
            return ""
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
    return os.path.abspath("./include")


def cuaev_extension_kwargs(
    sms: tp.Set[str],
    debug: bool,
    opt: bool,
) -> tp.Dict[str, tp.Any]:
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
    return dict(
        name="torchani.cuaev",
        sources=["torchani/csrc/cuaev.cpp", "torchani/csrc/aev.cu"],
        include_dirs=[os.path.abspath("torchani/csrc/")],
        extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
    )


def mnp_extension_kwargs(debug: bool) -> tp.Dict[str, tp.Any]:
    print("-" * 75)
    print("Will build MNP")

    cxx_args = ["-std=c++17", "-fopenmp"]
    if debug:
        cxx_args.append("-DTORCHANI_DEBUG")

    print("C++ compiler args:")
    for arg in cxx_args:
        print(f"    {arg}")
    return dict(
        name="torchani.mnp",
        sources=["torchani/csrc/mnp.cpp"],
        extra_compile_args={"cxx": cxx_args},
    )


def will_not_build_extensions_warning(torch_import_error: bool = False) -> None:
    print("-" * 75)
    if torch_import_error:
        print("Torch could not be imported")
    print(
        textwrap.dedent(
            """
            Will not install TorchANI extensions (cuAEV and MNP)
            To build the extensions with the pip frontend:
            - Make sure Torch binaries compiled with CUDA support are installed
            - Make sure a compatible CUDA Toolkit version is available
            - Add the --no-build-isolation flag to pip
            - Add --config-settings=--global-option=ext (verbatim) flag to pip
            """
        ).strip()
    )
    print("-" * 75)


TORCHANI_FLAGS = {"ext", "ext-all-sms", "ext-debug", "ext-no-opt"}
SUPPORTED_SMS = {"60", "61", "70", "75", "80", "86"}
for sm in SUPPORTED_SMS:
    TORCHANI_FLAGS.add(f"ext-sm{sm}")


def strip_argv():
    argv = deepcopy(sys.argv)
    for arg in argv:
        if arg in TORCHANI_FLAGS:
            sys.argv.remove(arg)


def setup_kwargs() -> tp.Dict[str, tp.Any]:
    # setuptools executes this file:
    # - 3 times in case of build-isolation mode
    #   (egg_info, dist_info, (editable|bdist)_wheel)
    # - 2 times in case of no-build-isolation mode
    #   (dist_info, (editable|bdist)_wheel)
    #
    # Extensions may only be built when building the actual wheel,
    # In other cases executing this file is a no-op.
    if "dist_info" in sys.argv or "egg_info" in sys.argv:
        # In this case setuptools is just building the metadata
        # --global-option is passed to all stages of the build, but
        # we need the options only when building the wheel, so we strip sys.argv
        # of the options in other cases
        strip_argv()
        return dict()

    # Building the actual wheel, so attempt to import torch:
    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension
        from torch.utils.cpp_extension import BuildExtension

        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False

    if not TORCH_AVAILABLE:
        will_not_build_extensions_warning(torch_import_error=True)
        strip_argv()
        return dict()

    def collect_all_sms() -> tp.Set[str]:
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
        return sms

    def collect_compatible_sms() -> tp.Set[str]:
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
        if sms:
            return sms
        print("No compatible devices found")
        return collect_all_sms()

    # Flags for requesting specific SMs
    sms: tp.Set[str] = set()
    for sm in SUPPORTED_SMS:
        if f"ext-sm{sm}" in sys.argv:
            sys.argv.remove(f"ext-sm{sm}")
            sms.add(sm)
    # Flag for requesting compatible SMs detection
    if "ext" in sys.argv:
        sys.argv.remove("ext")
        sms.update(collect_compatible_sms())
    # Flag for requesting all sms
    if "ext-all-sms" in sys.argv:
        sys.argv.remove("ext-all-sms")
        sms.update(collect_all_sms())

    # Compile extensions with DEBUG infomation
    debug = False
    if "ext-debug" in sys.argv:
        sys.argv.remove("ext-debug")
        debug = True

    # Compile optimized extensions
    # (intrinsic math fns and -use_fast_math nvcc flag)
    opt = True
    if "ext-no-opt" in sys.argv:
        sys.argv.remove("ext-no-opt")
        opt = False

    # At least 1 SM is always added to the "sms" set if extensions need to be built
    # If nothing is added, then don't build the extensions
    if not sms:
        will_not_build_extensions_warning()
        return dict()

    cuaev_kwargs = cuaev_extension_kwargs(sms, debug, opt)
    mnp_kwargs = mnp_extension_kwargs(debug=debug)

    # CUB needed to build the cuAEV, download it if not found bundled with Torch
    torch_include_dirs = torch.utils.cpp_extension.include_paths(cuda=True)
    cub_include_dir = maybe_download_cub(torch_include_dirs)
    if cub_include_dir:
        cuaev_kwargs["include_dirs"].append(cub_include_dir)

    # MNP extension doesn't need nvcc to be compiled, but it still uses torch
    # CUDA libraries, so CUDAExtension is needed
    print("-" * 75)
    return {
        "ext_modules": [CUDAExtension(**cuaev_kwargs), CUDAExtension(**mnp_kwargs)],
        "cmdclass": {
            "build_ext": BuildExtension.with_options(
                no_python_abi_suffix=True,
            )
        },
    }


setup(**setup_kwargs())
