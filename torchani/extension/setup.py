from setuptools import setup
#from torch.utils.cpp_extension import BuildExtension, CppExtension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuaev',
    version='0.1',
    ext_modules=[
        CUDAExtension(
            name='cuaev',
            sources=['aev.cu'],
            include_dirs=['.'],
            extra_compile_args={'cxx': ['-std=c++11'],
                            'nvcc': ['-arch=sm_70', "-Xptxas=-v", '--extended-lambda', '-use_fast_math']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

