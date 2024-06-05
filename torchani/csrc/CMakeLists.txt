cmake_minimum_required(
    VERSION 3.22
    FATAL_ERROR
)

project(
    TorchaniExt
    LANGUAGES C CXX
    VERSION 0.31416
    DESCRIPTION "C++ and CUDA extensions for Torchani"
    HOMEPAGE_URL "https://github.com/roitberg-group/torchani_sandbox.git"
)

# Export compilation database into compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# If we are in a conda env, append the conda prefix so CMake detects the conda
# packages
list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})

execute_process(
    COMMAND python -c "import torch; from pathlib import Path; print(Path(torch.__file__).resolve().parent, end='')"
    OUTPUT_VARIABLE INSTALLED_TORCH_ROOT
    RESULT_VARIABLE IMPORT_TORCH_FAILED
)
if(IMPORT_TORCH_FAILED)
    message(FATAL_ERROR "Could not import PyTorch and find its root dir")
else()
    message(STATUS "Successfully imported PyTorch and found root dir")
endif()
list(APPEND CMAKE_PREFIX_PATH ${INSTALLED_TORCH_ROOT})

find_package(Torch REQUIRED)
# Set Python_LIBRARIES, Python_INCLUDE_DIRS
find_package(Python REQUIRED COMPONENTS Interpreter Development)

add_library(
    cuaev
    SHARED
    aev.h
    cuaev.cpp
    aev.cu
    cuaev_cub.cuh
)
set_property(TARGET cuaev PROPERTY PREFIX "")
target_compile_features(cuaev PRIVATE cxx_std_17)
target_link_libraries(cuaev PRIVATE ${TORCH_LIBRARIES} ${Python_LIBRARIES})
set(CUAEV_CUB_NAMESPACE_DEFS "CUB_NS_QUALIFIER=::cuaev::cub" "CUB_NS_PREFIX=namespace cuaev {" "CUB_NS_POSTFIX=}")
target_compile_definitions(
    cuaev
        PRIVATE
            ${TORCH_CXX_FLAGS}
            $<$<COMPILE_LANGUAGE:CUDA>:TORCHANI_OPT>
            $<$<COMPILE_LANGUAGE:CUDA>:${CUAEV_CUB_NAMESPACE_DEFS}>
)
target_compile_options(
    cuaev
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
        )
target_include_directories(cuaev PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${Python_INCLUDE_DIRS})

# Required for MNP only
find_package(OpenMP REQUIRED COMPONENTS CXX)
add_library(
    mnp
    SHARED
    mnp.cpp
)
set_property(TARGET mnp PROPERTY PREFIX "")
target_compile_features(mnp PRIVATE cxx_std_17)
target_link_libraries(mnp PRIVATE ${TORCH_LIBRARIES} ${Python_LIBRARIES} OpenMP::OpenMP_CXX)
target_compile_definitions(mnp PRIVATE  ${TORCH_CXX_FLAGS})
target_include_directories(mnp PRIVATE ${Python_INCLUDE_DIRS})

list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})

# By default install to the torchani repo src directory, "repo-name/torchani/"
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    get_filename_component(TORCHANI_SRC ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)
    get_filename_component(TORCHANI_ROOT ${TORCHANI_SRC} DIRECTORY)
    set(CMAKE_INSTALL_PREFIX ${TORCHANI_ROOT} CACHE PATH "Install location" FORCE)
endif()

# Permissions may be overkill, but this is how setuptools installs the extensions
install(
    TARGETS
        mnp
        cuaev
    LIBRARY
        DESTINATION "torchani/"
    PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_WRITE
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
)
