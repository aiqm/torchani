import os
import warnings
import importlib.metadata

CUAEV_IS_INSTALLED = "torchani.cuaev" in importlib.metadata.metadata(
    __package__.split(".")[0]
).get_all("Provides", [])

MNP_IS_INSTALLED = "torchani.mnp" in importlib.metadata.metadata(
    __package__.split(".")[0]
).get_all("Provides", [])

# This env var is meant to be used by developers to manually disable extensions
# for testing purposes
if "TORCHANI_DISABLE_EXTENSIONS" in os.environ:
    CUAEV_IS_INSTALLED = False
    MNP_IS_INSTALLED = False

if "TORCHANI_NO_WARN_EXTENSIONS" not in os.environ:
    if not CUAEV_IS_INSTALLED:
        warnings.warn(
            "The AEV CUDA extension is not installed and will not be available."
            " To suppress warn set the env var TORCHANI_NO_WARN_EXTENSIONS to any value"
        )
    if not MNP_IS_INSTALLED:
        warnings.warn(
            "The MNP C++ extension is not installed and will not be available."
            " To suppress warn set the env var TORCHANI_NO_WARN_EXTENSIONS to any value"
        )


__all__ = ["CUAEV_IS_INSTALLED", "MNP_IS_INSTALLED"]
