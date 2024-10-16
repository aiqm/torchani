from pathlib import Path
import os
import warnings

import torch

_CUAEV_PATH = str(Path(__file__).resolve().parent.parent / "cuaev.so")
_MNP_PATH = str(Path(__file__).resolve().parent.parent / "mnp.so")

try:
    torch.ops.load_library(_CUAEV_PATH)
    CUAEV_IS_INSTALLED = True
except Exception:
    CUAEV_IS_INSTALLED = False

try:
    torch.ops.load_library(_MNP_PATH)
    MNP_IS_INSTALLED = True
except Exception:
    MNP_IS_INSTALLED = False

# This env var is meant to be used by developers to manually disable extensions
# for testing purposes
if os.getenv("TORCHANI_DISABLE_EXTENSIONS") == "1":
    CUAEV_IS_INSTALLED = False
    MNP_IS_INSTALLED = False

if os.getenv("TORCHANI_NO_WARN_EXTENSIONS") != "1":
    if not CUAEV_IS_INSTALLED:
        warnings.warn(
            "The AEV CUDA extension is not installed and will not be available."
            " To suppress warn set the env var TORCHANI_NO_WARN_EXTENSIONS=1"
            " For example, if using bash,"
            " you may add `export TORCHANI_NO_WARN_EXTENSIONS=1` to your .bashrc"
        )
    if not MNP_IS_INSTALLED:
        warnings.warn(
            "The MNP C++ extension is not installed and will not be available."
            " To suppress warn set the env var TORCHANI_NO_WARN_EXTENSIONS=1"
            " For example, if using bash,"
            " you may add `export TORCHANI_NO_WARN_EXTENSIONS=1` to your .bashrc"
        )


__all__ = ["CUAEV_IS_INSTALLED", "MNP_IS_INSTALLED"]
