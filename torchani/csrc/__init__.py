from pathlib import Path
import os
import warnings

import torch
from torchani.paths import data_dir


internal_path = Path(__file__).resolve().parent.parent
external_path = data_dir().parent.parent / "lib" / "Torchani"

# This env var is meant to be used by developers to manually disable extensions
# for testing purposes
if os.getenv("TORCHANI_DISABLE_EXTENSIONS") == "1":
    CUAEV_IS_INSTALLED = False
    MNP_IS_INSTALLED = False
    CLIST_IS_INSTALLED = False
    _missing = ["cuaev", "mnp", "cell_list"]
else:
    _missing = []
    try:
        torch.ops.load_library(internal_path / "cuaev.so")
        CUAEV_IS_INSTALLED = True
    except Exception:
        try:
            torch.ops.load_library(external_path / "cuaev.so")
            CUAEV_IS_INSTALLED = True
        except Exception:
            _missing.append("cuaev")
            CUAEV_IS_INSTALLED = False

    try:
        torch.ops.load_library(internal_path / "mnp.so")
        MNP_IS_INSTALLED = True
    except Exception:
        try:
            torch.ops.load_library(external_path / "mnp.so")
            MNP_IS_INSTALLED = True
        except Exception:
            _missing.append("mnp")
            MNP_IS_INSTALLED = False

    try:
        torch.ops.load_library(internal_path / "cell_list.so")
        CLIST_IS_INSTALLED = True
    except Exception:
        try:
            torch.ops.load_library(external_path / "cell_list.so")
            CLIST_IS_INSTALLED = True
        except Exception:
            _missing.append("cell_list")
            CLIST_IS_INSTALLED = False

if os.getenv("TORCHANI_NO_WARN_EXTENSIONS") != "1":
    if _missing:
        warnings.warn(
            f"The extensions: {_missing} are not installed and will not be available."
            " To install the extensions first install the CUDA Toolkit, and afterwards "
            " run `ani build-extensions`"
            " To suppress warn set the env var TORCHANI_NO_WARN_EXTENSIONS=1"
            " For example, if using bash,"
            " you may add `export TORCHANI_NO_WARN_EXTENSIONS=1` to your .bashrc"
        )


__all__ = ["CUAEV_IS_INSTALLED", "MNP_IS_INSTALLED", "CLIST_IS_INSTALLED"]
