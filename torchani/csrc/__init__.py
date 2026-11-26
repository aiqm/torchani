from pathlib import Path
import os
import warnings

import torch
from torchani.paths import extensions_dir, extensions_build_tag

_env_internal_dir = os.getenv("TORCHANI_EXTENSIONS_INTERNAL_DIR")
if _env_internal_dir:
    internal_path = Path(_env_internal_dir).expanduser().resolve()
else:
    internal_path = Path(__file__).resolve().parent.parent
external_path = extensions_dir()
internal_suffix = ".pyd" if os.name == "nt" else ".so"
external_suffix = ".dll" if os.name == "nt" else ".so"
_names = ["cuaev", "mnp", "cell_list"]


def _internal_build_tag() -> str | None:
    build_tag_path = internal_path / "_extensions_build_tag.txt"
    if not build_tag_path.is_file():
        return None
    return build_tag_path.read_text(encoding="utf-8").strip()


def _load_extension(name: str, internal_is_compatible: bool) -> bool:
    if internal_is_compatible:
        try:
            torch.ops.load_library(internal_path / f"{name}{internal_suffix}")
            return True
        except Exception:
            pass
    try:
        torch.ops.load_library(external_path / f"{name}{external_suffix}")
        return True
    except Exception:
        return False


def _other_external_builds() -> list[str]:
    r"""Extensions built by ``ani build-extensions`` for other torch versions"""
    root = external_path.parent
    if not root.is_dir():
        return []
    builds = []
    for d in sorted([root] + [p for p in root.iterdir() if p.is_dir()]):
        if d == external_path:
            continue
        found = [n for n in _names if (d / f"{n}{external_suffix}").is_file()]
        if found:
            tag = "unknown version" if d == root else d.name
            builds.append(f"{tag} ({', '.join(found)})")
    return builds


_warn = os.getenv("TORCHANI_NO_WARN_EXTENSIONS") != "1"

# This env var is meant to be used by developers to manually disable extensions
# for testing purposes
if os.getenv("TORCHANI_DISABLE_EXTENSIONS") == "1":
    _missing = list(_names)
else:
    _internal_tag = _internal_build_tag()
    _internal_is_compatible = _internal_tag == extensions_build_tag()
    _internal_exists = any(
        (internal_path / f"{n}{internal_suffix}").is_file() for n in _names
    )
    if _warn and _internal_exists and not _internal_is_compatible:
        warnings.warn(
            f"The extensions in {internal_path} were built for torch"
            f" {_internal_tag or '(unknown version)'}, but the installed torch is"
            f" {extensions_build_tag()}, so they will not be loaded."
            " Reinstall torchani with the extensions, or run `ani build-extensions`"
        )
    _missing = [n for n in _names if not _load_extension(n, _internal_is_compatible)]

CUAEV_IS_INSTALLED = "cuaev" not in _missing
MNP_IS_INSTALLED = "mnp" not in _missing
CLIST_IS_INSTALLED = "cell_list" not in _missing

if _warn:
    if _missing:
        _other_builds = _other_external_builds()
        if _other_builds:
            _other_builds_msg = (
                f" Extensions in {external_path.parent} were found only for other"
                f" torch versions: {'; '.join(_other_builds)}."
                f" The installed torch is {extensions_build_tag()}."
            )
        else:
            _other_builds_msg = ""
        warnings.warn(
            f"The extensions: {_missing} are not installed and will not be available."
            f"{_other_builds_msg}"
            " To install the extensions first install the CUDA Toolkit, and afterwards "
            " run `ani build-extensions`"
            " To suppress warn set the env var TORCHANI_NO_WARN_EXTENSIONS=1"
            " For example, if using bash,"
            " you may add `export TORCHANI_NO_WARN_EXTENSIONS=1` to your .bashrc"
        )

if CUAEV_IS_INSTALLED:
    import torchani.csrc._cuaev_fake  # noqa: F401

if CLIST_IS_INSTALLED:
    import torchani.csrc._cell_list_fake  # noqa: F401


__all__ = ["CUAEV_IS_INSTALLED", "MNP_IS_INSTALLED", "CLIST_IS_INSTALLED"]
