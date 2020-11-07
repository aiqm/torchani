import warnings
import importlib_metadata

is_installed = 'torchani.cuaev' in importlib_metadata.metadata('torchani').get_all('Provides')

if is_installed:
    import _real_cuaev
    cuComputeAEV = _real_cuaev.cuComputeAEV
else:
    warnings.warn("cuaev not installed")

    def cuComputeAEV(*args, **kwargs):
        raise RuntimeError("cuaev is not installed")
