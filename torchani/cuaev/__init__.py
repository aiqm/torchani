import warnings

try:
    import _real_cuaev as the_cuaev
except ImportError:
    warnings.warn("cuaev not installed")
    from . import stub as the_cuaev

is_installed = the_cuaev.is_installed
cuComputeAEV = the_cuaev.cuComputeAEV
