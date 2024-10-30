r"""Unit conversion factors used in TorchANI

The ANI models work internally in Hartrees (energy), Angstroms (distance), and AMU
(mass).

In some example code and scripts we convert to other more commonly used units.
Conversion factors are consistent with `CODATA 2014 recommendations`_, which is also
consistent with the `units used in ASE`_ by default. (However, take into account that
ASE uses electronvolt as its base energy unit, so the appropriate conversion factors
should always be applied when converting from ASE to TorchANI) Joule-to-kcal conversion
is taken from the `IUPAC Goldbook`_.  All the conversion factors we use are defined in
this module, and convenience functions to convert between different units are provided.

.. _units used in ASE:
    https://wiki.fysik.dtu.dk/ase/ase/units.html#units

.. _CODATA 2014 recommendations:
    https://arxiv.org/pdf/1507.07956.pdf

.. _IUPAC Goldbook:
    https://goldbook.iupac.org/terms/view/C00784
"""

import math


# Comments on ASE:

# ASE uses its own constants, which they take from CODATA 2014 as of
# 03/10/2019. ASE's units are created when ase.units is imported.
# Since we don't want to be dependent on ASE changing their module
# we define here our own units for our own purposes, but right now we define
# them to be consistent with ASE values (i. e. our values are also CODATA 2014)
# the difference between CODATA 2014 and CODATA 2018 is negligible.


# General conversion:

# the codata value for hartree in units of eV can be obtained from
# m_e * e^3 / ( 16 * pi^2 * eps_0^2 hbar^2 )
ANGSTROM_TO_BOHR = 1.8897261258369282
HARTREE_TO_EV = 27.211386024367243  # equal to ase.units.Hartree
EV_TO_JOULE = 1.6021766208e-19  # equal to ase.units._e (electron charge)
JOULE_TO_KCAL = 1 / 4184.0  # exact
HARTREE_TO_JOULE = HARTREE_TO_EV * EV_TO_JOULE
AVOGADROS_NUMBER = 6.022140857e23  # equal to ase.units._Nav
SPEED_OF_LIGHT = 299792458.0  # equal to ase.units._c
AMU_TO_KG = 1.660539040e-27  # equal to ase.units._amu
ANGSTROM_TO_METER = 1e-10
NEWTON_TO_MILLIDYNE = 1e8  # exact relation
HARTREE_TO_KCALPERMOL = HARTREE_TO_JOULE * JOULE_TO_KCAL * AVOGADROS_NUMBER
HARTREE_TO_KJOULEPERMOL = HARTREE_TO_JOULE * AVOGADROS_NUMBER / 1000
EV_TO_KCALPERMOL = EV_TO_JOULE * JOULE_TO_KCAL * AVOGADROS_NUMBER
EV_TO_KJOULEPERMOL = EV_TO_JOULE * AVOGADROS_NUMBER / 1000
DEBYE_TO_ELECTRON_ANGSTROM = 0.2081943

# For vibrational analysis:

INVCM_TO_EV = 0.0001239841973964072  # equal to ase.units.invcm
# To convert from the sqrt of eigenvalues (mass-scaled hessian units of
# sqrt(hartree / (amu * A^2)) into cm^-1
# it is necessary to multiply by the sqrt ( HARTREE_TO_JOULE / AMU_TO_KG ),
# then we convert angstroms to meters and divide by 1/ speed_of_light (to
# convert seconds into meters). Finally divide by 100 to get inverse cm
# The resulting value should be close to 17092
SQRT_MHESSIAN_TO_INVCM = (
    math.sqrt(HARTREE_TO_JOULE / AMU_TO_KG) / ANGSTROM_TO_METER / SPEED_OF_LIGHT
) / 100
# meV is other common unit used to present vibrational transition energies
SQRT_MHESSIAN_TO_MILLIEV = SQRT_MHESSIAN_TO_INVCM * INVCM_TO_EV * 1000
# To convert units form mass-scaled hessian units into mDyne / Angstrom (force
# constant units) factor should be close to 4.36
MHESSIAN_TO_FCONST = HARTREE_TO_JOULE * NEWTON_TO_MILLIDYNE / ANGSTROM_TO_METER


def angstrom2bohr(x):
    r"""Angstrom to Bohr conversion factor from 2014 CODATA"""
    return x * ANGSTROM_TO_BOHR


def bohr2angstrom(x):
    r"""Bohr to Angstrom conversion factor from 2014 CODATA"""
    return x / ANGSTROM_TO_BOHR


def sqrt_mhessian2invcm(x):
    r"""Converts sqrt(mass-scaled hessian units) into cm^-1

    Converts form units of sqrt(Hartree / (amu * Angstrom^2))
    which are sqrt(units of the mass-scaled hessian matrix)
    into units of inverse centimeters.

    Take into account that to convert the actual eigenvalues of the hessian
    into wavenumbers it is necessary to multiply by an extra factor of 1 / (2 *
    pi)"""
    return x * SQRT_MHESSIAN_TO_INVCM


def sqrt_mhessian2milliev(x):
    r"""Converts sqrt(mass-scaled hessian units) into meV

    Converts form units of sqrt(Hartree / (amu * Angstrom^2))
    which are sqrt(units of the mass-scaled hessian matrix)
    into units of milli-electronvolts.

    Take into account that to convert the actual eigenvalues of the hessian
    into wavenumbers it is necessary to multiply by an extra factor of 1 / (2 *
    pi)"""
    return x * SQRT_MHESSIAN_TO_MILLIEV


def mhessian2fconst(x):
    r"""Converts mass-scaled hessian units into mDyne/Angstrom

    Converts from units of mass-scaled hessian (Hartree / (amu * Angstrom^2)
    into force constant units (mDyne/Angstom), where 1 N = 1 * 10^8 mDyne"""
    return x * MHESSIAN_TO_FCONST


def hartree2ev(x):
    r"""Hartree to eV conversion factor from 2014 CODATA"""
    return x * HARTREE_TO_EV


def ev2kjoulepermol(x):
    r"""Electronvolt to kJ/mol conversion factor from CODATA 2014"""
    return x * EV_TO_KJOULEPERMOL


def ev2kcalpermol(x):
    r"""Electronvolt to kcal/mol conversion factor from CODATA 2014"""
    return x * EV_TO_KCALPERMOL


def hartree2kjoulepermol(x):
    r"""Hartree to kJ/mol conversion factor from CODATA 2014"""
    return x * HARTREE_TO_KJOULEPERMOL


def hartree2kcalpermol(x):
    r"""Hartree to kJ/mol conversion factor from CODATA 2014"""
    return x * HARTREE_TO_KCALPERMOL


def ea2debye(x):
    """Dipole conversion, eA to Debye from NIST CCCBDB"""
    return x / DEBYE_TO_ELECTRON_ANGSTROM


# Add actual values to docstrings on import
angstrom2bohr.__doc__ = (
    str(angstrom2bohr.__doc__) + f"\n\n1 Angstrom = {angstrom2bohr(1)} Bohr"
)
bohr2angstrom.__doc__ = (
    str(bohr2angstrom.__doc__) + f"\n\n1 Bohr = {bohr2angstrom(1)} Angstrom"
)
hartree2ev.__doc__ = str(hartree2ev.__doc__) + f"\n\n1 Hartree = {hartree2ev(1)} eV"
hartree2kcalpermol.__doc__ = (
    str(hartree2kcalpermol.__doc__)
    + f"\n\n1 Hartree = {hartree2kcalpermol(1)} kcal/mol"
)
hartree2kjoulepermol.__doc__ = (
    str(hartree2kjoulepermol) + f"\n\n1 Hartree = {hartree2kjoulepermol(1)} kJ/mol"
)
ev2kjoulepermol.__doc__ = (
    str(ev2kjoulepermol.__doc__) + f"\n\n1 eV = {ev2kjoulepermol(1)} kJ/mol"
)
ev2kcalpermol.__doc__ = (
    str(ev2kcalpermol.__doc__) + f"\n\n1 eV = {ev2kcalpermol(1)} kcal/mol"
)
mhessian2fconst.__doc__ = (
    str(mhessian2fconst.__doc__)
    + f"\n\n1 Hartree / (AMU * Angstrom^2) = {ev2kcalpermol(1)} mDyne/Angstrom"
)
sqrt_mhessian2milliev.__doc__ = (
    str(sqrt_mhessian2milliev.__doc__)
    + f"\n\n1 sqrt(Hartree / (AMU * Angstrom^2)) = {sqrt_mhessian2milliev(1)} meV"
)
sqrt_mhessian2invcm.__doc__ = (
    str(sqrt_mhessian2invcm.__doc__)
    + f"\n\n1 sqrt(Hartree / (AMU * Angstrom^2)) = {sqrt_mhessian2invcm(1)} cm^-1"
)
ea2debye.__doc__ = (
    str(ea2debye.__doc__)
    + f"\n\n1 Debye = {DEBYE_TO_ELECTRON_ANGSTROM} electron Angstroms"
)


# Old aliases (try not to use if possible)
ev2kcalmol = ev2kcalpermol
hartree2kcalmol = hartree2kcalpermol
ev2kjoulemol = ev2kjoulepermol
hartree2kjoulemol = hartree2kjoulepermol
HARTREE_TO_KCALMOL = HARTREE_TO_KCALPERMOL
EV_TO_KCALMOL = EV_TO_KCALPERMOL
HARTREE_TO_KJOULEMOL = HARTREE_TO_KJOULEPERMOL
EV_TO_KJOULEMOL = EV_TO_KJOULEPERMOL
