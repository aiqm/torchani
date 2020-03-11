r"""Unit conversion factors

All values derived from:
    CODATA Recommended Values of the Fundamental Physical Constants: 2014
    Peter J. Mohr, David B. Newell, Barry N. Taylor
    November 22, 2016

Except from the Joule-to-kcal conversion, taken to be 1 cal = 4.184 J
    IUPAC's Gold Book, IUPAC. Compendium of Chemical Terminology, 2nd ed.,
    the "thermochemical calorie".
"""

import math

# General conversion

# the codata value for hartree in units of eV can be obtained from
# m_e * e^3 / ( 16 * pi^2 * eps_0^2 hbar^2 )
HARTREE_TO_EV = 27.211386024367243  # equal to ase.units.Hartree
EV_TO_JOULE = 1.6021766208e-19  # equal to ase.units._e (electron charge)
JOULE_TO_KCAL = 1 / 4184.  # exact
HARTREE_TO_JOULE = HARTREE_TO_EV * EV_TO_JOULE

AVOGADROS_NUMBER = 6.022140857e+23  # equal to ase.units._Nav
SPEED_OF_LIGHT = 299792458.0  # equal to ase.units._c
AMU_TO_KG = 1.660539040e-27  # equal to ase.units._amu
ANGSTROM_TO_METER = 1e-10
NEWTON_TO_MILLIDYNE = 1e8  # exact relation

HARTREE_TO_KCALMOL = HARTREE_TO_JOULE * JOULE_TO_KCAL * AVOGADROS_NUMBER
HARTREE_TO_KJOULEMOL = HARTREE_TO_JOULE * AVOGADROS_NUMBER / 1000

EV_TO_KCALMOL = EV_TO_JOULE * JOULE_TO_KCAL * AVOGADROS_NUMBER
EV_TO_KJOULEMOL = EV_TO_JOULE * AVOGADROS_NUMBER / 1000

# For vibrational analysis:

# To convert from the sqrt of eigenvalues (mass-scaled hessian units of
# sqrt(hartree / (amu * A^2)) into cm^-1
# it is necessary to multiply by the sqrt ( HARTREE_TO_JOULE / AMU_TO_KG ),
# then we convert angstroms to meters and divide by 1/ speed_of_light (to
# convert seconds into meters). Finally divide by 100 to get inverse cm
# The resulting value should be close to 17092
INVCM_TO_EV = 0.0001239841973964072  # equal to ase.units.invcm
SQRT_MHESSIAN_TO_INVCM = (math.sqrt(HARTREE_TO_JOULE / AMU_TO_KG) / ANGSTROM_TO_METER / SPEED_OF_LIGHT) / 100
SQRT_MHESSIAN_TO_MILLIEV = SQRT_MHESSIAN_TO_INVCM * INVCM_TO_EV * 1000
# To convert units form mass-scaled hessian units into mDyne / Angstrom (force
# constant units) factor should be close to 4.36
MHESSIAN_TO_FCONST = HARTREE_TO_JOULE * NEWTON_TO_MILLIDYNE / ANGSTROM_TO_METER


# Comments on ASE:

# ASE uses its own constants, which they take from CODATA 2014 as of
# 03/10/2019. ASE's units are created when ase.units is imported.
# Since we don't want to be dependent on ASE changing their module
# we define here our own units for our own purposes, but right now we define
# them to be consistent with ASE values (i. e. our values are also CODATA 2014)
# the difference between CODATA 2014 and CODATA 2018 is negligible.

def sqrt_mhessian2invcm(x):
    r"""Convert from sqrt_mhessian into cm^-1

    Converts form units of sqrt(hartree / (amu * A^2) )
    which are the units of the mass-scaled hessian matrix
    into units of inverse centimeters. Take into account that to
    convert the eigenvalues of the hessian into wavenumbers it is
    necessary to multiply by an extra factor of 1/ (2 pi)"""
    return x * SQRT_MHESSIAN_TO_INVCM


def sqrt_mhessian2milliev(x):
    r"""Convert from sqrt_mhessian into millieV

    Converts form units of sqrt(hartree / (amu * A^2) )
    which are the units of the mass-scaled hessian matrix
    into units of milli-electronvolts. Take into account that to
    convert the eigenvalues of the hessian into wavenumbers it is
    necessary to multiply by an extra factor of 1/ (2 pi)"""
    return x * SQRT_MHESSIAN_TO_MILLIEV


def mhessian2fconst(x):
    r"""Converts form the mhessian into mDyne/A

    Converts from units of mass-scaled hessian (hartree / (amu * A^2)
    into force constant units (mDyne/Angstom)
    """
    return x * MHESSIAN_TO_FCONST


def hartree2ev(x):
    r"""Hartree to Electronvolt conversion factor

    Ha-to-eV factor, calculated from 2014 CODATA recommended values"""
    return x * HARTREE_TO_EV


def ev2kjoulemol(x):
    r"""Electronvolt to kJ/mol conversion factor

    eV-to-kJ/mol factor, calculated from 2014 CODATA recommended values"""
    return x * EV_TO_KJOULEMOL


def ev2kcalmol(x):
    r"""Electronvolt to kcal/mole conversion factor,

    eV-to-kcal/mol factor, calculated from 2014 CODATA recommended values"""
    return x * EV_TO_KCALMOL


def hartree2kjoulemol(x):
    r"""Hartree to kJ/mol conversion factor

    Ha-to-kJ/mol factor, calculated from 2014 CODATA recommended values"""
    return x * HARTREE_TO_KJOULEMOL


def hartree2kcalmol(x):
    r"""Hartree to kJ/mol conversion factor

    Ha-to-kcal/mol factor, calculated from 2014 CODATA recommended values"""
    return x * HARTREE_TO_KCALMOL
