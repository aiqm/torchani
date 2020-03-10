r"""Unit conversion factors

All values taken from:

Eite Tiesinga, Peter J. Mohr, David B. Newell, and Barry N. Taylor (2020), "The
2018 CODATA Recommended Values of the Fundamental Physical Constants" (Web
Version 8.1). Database developed by J. Baker, M. Douma, and S. Kotochigova.
Available at http://physics.nist.gov/constants, National Institute of Standards
and Technology, Gaithersburg, MD 20899.

except from the Joule-to-kcal conversion, taken from:

IUPAC. Compendium of Chemical Terminology, 2nd ed. (the "Gold Book"). Compiled
by A. D. McNaught and A. Wilkinson. Blackwell Scientific Publications, Oxford
(1997). Online version (2019-) created by S. J. Chalk. ISBN 0-9678550-9-8.
https://doi.org/10.1351/goldbook. 

where the thermochemical calorie was used
"""

HARTREE_TO_EV = 2.7211386245988e1 # uncertainty (53)
HARTREE_TO_JOULE = 4.3597447222071e-18 # uncertainty (85)
EV_TO_JOULE = 1.602176634e-19 # exact
JOULE_TO_KCAL = 1/4184. # exact

AVOGADROS_NUMBER = 6.02214076e23 # exact
SPEED_OF_LIGHT = 2.99792458e8 # exact

HARTREE_TO_KCALMOL = HARTREE_TO_JOULE * JOULE_TO_KCAL * AVOGADROS_NUMBER
HARTREE_TO_JOULEMOL = HARTREE_TO_JOULE * AVOGADROS_NUMBER

EV_TO_KCALMOL = EV_TO_JOULE * JOULE_TO_KCAL * AVOGADROS_NUMBER
EV_TO_JOULEMOL = EV_TO_JOULE * AVOGADROS_NUMBER


def hartree2ev(x):
    """Hartree to Electronvolt conversion factor
    taken directly from 2018 CODATA recommended values"""
    return x * HARTREE_TO_EV


def ev2joulemol(x):
    """Electronvolt to Joule/mole conversion factor
    taken directly from 2018 CODATA recommended values"""
    return x * EV_TO_JOULEMOL

def ev2kcalmol(x):
    """Electronvolt to (thermochemical) kilocalorie/mole conversion factor, derived
    from the Electronvolt-to-Joule conversion factor, Avogadro's number (both CODATA)
    and the Joule-to-kcal
    conversion factor (IUPAC)"""
    return x * EV_TO_KCALMOL

def hartree2joulemol(x):
    """Hartree to Joule/mole conversion factor
    taken directly from 2018 CODATA recommended values"""
    return x * HARTREE_TO_JOULEMOL

def hartree2kcalmol(x):
    """Hartree to (thermochemical) kilocalorie/mole conversion factor, derived
    from the Hartree-to-Joule conversion factor, Avogadro's number (both CODATA)
    and the Joule-to-kcal
    conversion factor (IUPAC)"""
    return x * HARTREE_TO_KCALMOL

