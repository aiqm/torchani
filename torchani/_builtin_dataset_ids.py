from enum import Enum


class _DatasetId(Enum):
    TESTDATA = "TestData"
    TESTDATAIONS = "TestDataIons"
    TESTDATAFORCESDIPOLES = "TestDataForcesDipoles"
    IONSVERYHEAVY = "IonsVeryHeavy"
    IONSHEAVY = "IonsHeavy"
    IONSLIGHT = "IonsLight"
    ANI1Q = "ANI1q"
    ANI2QHEAVY = "ANI2qHeavy"
    ANI1CCX = "ANI1ccx"
    ANI1X = "ANI1x"
    ANI2X = "ANI2x"
    COMP6V1 = "COMP6v1"
    COMP6V2 = "COMP6v2"
    ANI1E = "ANI1e"


class _LotId(Enum):
    DEFAULT = "default"
    ALL = "all"
    B973C_DEF2MTZVP = "b973c-def2mtzvp"
    CCSD_PTP_STAR_CBS = "ccsd(t)star-cbs"
    WB97MD3BJ_DEF2TZVPP = "wb97md3bj-def2tzvpp"
    WB97MV_DEF2TZVPP = "wb97mv-def2tzvpp"
    WB97X_631GD = "wb97x-631gd"
    WB97X_DEF2TZVPP = "wb97x-def2tzvpp"
