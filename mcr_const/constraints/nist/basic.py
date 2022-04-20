from enum import IntEnum


class VarType(IntEnum):
    CONCENTRATION = 1,
    SPECTRA = 2


class NormMethod(IntEnum):
    TAIL_ONLY = 1,
    EXTENDED_TAIL = 2,
    AVERAGE = 3
