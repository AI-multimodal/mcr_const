from pymcr.constraints import Constraint


class SharedGlobalPrefactor(object):
    def __init__(self, prefactor=None, total_conc=None, factor_initialized=None):
        self.prefactor = prefactor
        self.raw_total_concentration = total_conc
        self.factor_initialized = factor_initialized


class ConstraintGlobalPrefactor(Constraint):
    def __init__(self, stage, shared_prefactor: SharedGlobalPrefactor, sum_axis=-1, copy=False):
        super(ConstraintGlobalPrefactor, self).__init__()
        self.copy = copy
        assert stage in [1, 2]
        self.stage = stage
        self.shared_prefactor = shared_prefactor
        assert sum_axis == -1
        self.sum_axis = sum_axis

    def transform(self, A):
        ssp = self.shared_prefactor
        if self.copy:
            A = A.copy()
        if self.stage == 1:
            ssp.raw_total_concentration = A.sum(axis=self.sum_axis)
            ssp.raw_total_concentration.clip(0)

            assert ssp.prefactor is not None or not ssp.factor_initialized
            if ssp.factor_initialized:
                if self.sum_axis == -1:
                    A = (A.T / ssp.prefactor).T
                else:
                    A = A / ssp.prefactor
        elif self.stage == 2:
            ssp.prefactor = ssp.raw_total_concentration / A.sum(axis=self.sum_axis).clip(1.0E-10)
            if self.sum_axis == -1:
                A = (A.T * ssp.prefactor).T
            else:
                A = A * ssp.prefactor
        ssp.factor_initialized = True
        return A
