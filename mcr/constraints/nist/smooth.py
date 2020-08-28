from pymcr.constraints import Constraint
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import curve_fit
import inspect
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class ConstraintSmooth(Constraint):
    def __init__(self, line_indices=(), exponent=5, smoothing_factor=1.0, copy=False):
        self.copy = copy
        self.line_indices = tuple(line_indices)
        self.exponent = exponent
        self.smoothing_factor = smoothing_factor

    def transform(self, A):
        if self.copy:
            A = A.copy()
        for p in self.line_indices:
            p = tuple(p)
            assert len(p) == 2
            assert 'numpy.ndarray' in str(type(p[0]))
            assert 'numpy.ndarray' in str(type(p[1]))
            assert p[0].shape == p[1].shape
            assert len(p[0].shape) == 1
            x = np.arange(p[0].shape[0])
            y = A[p]
            spl = UnivariateSpline(x, y, k=self.exponent, s=self.smoothing_factor)
            y2 = spl(x)
            y2[y2<0] = 0.0
            A[p] = y2
        return A


class ConstraintConstant(Constraint):
    def __init__(self, line_indices=(), copy=False):
        self.copy = copy
        self.line_indices = tuple(line_indices)

    def transform(self, A):
        if self.copy:
            A = A.copy()
        for p in self.line_indices:
            p = tuple(p)
            assert len(p) == 2
            assert 'numpy.ndarray' in str(type(p[0]))
            assert 'numpy.ndarray' in str(type(p[1]))
            assert p[0].shape == p[1].shape
            assert len(p[0].shape) == 1
            y = A[p]
            A[p] = y.mean()
        return A


class ConstraintElasticConstant(Constraint):
    def __init__(self, line_indices=(), copy=False, k=5, width=0.5):
        self.copy = copy
        self.line_indices = tuple(line_indices)
        self.k = k
        self.width = width

    def transform(self, A):
        if self.copy:
            A = A.copy()
        for p in self.line_indices:
            p = tuple(p)
            assert len(p) == 2
            assert 'numpy.ndarray' in str(type(p[0]))
            assert 'numpy.ndarray' in str(type(p[1]))
            assert p[0].shape == p[1].shape
            assert len(p[0].shape) == 1
            y = A[p]
            x = np.full_like(y, fill_value=y.mean())

            phi = (y - x) / self.width
            scale = 1.0 / (1.0 + np.exp(-self.k * phi))
            scale = 2 * scale - 1.0
            excess_indices = np.fabs(scale) > np.fabs(phi)
            scale[excess_indices] = phi[excess_indices]
            scale = scale * self.width
            y2 = x + scale
            A[p] = y2
        return A
