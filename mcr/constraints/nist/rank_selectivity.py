from pymcr.constraints import Constraint
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import curve_fit
import inspect
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def generate_phase_law_constraint(nspecies, npos, interface_positions, threshold=1.0E-3):
    assert nspecies == len(interface_positions) + 2
    zero_positions = [np.r_[0:pre, post:npos]
                      for pre, post in zip(
                          [0, 0] + interface_positions,
                          interface_positions + [npos, npos])
                      ]
    zero_indices = [(zp, np.full_like(zp, fill_value=i))
                    for i, zp in enumerate(zero_positions)]
    c_zero_constraint = ConstraintPointBelow(point_indices=zero_indices, value=threshold)
    return c_zero_constraint


class ConstraintPointBelow(Constraint):
    def __init__(self, point_indices=((0, 0),), value=0.01, copy=False, retain_mean=True):
        self.copy = copy
        self.point_indices = point_indices
        self.value = value
        self.retain_mean = retain_mean

    def transform(self, A):

        if self.copy:
            A = A.copy()
        prev_mean = A.mean()
        for p in self.point_indices:
            if 'numpy.ndarray' in str(type(A[p])):
                if (A[p] > self.value).any():
                    p_above_index = A[p] > self.value
                    p_above = tuple([dim[p_above_index] for dim in p])
                    A[p_above] = self.value
            else:
                if A[p] > self.value:
                    A[p] = self.value
        if self.retain_mean:
            A *= prev_mean / A.mean()
        return A


class ConstraintMonotonic(Constraint):
    def __init__(self, line_indices=(), copy=False, descending=False):
        self.copy = copy
        self.line_indices = tuple(line_indices)
        self.descending = descending

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
            y2 = np.sort(y)
            if self.descending:
                y2 = y2[::-1]
            A[p] = y2
        return A


class ConstraintElasticBand(Constraint):
    def __init__(self, line_indices=(), copy=False, ref_line=None, k=5, width=0.5, bottom=True, top=True):
        self.copy = copy
        self.line_indices = tuple(line_indices)
        assert 'numpy.ndarray' in str(type(ref_line))
        self.ref_line = ref_line
        self.k = k
        self.width = width
        self.bottom = bottom
        self.top = top

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
            x = self.ref_line.copy()
            y = A[p]
            phi = (y-x)/self.width
            scale = 1.0/(1.0+np.exp(-self.k * phi))
            scale = 2*scale - 1.0
            excess_indices = np.fabs(scale) > np.fabs(phi)
            scale[excess_indices] = phi[excess_indices]
            if not self.top:
                excess_indices = phi > 0.0
                scale[excess_indices] = phi[excess_indices]
            if not self.bottom:
                excess_indices = phi < 0.0
                scale[excess_indices] = phi[excess_indices]
            scale = scale * self.width
            y2 = x + scale
            A[p] = y2
        return A