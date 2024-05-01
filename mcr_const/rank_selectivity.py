import numpy as np
from pymcr.constraints import Constraint
from typing import List, Tuple

from .basic import VarType


class ConstraintPointBelow(Constraint):
    def __init__(self, point_indices: List[Tuple[np.ndarray, np.ndarray]], threshold: float = 1.0E-3,
                 copy: bool = False, retain_mean: bool = True,
                 ratio=1.0):
        """
        Push the values at specific region to be bellow threshold

        :param point_indices: numpy indices to access the C or ST matrix, each region is tuple of of the index for two
                              dimension. Multiple region are supported.
        :param threshold: The threshold to meet.
        :param copy: Whether keep original matrix
        :param retain_mean: Whether do scaling after pushing to maintain to the overall amplitudes of the profiles.
        """
        super(ConstraintPointBelow, self).__init__()
        self.copy = copy
        self.point_indices = point_indices
        self.threshold = threshold
        self.retain_mean = retain_mean
        self.ratio = ratio

    def transform(self, A):

        if self.copy:
            A = A.copy()
        prev_mean = A.mean()
        for p in self.point_indices:
            if 'numpy.ndarray' in str(type(A[p])):
                if (A[p] > self.threshold).any():
                    p_above_index = A[p] > self.threshold
                    p_above = tuple([dim[p_above_index] for dim in p])
                    A[p_above] = A[p_above] + (A[p_above] - self.threshold) * self.ratio
            else:
                if A[p] > self.threshold:
                    A[p] = A[p] + (self.threshold - A[p]) * self.ratio
        if self.retain_mean:
            A *= prev_mean / A.mean()
        return A

    @classmethod
    def from_phase_law(cls, n_species: int, sequence_length: int, interface_positions: List[int],
                       threshold: float = 1.0E-3, ratio: float = 1.0) -> 'ConstraintPointBelow':
        """
        Generate a constraint based on phase law. Design to be use for the concentration matrix.

        :param n_species: Number of species in MCR.
        :param sequence_length: The length of concentration profile. Unit: data points.
        :param interface_positions: The positions (index) that separates different phases.
        :param threshold: The threshold to meet for the constraint.
        :return: ConstraintPointBelow instance
        """
        assert n_species == len(interface_positions) + 2
        zero_positions = [np.r_[0:pre, post:sequence_length]
                          for pre, post in zip(
                [0, 0] + interface_positions,
                interface_positions + [sequence_length, sequence_length])
                          ]
        zero_indices = [(zp, np.full_like(zp, fill_value=i))
                        for i, zp in enumerate(zero_positions)]
        c_zero_constraint = ConstraintPointBelow(point_indices=zero_indices, 
                                                 threshold=threshold,
                                                 ratio=ratio)
        return c_zero_constraint

    @classmethod
    def from_range(cls, i_specie: int, i_ranges: List[Tuple[int, int]], threshold: float = 1.0E-3,
                   var_type: VarType = VarType.CONCENTRATION) -> 'ConstraintPointBelow':
        """
        Construct a ConstraintPointBelow instance using specified range.

        :param i_specie: The index for targeting specie.
        :param i_ranges: The range in profile, express as tuple index along the time dimension. Multiple region are
                         supported.
        :param threshold: The threshold to meet for the constraint.
        :param var_type: Apply to concentration or spectra matrix.
        :return: ConstraintPointBelow instance
        """
        index_list = []
        for start, end in i_ranges:
            ci = (np.r_[start: end],
                  np.full(end - start, fill_value=i_specie))
            if var_type == VarType.SPECTRA:
                ci = tuple(reversed(ci))
            index_list.append(ci)
        const = ConstraintPointBelow(point_indices=index_list, threshold=threshold)
        return const


class ConstraintFixedSegment(Constraint):
    def __init__(self, point_indices: List[Tuple[np.ndarray, np.ndarray]], targets: List[np.ndarray],
                 copy: bool = False):
        """
        Fix some regions to specified value

        :param point_indices: numpy indices to access the C or ST matrix, each region is tuple of of the index for two
                              dimension. Multiple region are supported.
        :param targets: fixed values as a list of numpy arrays. Should have same number of species as point_indices.
        :param copy: Whether keep original matrix
        """
        super(ConstraintFixedSegment, self).__init__()
        self.copy = copy
        self.point_indices = point_indices
        self.targets = targets
        assert len(self.targets) == len(self.point_indices)

    def transform(self, A):
        if self.copy:
            A = A.copy()
        for p, t in zip(self.point_indices, self.targets):
            A[p] = t
        return A

    
    @classmethod
    def from_range(cls, i_specie: int, i_ranges: List[Tuple[int, int]], targets: List[np.ndarray],
                   var_type: VarType = VarType.CONCENTRATION) -> 'ConstraintFixedSegment':
        """
        Construct a ConstraintFixedSegment instance using specified range and target.

        :param i_specie: The index for targeting specie.
        :param i_ranges: The range in profile, express as tuple index along the time dimension. Multiple region are
                         supported.
        :param targets: fixed values as a list of numpy arrays. Should have same number of species as point_indices.
        :param var_type: Apply to concentration or spectra matrix.
        :return: ConstraintFixedSegment instance
        """
        index_list = []
        for start, end in i_ranges:
            ci = (np.r_[start: end],
                  np.full(end - start, fill_value=i_specie))
            if var_type == VarType.SPECTRA:
                ci = tuple(reversed(ci))
            index_list.append(ci)
        const = ConstraintFixedSegment(point_indices=index_list, targets=targets)
        return const


class ConstraintMonotonic(Constraint):
    def __init__(self, line_indices=(), copy=False, descending=False):
        super(ConstraintMonotonic, self).__init__()
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
        super(ConstraintElasticBand, self).__init__()
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
            phi = (y - x) / self.width
            scale = 1.0 / (1.0 + np.exp(-self.k * phi))
            scale = 2 * scale - 1.0
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
