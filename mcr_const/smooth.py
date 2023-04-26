import numpy as np
from pymcr.constraints import Constraint
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from typing import Tuple, List, Union, Optional
from scipy.ndimage import gaussian_filter1d

from .basic import VarType


class ConstraintSmooth(Constraint):
    def __init__(self, line_indices: List[Tuple[np.ndarray, np.ndarray]], exponent: int = 5,
                 smoothing_factor: Optional[float] = None, knots: Optional[List[int]] = None, 
                 gaussian_sigma: Optional[float] = None, copy: bool = False):
        """
        Smoothing concentration evolution or spectra.

        :param line_indices: numpy indices to access the C or ST matrix, each region is tuple of of the index for two
                              dimension. Multiple region are supported.
        :param exponent: The spline order.
        :param smoothing_factor: Smoothing factor for spline. Should be None if knots or gaussian_sigma is set.
        :param knots: list of int. Interior knots of spline. Should be None if smoothing_factor or gaussian_sigma is set.
        :param gaussian_sigma: float. Width of Gaussian Smoothing Kernel. Should be None if smoothing_factor or knots is set.
        :param copy: Whether keep original matrix.
        """
        super(ConstraintSmooth, self).__init__()
        self.copy = copy
        self.line_indices = tuple(line_indices)
        self.exponent = exponent
        self.smoothing_factor = smoothing_factor
        self.knots = knots
        self.gaussian_sigma = gaussian_sigma
        total_num_smoothing_options = 0
        if self.smoothing_factor is not None:
            total_num_smoothing_options += 1
        if self.knots is not None:
            total_num_smoothing_options += 1
        if self.gaussian_sigma is not None:
            total_num_smoothing_options += 1
        assert total_num_smoothing_options == 1, \
            "Among smoothing_factor, knots and gaussian_sigma, " \
            "there should be one and only one active "

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
            if self.smoothing_factor is not None:
                spl = UnivariateSpline(x, y, k=self.exponent, s=self.smoothing_factor)
                y2 = spl(x)
            elif self.knots is not None:
                spl = LSQUnivariateSpline(x, y, self.knots, k=self.exponent)
                y2 = spl(x)
            else:
                y2 = gaussian_filter1d(y, sigma=self.gaussian_sigma)
            
            y2[y2 < 0] = 0.0
            A[p] = y2
        return A

    @classmethod
    def from_range(cls, i_specie: int, i_range: Tuple[int, int], exponent: int = 5, smoothing_factor: float = 1.0,
                   knots: Union[None, List[int]] = None,
                   var_type: VarType = VarType.CONCENTRATION) -> 'ConstraintSmooth':
        """
        Construct a ConstraintSmooth instance using specified range.

        :param i_specie: The index for targeting specie.
        :param i_range: The range in profile, express as tuple index along the time dimension. Single region only.
        :param exponent: The spline order.
        :param smoothing_factor: Smoothing factor for spline.
        :param knots: list of int. Interior knots of spline. This option is exclusive with smoothing_factor
        :param var_type: Apply to concentration or spectra matrix.
        :return: ConstraintSmooth instance
        """
        start, end = i_range
        ci = (np.r_[start: end],
              np.full(end - start, fill_value=i_specie))
        if var_type == VarType.SPECTRA:
            ci = tuple(reversed(ci))
        const = ConstraintSmooth(line_indices=[ci], exponent=exponent, smoothing_factor=smoothing_factor, knots=knots)
        return const


class ConstraintConstant(Constraint):
    def __init__(self, line_indices: List[Tuple[np.ndarray, np.ndarray]], copy: bool = False):
        """
        Make the profile a constance, the value of the constant is determined iteratively.

        :param line_indices: numpy indices to access the C or ST matrix, each region is tuple of of the index for two
                             dimension. Multiple region are supported.
        :param copy: Whether keep original matrix
        """
        super(ConstraintConstant, self).__init__()
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

    @classmethod
    def from_range(cls, i_specie: int, i_ranges: List[Tuple[int, int]], var_type: VarType = VarType.CONCENTRATION
                   ) -> 'ConstraintConstant':
        """
        Construct ConstraintConstant instance using specified range.
        :param i_specie: The index for targeting specie.
        :param i_ranges: The range in profile, express as tuple index along the time dimension. Multiple region are
                         supported.
        :param var_type: Apply to concentration or spectra matrix.
        :return:
        """
        index_list = []
        for start, end in i_ranges:
            ci = (np.r_[start: end],
                  np.full(end - start, fill_value=i_specie))
            if var_type == VarType.SPECTRA:
                ci = tuple(reversed(ci))
            index_list.append(ci)
        const = ConstraintConstant(line_indices=index_list)
        return const


class ConstraintElasticConstant(Constraint):
    def __init__(self, line_indices=(), copy=False, k=5, width=0.5):
        super(ConstraintElasticConstant, self).__init__()
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
