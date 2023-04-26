import inspect
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pymcr.constraints import Constraint
from scipy import special
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from typing import Callable


def exp_mod_gaussian(x, area, center, width, distortion):
    if distortion / width < 0.001:
        distortion = width * 0.001
    a = area, center, width, distortion
    part_pre = a[0]/(2*a[3])
    part_exp = np.exp(a[2]**2/(2*a[3]**2) + (a[1]-x)/a[3])
    part_erf = special.erf((x-a[1])/(2**0.5 * a[2]) - a[2]/(2**0.5*a[3])) + a[3]/math.fabs(a[3])
    return part_pre * part_exp * part_erf


def halfgau_mod_gaussian(x, area, center, width, distortion):
    if distortion / width < 0.001:
        distortion = width * 0.001
    a = area, center, width, distortion
    part_exp = a[0] * np.exp(-0.5*(x-a[1])**2 / (a[3]**2 + a[2]**2))
    part_erf = 1.0 + special.erf(a[3]*(x-a[1]) / (2**0.5 * a[2] * (a[3]**2 + a[2]**2)**0.5))
    part_div = (2*math.pi)**0.5 * (a[3]**2 + a[2]**2)**0.5
    return (part_exp * part_erf) / part_div


def exp_n_halfgau_mod_gaussian(x, area, center, width, distortion1, distortion2):
    if distortion1 / width < 0.001:
        distortion1 = width * 0.001
    if distortion2 / width < 0.001:
        distortion2 = width * 0.001
    a = area, center, width, distortion1, distortion2
    part1_exp = (a[0]/(4*a[3])) * np.exp((2*a[1]*a[3] - 2*a[3]*x + a[2]**2)/(a[3]**2))
    part1_erfc = special.erfc((a[1]*a[3] - a[3]*x + a[2]**2) / (2**0.5 * a[2] * a[3]))
    part2_pre = a[0]/(2 * (2*math.pi)**0.5 * (a[2]**2+a[4]**2)**0.5)
    part2_exp = np.exp(-0.5 * (a[1]-x)**2 / (a[2]**2 + a[4]**2))
    part2_erfc = special.erfc(a[4] * (a[1]-x) / (2**0.5 * a[2] * (a[2]**2 + a[4]**2)))
    return part1_exp * part1_erfc + part2_pre * part2_exp * part2_erfc


class ConstraintWithFunction(Constraint):
    def __init__(self, line_indices=(), func: Callable = None, xgrid=None, initial_guess=None, bounds=(-np.inf, np.inf),
                 method='trf', copy=False):
        super(ConstraintWithFunction, self).__init__()
        self.copy = copy
        assert inspect.isfunction(func)
        sig = inspect.signature(func)
        assert len(sig.parameters) - 1 == len(initial_guess)
        self.func = func
        self.xgrid = xgrid
        self.bounds = bounds
        self.method = method
        self.initial_guess = np.array(initial_guess)
        self.line_indices = tuple(line_indices)
        self.fitting_result = []

    def transform(self, A):
        if self.copy:
            A = A.copy()
        self.fitting_result = []
        for p in self.line_indices:
            p = tuple(p)
            assert len(p) == 2
            assert 'numpy.ndarray' in str(type(p[0]))
            assert 'numpy.ndarray' in str(type(p[1]))
            assert p[0].shape == p[1].shape
            assert len(p[0].shape) == 1
            y = A[p]
            assert len(y.shape) == 1
            x = self.xgrid if self.xgrid is not None else np.arange(y.shape[0])
            result_x, pcov = curve_fit(self.func, x, y, p0=self.initial_guess.copy(),
                                       bounds=list(zip(*self.bounds)), method=self.method)
            self.fitting_result.append(result_x)
            result_y = self.func(x, *result_x)
            A[p] = result_y
        return A


class ConstraintGuinier(Constraint):
    def __init__(self, line_indices, qgrid, qscale_vector, q_max=0.2, q_guinier=0.1, mix_ratio=1.0, default_rg=5.0,
                 guinier_smoothing_factor=1.0, linear_grad_thresh=1.0, plot_ax=None, color_list=None, copy=False):
        super(ConstraintGuinier, self).__init__()
        self.copy = copy
        self.line_indices = tuple(line_indices)
        self.q_max = q_max
        self.q_guinier = q_guinier
        self.qscale_vector = qscale_vector
        assert 0.0 < mix_ratio <= 1.0
        self.mix_ratio = mix_ratio
        self.default_rg = default_rg
        self.qgrid = qgrid
        self.guinier_smoothing_factor = guinier_smoothing_factor
        self.linear_grad_thresh = linear_grad_thresh
        self.min_points = 10
        self.plot_ax = plot_ax
        self.color_idx = 0
        self.color_list = color_list

    @staticmethod
    def fit_guinier_spec(qgrid, xs, qh, default_rg, guinier_smoothing_factor=1.0, linear_grad_thresh=1.0, min_points=10,
                         ax=None, color_list=None, color_idx=0, intensity_lower_bound=1.0E-10):
        vq = (qgrid < qh) & (xs > intensity_lower_bound)
        spl = UnivariateSpline(qgrid[vq] ** 2, np.log(xs[vq]), k=2, s=guinier_smoothing_factor)
        linear_indices = np.fabs(spl.derivative(n=2)(qgrid[vq] ** 2)) * 1.0E-4 < linear_grad_thresh
        full_indices = np.arange(xs.shape[0])
        vi = full_indices[vq][linear_indices]
        vq[:] = False
        vq[vi] = True

        if qgrid[vq].shape[0] < min_points:
            return xs
        # noinspection PyTupleAssignmentBalance
        rg, i0 = np.polyfit(x=qgrid[vq] ** 2,
                            y=np.log(xs[vq]),
                            deg=1)
        i0 = np.exp(i0)
        if rg < 0:
            rg = np.sqrt(-rg * 3.)
        else:
            rg = default_rg
        gspec = i0 * np.exp(-((qgrid * rg) ** 2) / 3.)
        if ax is not None:
            color = color_list[color_idx] if color_list is not None else None
            ax.loglog(qgrid, gspec, lw=0.5, c=color)
            ax.loglog(qgrid, xs, "--", lw=0.5, c=color)
            ax.loglog(qgrid[vq], gspec[vq], ".", ms=5.0, c=color)
        return gspec

    @staticmethod
    def plot_fitting(nspecies, subsize=4, ylim=(1.0, 1.0E4), palette=("coolwarm", 300)):
        color_list = sns.color_palette(*palette)
        # noinspection PyTypeChecker
        fig, ax_array = plt.subplots(nrows=1, ncols=nspecies, sharey=True, sharex=True,
                                     figsize=(nspecies * subsize, subsize))
        if nspecies == 1:
            ax_array = [ax_array]
        ax_array[0].set_ylim(ylim)
        return ax_array, color_list

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
            ospec = A[p] / self.qscale_vector
            assert ospec.shape[0] == self.qgrid.shape[0]
            gspec = self.fit_guinier_spec(self.qgrid, ospec, self.q_max, self.default_rg,
                                          ax=self.plot_ax, min_points=self.min_points,
                                          guinier_smoothing_factor=self.guinier_smoothing_factor,
                                          linear_grad_thresh=self.linear_grad_thresh,
                                          color_list=self.color_list, color_idx=self.color_idx)
            pspec = ospec.copy()
            pspec[self.qgrid < self.q_guinier] = gspec[self.qgrid < self.q_guinier]
            A[p] = (pspec ** self.mix_ratio) * (ospec ** (1.0 - self.mix_ratio)) * self.qscale_vector
            if self.plot_ax is not None:
                color = self.color_list[self.color_idx] if self.color_list is not None else None
                self.plot_ax.loglog(self.qgrid, A[p] / self.qscale_vector, "2", alpha=0.2, c=color)
        self.color_idx += 1
        return A
