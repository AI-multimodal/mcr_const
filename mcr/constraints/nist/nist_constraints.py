from pymcr.constraints import Constraint
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import curve_fit
import inspect
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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


class ConstraintWithFunction(Constraint):
    def __init__(self, line_indices=(), func=None, xgrid=None, initial_guess=None, bounds=(-np.inf, np.inf),
                 method='trf',copy=False):
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


class ConstraintWithTotalConcentration(Constraint):
    def __init__(self, line_indices=(), shift_guess=0, scale_guess=1.0, bounds=((-10, 10), (0.9, 1.1)),
                 method='trf', total_conc=None, copy=False, min_conc=1.0E-2, suppress_only=False,
                 specie_wise=False, peak_ranges=8):
        self.copy = copy
        self.bounds = bounds
        self.method = method
        self.shift = shift_guess
        self.scale = scale_guess
        self.line_indices = tuple(line_indices)
        self.fitting_result = []
        assert 'numpy.ndarray' in str(type(total_conc))
        self.total_conc = total_conc
        self.min_conc = min_conc
        self.suppress_only = suppress_only
        self.specie_wise = specie_wise
        if isinstance(peak_ranges, int):
            self.peak_ranges = [peak_ranges] * len(self.line_indices)
        else:
            self.peak_ranges = peak_ranges

    @staticmethod
    def shift_and_scale_conc(conc, shift, scale):
        x = np.arange(conc.shape[0])
        shifted_x = (x - shift) % (x.shape[0] - 1)
        total_pred = interp1d(x, conc, kind='cubic')(shifted_x)
        total_pred *= scale
        return total_pred

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
            assert p[0].shape == self.line_indices[0][0].shape
            assert len(p[0].shape) == 1
            assert len(A[p].shape) == 1

        total_conc_A = np.stack([A[p] for p in self.line_indices]).sum(axis=0)
        assert len(total_conc_A.shape) == 1
        if len(self.fitting_result) > 0:
            ig = self.fitting_result[-1]
        else:
            ig = [self.shift, self.scale]
        result_x, pcov = curve_fit(self.shift_and_scale_conc, self.total_conc, total_conc_A, p0=ig,
                                   bounds=list(zip(*self.bounds)), method=self.method)
        self.fitting_result.append(result_x)
        result_y = self.shift_and_scale_conc(self.total_conc, *result_x)
        if not self.specie_wise:
            total_conc_A[total_conc_A < self.min_conc] = self.min_conc
            result_y[result_y<self.min_conc] = self.min_conc
            tot_scale_vec = result_y / total_conc_A
        else:
            tot_scale_vec = []
            for p, pr in zip(self.line_indices, self.peak_ranges):
                i_a = np.argmax(A[p])
                if pr == 0:
                    i_t = i_a
                else:
                    i_t = np.argmax(result_y[i_a-pr: i_a+pr+1]) + i_a-pr
                tot_scale_vec.append(result_y[i_t]/A[p][i_a])
            tot_scale_vec = np.array(tot_scale_vec)
        if self.suppress_only:
            tot_scale_vec[tot_scale_vec>1.0] = 1.0
        for i, p in enumerate(self.line_indices):
            p = tuple(p)
            if not self.specie_wise:
                A[p] *= tot_scale_vec
            else:
                A[p] *= tot_scale_vec[i]
        return A


class ConstraintWithFunctionAndTotalConcentration(Constraint):
    def __init__(self, line_indices=(), gau_func=None, gau_initial_guess=None, gau_bounds=(-np.inf, np.inf),
                 gau_method='trf', gau_var_bounds_perc=0.1,
                 shift_scale_guess=(0.0, 1.0), shift_scale_bounds=((-10, 10), (0.9, 1.1)),
                 tot_method='trf', total_conc=None, copy=False):
        self.copy = copy
        assert inspect.isfunction(gau_func)
        sig = inspect.signature(gau_func)
        assert len(sig.parameters) - 1 == len(gau_initial_guess) // len(line_indices)
        assert len(gau_initial_guess) % len(line_indices) == 0
        self.param_per_gau = len(sig.parameters) - 1
        self.gau_func = gau_func
        self.gau_bounds = gau_bounds
        self.gau_method = gau_method
        self.gau_initial_guess = np.array(gau_initial_guess)
        self.gau_var_bounds_perc = gau_var_bounds_perc
        self.line_indices = tuple(line_indices)
        self.fitting_result = []

        self.shift_scale_bounds = shift_scale_bounds
        self.tot_method = tot_method
        self.shift, self.scale = shift_scale_guess

        assert 'numpy.ndarray' in str(type(total_conc))
        self.total_conc = total_conc

    @staticmethod
    def shift_and_scale_conc(conc, shift, scale):
        x = np.arange(conc.shape[0])
        shifted_x = (x - shift) % (x.shape[0] - 1)
        total_pred = interp1d(x, conc, kind='cubic')(shifted_x)
        total_pred *= scale
        return total_pred

    def combine_individual_func_pred(self, x, *gau_params):
        ind_pred_list = []
        for i in range(len(self.line_indices)):
            iparam1, iparam2 = i * self.param_per_gau, (i + 1) * self.param_per_gau
            ind = self.gau_func(x, *gau_params[iparam1: iparam2])
            ind_pred_list.append(ind)
        ind_pred_list = np.stack(ind_pred_list)
        return ind_pred_list.sum(axis=0)

    def transform(self, A):
        if self.copy:
            A = A.copy()
        if len(self.fitting_result) == 0:
            gau_ig = self.gau_initial_guess
            tot_ig = self.shift, self.scale
        else:
            gau_ig, tot_ig = self.fitting_result[-1]
        x = np.arange(A[self.line_indices[0]].shape[0])

        result_gau_indiv_conc_list_1 = []
        result_gau_params_1 = []
        for i, p in enumerate(self.line_indices):
            p = tuple(p)
            assert len(p) == 2
            assert 'numpy.ndarray' in str(type(p[0]))
            assert 'numpy.ndarray' in str(type(p[1]))
            assert p[0].shape == p[1].shape
            assert len(p[0].shape) == 1
            indiv_conc = A[p]
            assert indiv_conc.shape == x.shape
            assert len(indiv_conc.shape) == 1
            iparam1, iparam2 = i * self.param_per_gau, (i + 1) * self.param_per_gau
            result_gpar, pcov = curve_fit(self.gau_func, x, indiv_conc, p0=gau_ig[iparam1:iparam2],
                                          bounds=list(zip(*self.gau_bounds[iparam1:iparam2])), method=self.gau_method)
            indiv_conc_pred_1 = self.gau_func(x, *result_gpar)
            result_gau_indiv_conc_list_1.append(indiv_conc_pred_1)
            result_gau_params_1.extend(result_gpar)
        result_gau_indiv_conc_list_1 = np.stack(result_gau_indiv_conc_list_1)
        result_gau_params_1 = np.array(result_gau_params_1)

        gau_sum = result_gau_indiv_conc_list_1.sum(axis=0)
        result_tot_param, pcov = curve_fit(self.shift_and_scale_conc, self.total_conc, gau_sum, p0=tot_ig,
                                           bounds=list(zip(*self.shift_scale_bounds)), method=self.tot_method)
        tot_shifted = self.shift_and_scale_conc(self.total_conc, *result_tot_param)

        iter2_gau_ig = result_gau_params_1.copy()
        eps = 1.0E-20
        lb = iter2_gau_ig - (np.fabs(iter2_gau_ig) + eps) * self.gau_var_bounds_perc
        ub = iter2_gau_ig + (np.fabs(iter2_gau_ig) + eps) * self.gau_var_bounds_perc
        m_lb, m_ub = np.array(self.gau_bounds).T
        lb[lb < m_lb] = m_lb[lb < m_lb]
        ub[ub > m_ub] = m_ub[ub > m_ub]

        result_gau_params_2, pcov = curve_fit(self.combine_individual_func_pred, x, tot_shifted, p0=iter2_gau_ig,
                                              bounds=(lb, ub), method=self.gau_method)

        for i, p in enumerate(self.line_indices):
            p = tuple(p)
            iparam1, iparam2 = i * self.param_per_gau, (i + 1) * self.param_per_gau
            A[p] = self.gau_func(x, *result_gau_params_2[iparam1:iparam2])
        self.fitting_result.append((result_gau_params_2, result_tot_param))
        return A

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


class SharedGlobalPrefactor(object):
    def __init__(self, prefactor=None, total_conc=None, factor_initialized=None):
        self.prefactor = prefactor
        self.raw_total_concentration = total_conc
        self.factor_initialized = factor_initialized


class ConstraintGlobalPrefactor(Constraint):
    def __init__(self, stage, shared_prefactor: SharedGlobalPrefactor, sum_axis=-1, copy=False):
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


class ConstraintGuinier(Constraint):
    def __init__(self, line_indices, qgrid, qscale_vector, q_max=0.2, q_guinier=0.1, mix_ratio=1.0, default_rg=5.0,
                 guinier_smoothing_factor=1.0, linear_grad_thresh=1.0, plot_ax=None, color_list=None, copy=False):
        self.copy = copy
        self.line_indices = tuple(line_indices)
        self.q_max = q_max
        self.q_guinier = q_guinier
        self.qscale_vector = qscale_vector
        assert mix_ratio > 0.0 and mix_ratio <= 1.0
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
        vq = (qgrid < qh) & (xs >  intensity_lower_bound)
        spl = UnivariateSpline(qgrid[vq] ** 2, np.log(xs[vq]), k=2, s=guinier_smoothing_factor)
        linear_indices = np.fabs(spl.derivative(n=2)(qgrid[vq] ** 2)) * 1.0E-4 < linear_grad_thresh
        full_indices = np.arange(xs.shape[0])
        vi = full_indices[vq][linear_indices]
        vq[:] = False
        vq[vi] = True

        if qgrid[vq].shape[0] < min_points:
            return xs
        rg, i0 = np.polyfit(qgrid[vq] ** 2, np.log(xs[vq]), 1)
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
    def plot_fitting(nspecies, subsize=4, ylim=[1.0, 1.0E4], palette=("coolwarm", 300)):
        color_list = sns.color_palette(*palette)
        fig, ax_array = plt.subplots(nrows=1, ncols=nspecies, sharey=True, sharex=True, figsize=(nspecies * subsize, subsize))
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

