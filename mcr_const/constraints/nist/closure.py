import inspect

import numpy as np
from pymcr.constraints import Constraint
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from mcr_const.constraints.nist.basic import NormMethod


class ConstraintWithTotalConcentration(Constraint):
    def __init__(self, line_indices=(), shift_guess=0, scale_guess=1.0, bounds=((-10, 10), (0.9, 1.1)),
                 method='trf', total_conc=None, copy=False, min_conc=1.0E-2, suppress_only=False,
                 specie_wise=False, peak_ranges=8):
        super(ConstraintWithTotalConcentration, self).__init__()
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
            result_y[result_y < self.min_conc] = self.min_conc
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
            tot_scale_vec[tot_scale_vec > 1.0] = 1.0
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
        super(ConstraintWithFunctionAndTotalConcentration, self).__init__()
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


class StoichiometricNorm(Constraint):
    def __init__(self, i_species, edge_start_end_indices, norm_method=NormMethod.TAIL_ONLY):
        super(StoichiometricNorm, self).__init__()
        assert isinstance(i_species, (list, tuple))
        assert isinstance(edge_start_end_indices, (list, tuple))
        assert isinstance(edge_start_end_indices[0], (list, tuple))
        assert len(edge_start_end_indices[0]) == 2
        self.i_species = i_species
        self.edge_indices = [np.r_[start, end] for start, end in edge_start_end_indices]
        self.norm_method = norm_method

    def transform(self, A):
        if self.copy:
            A = A.copy()
        active_A = A[self.i_species, :]
        edge_specs = [active_A[:, ei] for ei in self.edge_indices]
        edge_steps = []
        for spec_el in edge_specs:
            if self.norm_method == NormMethod.TAIL_ONLY:
                es = spec_el[:, -1]
            else:
                raise ValueError(f"Normalization method {NormMethod} hasn't been implemented yet")
            edge_steps.append(es)
        edge_steps = np.stack(edge_steps, axis=-1)
        total_edge_steps = edge_steps.sum(axis=-1)
        active_A = active_A * total_edge_steps[:, np.newaxis]
        A[self.i_species, :] = active_A
        return A
