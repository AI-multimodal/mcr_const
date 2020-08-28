import numpy as np
from scipy import special
import math


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
