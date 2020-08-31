import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple


def generate_synthetic_spectra(step_pos: int, sigma: float, peaks: List[Tuple[int, int, int]],
                               sin_start: int, sin_amplitidue: int, sin_period: int, sin_extra: int = 20,
                               num_points: int = 250):
    step_spec = np.ones(num_points)
    step_spec[:step_pos] = 0

    peak_spec = np.zeros(num_points)
    for pos, inten, width in peaks:
        peak_spec[pos - width // 2:pos - width // 2 + width] = inten

    sin_spec = np.zeros(num_points)
    sin_spec[sin_start:] = np.sin(np.arange(num_points - sin_start) / sin_period * 2 * math.pi) * \
        sin_amplitidue * \
        np.arange(num_points - sin_start + sin_extra, sin_extra, -1) / (num_points - sin_start + sin_extra)

    sharp_spec = step_spec + peak_spec + sin_spec
    syn_spec = gaussian_filter1d(sharp_spec, sigma, mode='nearest')
    return syn_spec
