import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple


def generate_synthetic_spectra(step_pos: int, sigma: float, peaks: List[Tuple[int, int, int]], num_points: int = 250):
    step_spec = np.ones(num_points)
    step_spec[:step_pos] = 0

    peak_spec = np.zeros(num_points)
    for pos, inten, width in peaks:
        peak_spec[pos - width//2:pos - width//2 + width] = inten

    sharp_spec = step_spec + peak_spec
    syn_spec = gaussian_filter1d(sharp_spec, sigma, mode='nearest')
    return syn_spec
