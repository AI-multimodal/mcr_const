import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple


def generate_synthetic_spectra(step_pos: int, sigma: float, peaks: List[Tuple[int, float, int]],
                               sin_start: int, sin_amplitidue: float, sin_period: int, sin_extra: int = 20,
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


def generate_spectra_sequence():
    base_spec_1 = generate_synthetic_spectra(
        step_pos=61,
        sigma=8,
        peaks=[(20, 0.01, 2),
               (95, 0.8, 33),
               (115, 0.3, 37)
               ],
        sin_start=120,
        sin_amplitidue=0.2,
        sin_period=50
    )

    base_spec_2 = generate_synthetic_spectra(
        step_pos=62,
        sigma=5,
        peaks=[(25, 8.0, 1),
               (65, 1.0, 30),
               (80, -0.2, 35),
               (120, 0.4, 45)
               ],
        sin_start=80,
        sin_amplitidue=0.15,
        sin_period=40
    )

    base_spec_3 = generate_synthetic_spectra(
        step_pos=65,
        sigma=4,
        peaks=[(15, 0.5, 1),
               (28, 0.8, 1),
               (69, 0.7, 35),
               (87, 0.2, 55),
               (130, -0.1, 25)
               ],
        sin_start=89,
        sin_amplitidue=0.3,
        sin_period=70
    )

    base_spec_4 = generate_synthetic_spectra(
        step_pos=67,
        sigma=2.5,
        peaks=[(19, 0.8, 1),
               (26, 1.0, 1),
               (33, 0.8, 1),
               (53, 0.5, 23),
               (67, 0.8, 23),
               (83, -0.5, 15),
               (110, -0.1, 25)
               ],
        sin_start=76,
        sin_amplitidue=0.4,
        sin_period=57
    )

    base_specs = np.stack([base_spec_1, base_spec_2, base_spec_3, base_spec_4])
    syn_atom_counts = np.array([[13, 0, 11, 3],
                                [0, 11, 1, 1],
                                [5, 1, 7, 0],
                                [2, 15, 0, 17]])
    syn_atom_weights = (syn_atom_counts.T / syn_atom_counts.sum(axis=1)).T
    weighted_specs = syn_atom_weights @ base_specs

    num_concs = 60
    n_species = 4
    conc_breaker_1 = 19
    conc_breaker_2 = 37

    concs = np.zeros([num_concs, n_species])
    concs[:conc_breaker_1, 0] = np.arange(conc_breaker_1, 0, -1)
    concs[:conc_breaker_1, 1] = np.arange(conc_breaker_1)
    concs[conc_breaker_1:conc_breaker_2, 1] = np.arange(conc_breaker_2 - conc_breaker_1, 0, -1)
    concs[conc_breaker_1:conc_breaker_2, 2] = np.arange(conc_breaker_2 - conc_breaker_1)
    concs[conc_breaker_2:, 2] = np.arange(num_concs - conc_breaker_2, -1, -1.05)
    concs[conc_breaker_2:, 3] = np.arange(num_concs - conc_breaker_2)
    concs = (concs.T / concs.sum(axis=1)).T

    syn_sequence = concs @ weighted_specs

    n_noisy = 2
    noisy_concs = np.zeros([num_concs, n_noisy])
    noisy_concs[:, 0] = np.arange(num_concs) ** 2 / ((num_concs - 1) ** 2)
    noisy_concs[:, 0] += np.random.random(num_concs) * 0.3
    noisy_concs[:, 1] = 1.0 - noisy_concs[:, 0]
    noisy_seq = noisy_concs @ base_specs[:n_noisy]

    return base_specs, weighted_specs, syn_sequence, noisy_seq


if __name__ == '__main__':
    _, syn_specs, spectra_sequence, noisy_sequence = generate_spectra_sequence()
    np.savetxt("true_4_specs.csv", syn_specs, fmt='%8.6f', delimiter='\t')
    np.savetxt("spectra_4_sequence.csv", spectra_sequence, fmt='%8.6f', delimiter='\t')
    np.savetxt("spectra_2_noisy_sequence.csv", noisy_sequence, fmt='%8.6f', delimiter='\t')
