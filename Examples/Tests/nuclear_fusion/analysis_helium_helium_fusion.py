#!/usr/bin/env python3
# Copyright 2026
# License: BSD-3-Clause-LBNL

"""Check He3-He3 stoichiometry, conservation and the head-on fusion yield.

The yield reference checks implementation of the branch's piecewise model; it
is not an independent validation of the provenance of the high-energy table.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.constants as scc
import yt

# Match the masses and CODATA constants used by WarpX (ablastr/constant.H).
m_u = 1.66053906892e-27
m_e = 9.1093837139e-31
m_p = 1.67262192595e-27
m_he3 = 3.0160293201 * m_u - 2 * m_e
m_alpha = 4.00260325413 * m_u - 2 * m_e
q_value = (2 * m_he3 - m_alpha - 2 * m_p) * scc.c**2


def cross_section(energy_mev):
    """Piecewise model in barns; SF-III Eq. in Sec. V.2, arXiv:2405.06470."""
    energy = np.maximum(energy_mev, 1.0e-30)
    s_factor = 5.21 - 4.90 * energy + 11.21 * energy**2
    # Sommerfeld exponent for Z1 Z2 = 4 and reduced mass m_He3 / 2.
    fit_mass = 3.0160293 * m_u - 2 * m_e
    exponent = 4 * np.pi / 137.035999 * np.sqrt(
        fit_mass * scc.c**2 / (energy * 1.0e6 * scc.e)
    )
    low = s_factor / energy * np.exp(-exponent)
    # Use every supplied source point in the tabulated regime.
    source = np.loadtxt(
        Path(__file__).with_name("helium_helium_cross_section_endf.csv"), delimiter=","
    )
    energy_cm = source[:, 0] / 2e6
    tabulated = energy_cm >= 0.2
    high = np.interp(energy, energy_cm[tabulated], source[tabulated, 1])
    return np.where(energy_mev <= 0, 0, np.where(energy_mev <= 0.2, low, high))


def read_species(data, name, mass):
    weight = data[name, "particle_weight"].v
    momentum = np.stack(
        [data[name, f"particle_momentum_{axis}"].v for axis in "xyz"], axis=1
    )
    assert np.all(np.isfinite(weight)) and np.all(weight > 0)
    assert np.all(np.isfinite(momentum))
    p_squared = np.sum(momentum**2, axis=1)
    gamma = np.sqrt(1 + p_squared / (mass * scc.c) ** 2)
    # Stable kinetic energy avoids subtracting rest energy at low momentum.
    kinetic = p_squared / (mass * (gamma + 1))
    return weight, momentum, kinetic


def main():
    end_path = sys.argv[1]
    start = yt.load(end_path[:-4] + "0000")
    end = yt.load(end_path)
    initial, final = start.all_data(), end.all_data()
    dt = float(end.current_time - start.current_time)

    for case in range(1, 4):
        names = [f"He3_{case}a", f"He3_{case}b", f"alpha{case}", f"proton{case}"]
        masses = [m_he3, m_he3, m_alpha, m_p]
        before = [read_species(initial, name, mass) for name, mass in zip(names[:2], masses[:2])]
        after = [read_species(final, name, mass) for name, mass in zip(names, masses)]
        reaction_weight = after[2][0].sum()
        assert reaction_weight > 0
        np.testing.assert_allclose(after[3][0].sum(), 2 * reaction_weight, rtol=1e-12)
        assert after[3][0].size == 2 * after[2][0].size
        assert after[2][0].size % 2 == 0
        for old, new in zip(before, after[:2]):
            # Tiny depletion in case 1 is below floating-point resolution.
            np.testing.assert_allclose(
                old[0].sum(), new[0].sum() + reaction_weight, rtol=1e-12
            )

        energy_before = sum(np.dot(w, ke) for w, _, ke in before)
        energy_after = sum(np.dot(w, ke) for w, _, ke in after)
        np.testing.assert_allclose(
            energy_after, energy_before + q_value * reaction_weight, rtol=1e-10
        )
        momentum_before = sum(np.sum(w[:, None] * p, axis=0) for w, p, _ in before)
        momentum_after = sum(np.sum(w[:, None] * p, axis=0) for w, p, _ in after)
        scale = sum(np.sum(w * np.linalg.norm(p, axis=1)) for w, p, _ in after)
        np.testing.assert_allclose(
            momentum_after, momentum_before, rtol=1e-10, atol=1e-12 * scale
        )
        for name in names[2:]:
            identifiers = np.stack(
                [final[name, "particle_id"].v, final[name, "particle_cpu"].v], axis=1
            )
            assert len(np.unique(identifiers, axis=0)) == len(identifiers)

        if case == 1:
            # Identical counterstreaming momenta within each z slice.
            p_squared = m_he3 * (100e3 * scc.e) * np.arange(16) ** 2
            gamma = np.sqrt(1 + p_squared / (m_he3 * scc.c) ** 2)
            energy_cm = 2 * p_squared / (m_he3 * (gamma + 1))
            relative_speed = 2 * np.sqrt(p_squared) / (gamma * m_he3)
            expected = cross_section(energy_cm / (1e6 * scc.e)) * 1e-28 * relative_speed * 64 * dt
            observed = np.histogram(
                final[names[2], "particle_position_z"].v,
                bins=np.arange(17), weights=after[2][0],
            )[0]
            expected_events_per_slice = 64 * 10000 * 0.002
            # Five-sigma counting uncertainty, as in the two-product fusion tests.
            np.testing.assert_allclose(
                observed, expected, rtol=5 / np.sqrt(expected_events_per_slice), atol=0
            )
            counts = after[2][0].size / 2
            expected_events = 15 * expected_events_per_slice
            assert abs(counts - expected_events) < 5 * np.sqrt(expected_events)

    grids = [ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions) for ds in (start, end)]
    np.testing.assert_allclose(grids[0]["rho"].v, grids[1]["rho"].v, rtol=2e-11)


if __name__ == "__main__":
    main()
