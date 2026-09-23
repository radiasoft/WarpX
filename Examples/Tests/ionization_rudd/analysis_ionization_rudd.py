#!/usr/bin/env python3
# This file is part of WarpX.
# License: BSD-3-Clause-LBNL
"""One-step regression for Rudd sampling and secondary-electron routing."""

import sys

import numpy as np

# SI constants matching Source/ablastr/constant.H.
c = 299792458.0
e = 1.602176634e-19
m_e = 9.1093837139e-31

T = 100.0  # incident kinetic energy [eV]
I = 15.7596112  # ionization energy [eV]
MC2 = m_e * c**2 / e
N = 8**3 * 128
WEIGHT = 1.0e14 / N


def check_cdf(samples, cdf):
    """DKW bound with failure probability <= 1e-8 per independent sample set."""
    values = np.sort(cdf(np.asarray(samples)))
    n = len(values)
    assert n > 1000, f"Insufficient collision statistics: {n}"
    distance = max(
        np.max(np.arange(1, n + 1) / n - values),
        np.max(values - np.arange(n) / n),
    )
    limit = np.sqrt(np.log(2.0e8) / (2 * n))
    assert distance < limit, f"CDF distance {distance} exceeds {limit}"


def energy_cdf():
    """Integrate the secondary-electron SDCS on its sampling interval.

    This evaluates the target density directly, not the rejection algorithm.
    It is a regression reference for RuddImpactIonization.H, not an independent
    validation of the branch's transcription of the published model.
    """
    # At the largest proposal random_f = f1*(T-I)/(f2*(T+I)),
    # W = I*f2*random_f/(f1-f2*random_f) = (T-I)/2.
    # The primary retains T-I-W; it does not follow this same energy CDF.
    w = np.linspace(0, (T - I) / 2, 20001)
    a, b = w + I, T - w
    density = (
        1 / a**2
        + 4 * I / (3 * a**3)
        + 1 / b**2
        + 4 * I / (3 * b**3)
        - MC2 * (2 * T + MC2) / ((T + MC2) ** 2 * a * b)
        + 1 / (T + MC2) ** 2
    )
    assert np.all(density > 0)
    integral = np.concatenate(
        ([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(w)))
    )
    return lambda energy: np.interp(energy, w, integral / integral[-1])


def read_species(ds, data, species):
    if (species, "particle_weight") not in ds.field_list:
        return np.empty(0), np.empty((0, 3)), np.empty(0)
    weight = data[species, "particle_weight"].v
    momentum = np.column_stack(
        [data[species, f"particle_momentum_{axis}"].to_value("kg*m/s") for axis in "xyz"]
    )
    assert np.all(np.isfinite(momentum)), species
    np.testing.assert_allclose(weight, WEIGHT, rtol=2e-6)
    u2 = np.sum((momentum / (m_e * c)) ** 2, axis=1)
    # Avoid cancellation in gamma-1 for low-energy electrons.
    energy = MC2 * u2 / (np.sqrt(1 + u2) + 1)
    return weight, momentum, energy


def main(path):
    import yt

    ds = yt.load(path)
    data = ds.all_data()
    cdf = energy_cdf()
    secondary = read_species(ds, data, "secondary")
    for electrons, ions, separate in [
        ("beam", "ions", True),
        ("same", "same_ions", False),
    ]:
        weight, momentum, energy = read_species(ds, data, electrons)
        ion_weight, ion_momentum, _ = read_species(ds, data, ions)
        events = len(ion_weight)
        assert events > 1000
        assert len(weight) == N + (0 if separate else events)
        np.testing.assert_allclose(ion_momentum, 0, atol=0)
        # The null-collision majorant is close to n*sigma*sqrt(2*T/m).
        # Allow 0.2% absolute probability for scan discretization and u vs v.
        probability = -np.expm1(
            -3e24 * 2e-20 * np.sqrt(2 * T * e / m_e) * 1e-12
        )
        count_limit = 6 * np.sqrt(N * probability * (1 - probability)) + 0.002 * N
        assert abs(events - N * probability) < count_limit
        collided = energy < T - I / 2
        assert np.count_nonzero(~collided) == N - events
        np.testing.assert_allclose(energy[~collided], T, rtol=2e-6)
        outgoing = energy[collided]
        assert np.all((outgoing >= 0) & (outgoing <= T - I + 1e-3))
        total_energy = np.dot(weight, energy)
        total_weight = weight.sum()
        if separate:
            sw, sp, se = secondary
            assert len(sw) == events, "Secondary electrons routed to wrong species"
            assert np.all((se >= 0) & (se <= (T - I) / 2 + 1e-3))
            assert np.all(outgoing >= (T - I) / 2 - 1e-3)
            total_energy += np.dot(sw, se)
            total_weight += sw.sum()
            check_cdf(se, cdf)
            check_cdf(T - I - outgoing, cdf)
            # Conditional polar-angle CDF must be uniform relative to the
            # oblique incident direction; catches incorrect basis rotations.
            mu = sp @ (np.array([1, 2, 2]) / 3) / np.linalg.norm(sp, axis=1)
            g2 = np.sqrt(
                (se + I) * (T + 2 * MC2) / ((se + I + 2 * MC2) * T)
            )
            g3 = (
                0.6
                * (MC2 / (T + MC2)) ** 2
                * np.sqrt(
                    I * np.maximum(1 - (se + I) / T, 0) / np.maximum(se, 1e-12)
                )
            )
            low = np.arctan2(-1 - g2, g3)
            high = np.arctan2(1 - g2, g3)
            uniform = (np.arctan2(mu - g2, g3) - low) / (high - low)
            check_cdf(uniform, lambda x: x)
        else:
            # In the shared species, separate the lower-energy secondaries
            # from primaries. Each event contributes one sample to each CDF;
            # pooling would incorrectly treat correlated pairs as independent.
            lower = outgoing <= (T - I) / 2
            assert np.count_nonzero(lower) == events
            assert np.count_nonzero(~lower) == events
            check_cdf(outgoing[lower], cdf)
            check_cdf(T - I - outgoing[~lower], cdf)
        np.testing.assert_allclose(
            total_weight - ion_weight.sum(), N * WEIGHT, rtol=2e-6
        )
        np.testing.assert_allclose(
            total_energy + I * ion_weight.sum(), N * WEIGHT * T, rtol=2e-6
        )
        print(f"{electrons}: {events} events; counts, charge, energy and SDCS passed")

    cold_weight, _, cold_energy = read_species(ds, data, "cold")
    assert len(cold_weight) == N
    np.testing.assert_allclose(cold_energy, 10, rtol=2e-6)
    assert len(read_species(ds, data, "cold_ions")[0]) == 0
    print("Below-threshold population unchanged")


if __name__ == "__main__":
    main(sys.argv[1])
