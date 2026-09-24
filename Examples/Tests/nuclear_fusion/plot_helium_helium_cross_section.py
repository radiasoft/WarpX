#!/usr/bin/env python3
"""Compare the former and revised He3-He3 cross sections with the source export.

Based on the supplied fit_He3_crossection.py. Requires NumPy and matplotlib.
Run from any directory; use --output to select the PNG output path.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("He3_transition_comparison.png"))
    args = parser.parse_args()
    source = np.loadtxt(
        Path(__file__).with_name("helium_helium_cross_section_endf.csv"), delimiter=","
    )
    source_energy = source[:, 0] / 2e6  # Lab eV -> CM MeV
    source_sigma = source[:, 1]
    energy = np.unique(
        np.concatenate(
            [
                np.geomspace(0.003, 10, 1600),
                source_energy[source_energy > 0],
                [
                    0.2 * (1 - 1e-9), 0.2 * (1 + 1e-9),
                    0.4 * (1 - 1e-9), 0.4 * (1 + 1e-9),
                ],
            ]
        )
    )
    mass = 3.0160293 * 1.66053906892e-27 - 2 * 9.1093837139e-31
    factor = 4 * np.pi / 137.035999 * np.sqrt(
        mass * 299792458.0**2 / (1e6 * 1.602176634e-19)
    )
    penetration = np.exp(-factor / np.sqrt(energy)) / energy
    sf3 = (5.21 - 4.90 * energy + 11.21 * energy**2) * penetration
    uncertainty = np.sqrt(
        0.118 - 1.516 * energy + 14.037 * energy**2
        - 15.504 * energy**3 + 71.640 * energy**4
    ) * penetration
    old_grid = np.array(
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10]
    )
    old_values = np.interp(old_grid, source_energy, source_sigma)
    old = np.where(energy <= 0.4, sf3, np.interp(energy, old_grid, old_values))
    selected = source_energy >= 0.2
    new = np.where(
        energy <= 0.2, sf3,
        np.interp(energy, source_energy[selected], source_sigma[selected]),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax in axes:
        ax.plot(source_energy, source_sigma, "k.", label="Supplied ENDF export", zorder=5)
        ax.plot(energy, sf3, color="tomato", alpha=0.7, label="SF-III fit")
        ax.fill_between(
            energy, np.maximum(sf3 - uncertainty, 1e-40), sf3 + uncertainty,
            color="tomato", alpha=0.12, label=r"SF-III $\pm 1\sigma$",
        )
        ax.plot(energy, old, "--", color="gray", label="Previous: 0.4 MeV / 16 nodes")
        ax.plot(energy, new, color="royalblue", label="Revised: 0.2 MeV / 45 nodes")
        ax.axvline(0.2, color="royalblue", linestyle=":", alpha=0.6)
        ax.axvline(0.4, color="gray", linestyle=":", alpha=0.6)
        ax.set(
            xscale="log", yscale="log", xlabel="CM energy (MeV)",
            ylabel="Cross section (barn)",
        )
        ax.grid(True, which="both", alpha=0.2)
    axes[0].set(xlim=(0.003, 10), ylim=(1e-37, 1e3), title="Full energy range")
    axes[0].legend(fontsize=8)
    axes[1].set(
        xlim=(0.15, 0.55), ylim=(8e-5, 2e-2),
        title="Transition region: 1.60% versus 16.78% jump",
    )
    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
