# Rudd impact-ionization regression draft

This follows the collision tests' native-input + Python-analysis + CTest layout.
It targets the Rudd implementation in this fork. It uses AMReX plotfiles and yt,
so neither Python bindings nor openPMD are required for the simulation.

Run from a configured 3D WarpX build with testing enabled:

```sh
ctest --test-dir build -R '^test_3d_ionization_rudd\.' --output-on-failure
```

For a manual run, copy `ionization.dat` into the working directory, run the
3D executable with `inputs_test_3d_ionization_rudd`, then run:

```sh
python /absolute/path/to/analysis_ionization_rudd.py diags/diag1000001
```

The one-step test initializes 65,536 electrons per incident population:

- `beam` creates `secondary` electrons and `ions`, exercising the explicit
  `ionization_electron_species` option with an oblique incident direction.
- `same` creates electrons in its own population and `same_ions`, exercising
  the default two-species configuration with incidence along z.
- `cold` starts below threshold and must remain unchanged without producing ions.

The synthetic total cross section is deliberately not an argon data fit. Its
plateau includes the 100 eV beam and it vanishes below the ionization threshold
and above 101 eV. This keeps the null-collision majorant near the beam frequency
and yields roughly 19,600 events per above-threshold population. Zero background
temperature, disabled pushing and deposition, and a single step isolate the
collision operator from transport, fields, thermal energy, and later cascades.

The analysis checks particle counts and routing, inherited weights, weighted
charge balance, electron kinetic energy plus ionization cost, zero ion momentum,
and the outgoing energy distributions. A conditional polar-angle CDF check for
secondary electrons tests scattering relative to the oblique beam. Statistical
bounds allow different CPU/GPU random streams; no particle ordering is assumed.
The energy CDF integrates the target density directly rather than reproducing
the rejection sampler. Its expression comes from the current branch, so it is
not independent validation against the Rudd/Johnson/Kim publication.
The secondary-energy CDF is normalized on `0 <= W <= (T-I)/2`; the primary
has energy `T-I-W`. For shared electron species, the analysis separates the
lower- and upper-energy populations before comparing their CDFs, avoiding
double-counting correlated pairs as independent samples.

## Validation still needed before upstream submission

The corrected analysis passed on the supplied `diag1000001` simulation output:
19,563 separate-species events, 19,493 shared-species events, and no
below-threshold ionization. The original draft incorrectly normalized the
secondary CDF over `0 <= W <= T-I`, producing a CDF discrepancy near 0.5.
Correcting the integration interval and reflecting the primary energies fixes
that error without changing assertion tolerances. Synthetic checks also reject
equal-sharing and uniform secondary-energy distributions.

The CTest checksum argument is currently `OFF`: no benchmark was fabricated.
After a trusted simulation run, enable the conventional
`analysis_default_regression.py --path diags/diag1000001` checksum argument and
generate the baseline with the repository's `CHECKSUM_RESET=ON` workflow when
approved. The physics analysis already runs independently of checksums.

This test does not cover the legacy `electron_species` alias, PICMI parameter
translation, relativistic energies, or high-energy primary/secondary azimuthal
correlation. Those are useful follow-up cases for broader model coverage.
