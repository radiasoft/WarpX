Nuclear fusion tests
====================

Helium-3 fusion
---------------

The D-He3 regression checks fusion yield, conservation, product multiplicity,
and placement of protons at deuterium positions and alphas at helium3 positions.
Its two cases use opposite product orders to exercise both placement mappings.
The He3-He3 regression checks head-on fusion yield and particle counts, plus
stoichiometry, energy, momentum, identifiers and charge conservation for head-on,
beam-target and thermal cases. It does not validate the three-body spectral model
against measured proton spectra.

Run the fusion regressions with a 3D MPI build (openPMD is also needed for the
anisotropic cases):

.. code-block:: bash

   cmake -S . -B build -DWarpX_DIMS=3 -DWarpX_MPI=ON -DWarpX_OPENPMD=ON -DBUILD_TESTING=ON
   cmake --build build -j 8
   ctest --test-dir build -N -R 'test_3d_.*fusion'
   ctest --test-dir build --output-on-failure -R 'test_3d_.*fusion' -E '\.checksum$'

The new helium tests require NumPy, SciPy and yt for analysis. The anisotropic
analyses additionally require openPMD-viewer and matplotlib. New checksum
baselines must be generated and reviewed separately; these tests have physics
analyses that run independently of checksum comparison. Measure runtime on the
CI CPU configuration before enabling these tests in a merge-ready PR.

Cross-section reference and outstanding model review
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The low-energy polynomial follows `Solar Fusion III, Section V.2
<https://arxiv.org/html/2405.06470v2#S5.SS2>`__.
``helium_helium_cross_section_endf.csv`` preserves the numerical values in the
user-supplied ``ENDF_He3.txt`` export, converting ENDF-style exponents to ordinary
scientific notation. The original export SHA-256 is
``2fc08c9826dba615f63f5747e25503efa13122382b90e9e7a3b5894460534f53``.
For ENDF/B-VII.1 data, see https://www.nndc.bnl.gov/sigma/index.jsp.

The production table includes all 45 export points from 0.2 through 10 MeV CM,
using ``E_cm = E_lab / 2`` and linear interpolation between adjacent points.
This removes the additional interpolation error of the former 16-point subset.
The transition from SF-III to the table is at 0.2 MeV CM: the SF-III fit gives
approximately 0.000446492 b and the table starts at 0.000439370 b.
The downward discontinuity is approximately 1.60%, reduced from 16.78% at the
previous 0.4 MeV cutoff. The fits are not rescaled or blended, so a small jump
remains; this choice preserves the supplied table values.
Above 10 MeV CM, the model holds the final table value constant; this is an
extrapolation policy, not additional evaluated data.

To compare the former 0.4 MeV / 16-point model with the revised 0.2 MeV /
45-point model, run the following with NumPy and matplotlib installed:

.. code-block:: bash

   python Examples/Tests/nuclear_fusion/plot_helium_helium_cross_section.py --output He3_transition_comparison.png

The plot includes the SF-III uncertainty band and a close-up of both transition
energies. The C++ table remains embedded for GPU use, with values taken directly
from the committed export rather than a separate high-energy fit.

Anisotropic D-D and D-T beam-target fusion
-------------------------------------------

The automated anisotropic beam-target tests exercise the energy-dependent angular distribution of fusion products at the deuterium beam momentum ``beta*gamma = 0.1``.
The two additional momenta ``0.01`` and ``0.05`` can be run locally for both reactions to reproduce the quantities plotted in Figure 2 of `van de Wetering et al. (2025) <https://doi.org/10.1103/zwjx-jbxl>`__.

Each CTest analysis validates its run independently using integral neutron spectrum observables and kinematic consistency checks.
The tests use 20,000 particles per cell to reduce their runtime.
For a closer match to the benchmark figure in the paper, use 40,000 particles per cell for both reactant species in the D-D and D-T base input files.

From the WarpX source directory, run all six simulations directly with a 3D MPI-enabled WarpX executable.
The following commands use each reaction's 10% input file as a template and override ``deuterium_beam.momentum_function_uz`` on the command line.
They use the same two MPI ranks and runtime parameters as the automated tests and replace any existing output directories with the same names under ``build/bin``:

.. code-block:: bash

   # Reproduce the anisotropic D-D and D-T fusion plots in Figure 2 of the PRE paper.
   # Run from the WarpX root directory after building a 3D MPI-enabled executable.
   warpx_dir=$(pwd)
   test_dir="${warpx_dir}/Examples/Tests/nuclear_fusion"
   analysis_script="${test_dir}/analysis_fusion_anisotropic_beam_target.py"
   warpx_executable=$(find "${warpx_dir}/build/bin" -maxdepth 1 -type f -name 'warpx.3d*' -executable -print -quit)

   for reaction in deuterium_deuterium deuterium_tritium; do
     diag_dirs=()
     for momentum in 1 5 10; do
       test_name="test_3d_${reaction}_fusion_anisotropic_beam_target_uz_${momentum}pct"
       input_name="inputs_test_3d_${reaction}_fusion_anisotropic_beam_target_uz_10pct"
       printf -v beam_momentum '0.%02d' "${momentum}"
       run_dir="${warpx_dir}/build/bin/${test_name}"
       cmake -E remove_directory "${run_dir}"
       cmake -E make_directory "${run_dir}"
       (
         cd "${run_dir}"
         OMP_NUM_THREADS=1 AMREX_INPUTS_FILE_PREFIX="${test_dir}/" \
           mpiexec -n 2 "${warpx_executable}" "${input_name}" \
           "deuterium_beam.momentum_function_uz(x,y,z)=${beam_momentum}" \
           amrex.abort_on_unused_inputs=1 amrex.throw_exception=1 amrex.the_arena_init_size=0 \
           warpx.always_warn_immediately=1 warpx.do_dynamic_scheduling=0 warpx.serialize_initial_conditions=1
       )
       diag_dirs+=("${run_dir}/diags/diag1")
     done
     python "${analysis_script}" --plot "${diag_dirs[@]}"
   done

The script writes ``deuterium_deuterium_fusion_anisotropic_beam_target_neutron_spectrum.png`` and ``deuterium_tritium_fusion_anisotropic_beam_target_neutron_spectrum.png`` in the current directory.
Solid curves show the normalized neutron energy spectra; dashed curves show the weighted mean center-of-momentum-frame emission angle in each energy bin.
