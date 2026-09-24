#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from pywarpx import picmi, callbacks

constants = picmi.constants
q_e = constants.q_e
m_e = constants.m_e
c = constants.c

# =======================================================
# 1. SIMULATION PARAMETERS
# =======================================================
impact_energy_eV = 100.0  # Initial mono-energetic electrons
gamma_initial = 1.0 + (impact_energy_eV * q_e) / (m_e * c**2)
u_initial = np.sqrt(gamma_initial**2 - 1.0) * c

neutral_density = 3e23    # Artificially high to force collisions quickly
neutral_temp = 300.0      # K
dt = 1e-12                # Short timestep
max_steps = 1

# Domain setup (Small 3D box, 1 cell, periodic)
grid = picmi.Cartesian3DGrid(
    number_of_cells=[8, 8, 8],
    lower_bound=[-1e-3, -1e-3, -1e-3],
    upper_bound=[1e-3, 1e-3, 1e-3],
    lower_boundary_conditions=['periodic', 'periodic', 'periodic'],
    upper_boundary_conditions=['periodic', 'periodic', 'periodic'],
)

solver = picmi.ElectromagneticSolver(grid=grid, method='Yee', cfl=1.0)

# =======================================================
# 2. SPECIES & COLLISION SETUP
# =======================================================
electrons = picmi.Species(
    particle_type='electron', 
    name='electrons',
    initial_distribution=picmi.UniformDistribution(
        density=1e14,
        directed_velocity=[0.0, 0.0, u_initial], # All propagating in +z
        rms_velocity=[0.0, 0.0, 0.0]             # Mono-energetic
    )
)

ions = picmi.Species(particle_type='Ar', name='ions', charge_state=1)

mcc_ionization = picmi.MCCCollisions(
    name="eii",
    species=electrons,
    background_density=neutral_density,
    background_temperature=neutral_temp,
    background_mass=ions.mass,
    scattering_processes={
        "ionization": {
            "cross_section": (
                "/home/vagrant/jupyter/StaffScratch/ncook882/"
                "electrostatic/warpx-data/MCC_cross_sections/Ar/ionization.dat"
            ),
            "energy": 15.7596112,
            "species": ions,
            "kinematics": "rudd",
            "electron_species": electrons,
        }
    },
)

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=1,
    time_step_size=dt,
    warpx_collisions=[mcc_ionization],
)

sim.add_species(electrons, layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=20000))
sim.add_species(ions, layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=0))
#sim.add_collision(mcc_ionization)

# Add diagnostic
field_diag = picmi.FieldDiagnostic(
    name='field_diag',
    grid=grid,
    period=max_steps,
    data_list=['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'],
    write_dir='diags',
    warpx_openpmd_backend='h5',
    warpx_format='openpmd'
)
sim.add_diagnostic(field_diag)

# Add diagnostic for particles instead of fields
part_diag = picmi.ParticleDiagnostic(
    name='part_diag',
    period=max_steps,
    species=[electrons],
    data_list=['x', 'y', 'z', 'ux', 'uy', 'uz'], # Added x, y, z
    write_dir='diags',
    warpx_openpmd_backend='h5',
    warpx_format='openpmd'
)
sim.add_diagnostic(part_diag)


# 1. Translate PICMI objects into native PyWarpX buckets (including 'eii')
sim.initialize_inputs()

sim.step(max_steps)

# =======================================================
# 3. ANALYSIS & PLOTTING
# =======================================================
ts = OpenPMDTimeSeries('./diags/part_diag/')

# Get electron momentum at the final step (returned as dimensionless u/c)
iteration = ts.iterations[-1]
ux, uy, uz = ts.get_particle(['ux', 'uy', 'uz'], species='electrons', iteration=iteration)

# Calculate kinetic energy in eV
u_sq = ux**2 + uy**2 + uz**2  # This is already (u/c)^2
gamma = np.sqrt(1.0 + u_sq)   
energy_eV = (gamma - 1.0) * m_e * c**2 / q_e

# Plot the spectrum
plt.figure(figsize=(8, 5))
plt.hist(energy_eV, bins=200, range=(0, 110), color='blue', alpha=0.7, edgecolor='black')
plt.axvline(x=impact_energy_eV, color='red', linestyle='--', label='Initial Energy (100 eV)')
plt.title("Electron Energy Spectrum after Rudd Impact Ionization")
plt.xlabel("Kinetic Energy [eV]")
plt.ylabel("Macroparticle Count")
plt.yscale('log') # Log scale helps see the distribution clearly
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("./rudd_ionization_spectrum.png", dpi=150)
plt.show()