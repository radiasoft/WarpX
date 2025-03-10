#!/usr/bin/env python3
#
# --- Input file for MCC testing

import os
os.environ["OMP_NUM_THREADS"] = "2"

from pywarpx import picmi, callbacks
import pywarpx
import numpy as np

import scipy.constants
constants = picmi.constants

from scipy.optimize import brentq
from rsbeams.rsstats import kinematic
import rsfusion.injection
import rsfusion.diagnostics
import rsfusion.magnetic_field
import rsfusion.util
from rsfusion.picmi import SpeciesWithType, NuclearFusion, Coulomb_multi_init, GenerateBeams, GenerateInjectors
from rsfusion.injection import BeamInjection

##########################
# Parameters to Set
##########################

# resample_rate = 3000
resample_max_avg_ppc = 4 
resampling_min_ppc = 1
resample_cut = 1.2

self_consistent_fields = True
n_beams = 8
fusion_multiplier = 1e20

interactions = {'coulomb' : True, 
                'ndt' : 1, #period to call rxns
                'fusion' : True, 
               }

diagnostics = {'directory' : f'arx2_resample_mppc{resample_max_avg_ppc}_c{resample_cut}_1v', 
               'HDF5_particle_diagnostic' : True,
               'HDF5_field_diagnostic' : True,
              }

beam_specifics = {'injection_period' : 5,
                  'diag_period' : 500,
                  'radius' : 4.0e-3, #m
                  'nmp' : 1, #Num macroparticles to emit per emission step
                  'current_ramp_time': 20.0e-6,
                  'deltav_rms' : 0.0,
                 }

run_specifics = {'nx' : 64,
                 'ny' : 64,
                 'nz' : 32,
                 'xmax' : 1.0, #m
                 'ymax' : 1.0, #m
                 'zmax' : 0.5, #m 
                 'dt' : 0.5e-10, #s
                 'tmax' : 500.0e-6, #s
                }

deuterium_specifics = {
    'species_type' : 'deuterium',
    'particle_type' : 'deuterium',
    'n_beams' : n_beams,
    'mass' : 1874.61e6, #eV/c^2
    'v_x' : 1.696e6, #m/s
    'ke' : None, #eV
    'current' : 5.0e-4, #A
    'density' : None, #num/m^3
    'charge' : scipy.constants.e, #C
    'time_start' : 0.0, #s
    'time_duration' : run_specifics['tmax'], #s 
    'length' : None, #m
    # 'injection_radius' : 0.4, #m
    'injection_offset' : np.pi/2.0, #rad
    'injection_direction' : 1.0,
    'warpx_do_resampling' : True,
    'warpx_resampling_algorithm': 'leveling_thinning',
    'warpx_resampling_algorithm_target_ratio' : resample_cut,
    'warpx_resampling_trigger_max_avg_ppc' : resample_max_avg_ppc,
    'warpx_resampling_min_ppc' : resampling_min_ppc,
    # 'warpx_resampling_trigger_intervals' : resample_rate,
    
}


##########################
# physics components
##########################

magnetic_field = {'B0' : 0.1863, #0.245, #T 
                  'k' : 0.1, #0.1621
                  'A' : 0.0} #2.0}

def zero_canonical_angular_momentum(r):
    alpha  = magnetic_field['k'] / r**2
    mass   = 1.78266192e-36 * deuterium_specifics['mass']
    v_inj  = deuterium_specifics['v_x'] * deuterium_specifics['injection_direction']
    charge = deuterium_specifics['charge']
    B0     = magnetic_field['B0']
    return -mass * v_inj + 0.5 * charge * B0 * r * (1.0 - 0.5 * alpha * r**2)

magnetic_field['R'] = brentq(zero_canonical_angular_momentum, 0.01, 3.0)

print("Injection radius is "+str(magnetic_field['R'])+" meters.")

ALPHA = magnetic_field['k'] / (magnetic_field['R'] **2.0) #m^-2

# Now set injection radius equal to the migma R
deuterium_specifics['injection_radius'] = magnetic_field['R']

##########################
# numerics components
##########################

dn_array = np.array([(2.0* run_specifics['xmax'])/run_specifics['nx'],
                     (2.0* run_specifics['ymax'])/run_specifics['ny'],
                     (2.0* run_specifics['zmax'])/run_specifics['nz']])

grid = picmi.Cartesian3DGrid(
    number_of_cells=[run_specifics['nx'], 
                     run_specifics['ny'], 
                     run_specifics['nz']],
    lower_bound=[-run_specifics['xmax'], 
                 -run_specifics['ymax'], 
                 -run_specifics['zmax']],
    upper_bound=[run_specifics['xmax'], 
                 run_specifics['ymax'], 
                 run_specifics['zmax']],
    bc_xmin='neumann',
    bc_xmax='neumann',
    bc_ymin='neumann',
    bc_ymax='neumann',
    bc_zmin='neumann',
    bc_zmax='neumann',
    lower_boundary_conditions_particles=['absorbing', 'absorbing', 'absorbing'],
    upper_boundary_conditions_particles=['absorbing', 'absorbing', 'absorbing']
)

solver = picmi.ElectrostaticSolver(
    grid=grid,
    method='Multigrid',
    required_precision=1e-6,
    warpx_self_fields_verbosity = 0,
)

##########################
# define species
##########################

electrons = picmi.Species(
    particle_type='electron', name='electron',
    initial_distribution=None,
    warpx_do_not_deposit=not self_consistent_fields,
    warpx_self_fields_verbosity=0,
    warpx_do_resampling=True,
    warpx_resampling_algorithm="leveling_thinning",
    warpx_resampling_algorithm_target_ratio=resample_cut,
    warpx_resampling_trigger_max_avg_ppc=resample_max_avg_ppc,
    warpx_resampling_min_ppc=resampling_min_ppc,
    # warpx_resampling_trigger_intervals=2*resample_rate,
)

ar_mass = 6.6335209e-26 #kg
t_bkgd = 300.0
n_bkgd = 2.5e18 #1.25e18 # m^-3
arplus = picmi.Species(
    # particle_type='', 
    name='argon',
    charge=scipy.constants.e, mass=ar_mass,
    initial_distribution=None,
    warpx_do_not_deposit=not self_consistent_fields,
    warpx_self_fields_verbosity=0,
    warpx_do_resampling=True,
    warpx_resampling_algorithm="leveling_thinning",
    warpx_resampling_algorithm_target_ratio=resample_cut,
    warpx_resampling_trigger_max_avg_ppc=resample_max_avg_ppc,
    warpx_resampling_min_ppc=resampling_min_ppc,
    # warpx_resampling_trigger_intervals=2*resample_rate,
)

if interactions['fusion']:
    helium3 = SpeciesWithType(warpx_species_type='helium3',
                              mass=5.0082340395e-27,
                              charge = 0.0,
                              name='helium3', initial_distribution=None,
                              warpx_do_not_deposit=not self_consistent_fields,
                              warpx_self_fields_verbosity=0
                             )
    neutron = SpeciesWithType(warpx_species_type='neutron',
                              name='neutron', initial_distribution=None,
                              warpx_do_not_deposit=not self_consistent_fields,
                              warpx_self_fields_verbosity=0
                             )
    
beams = GenerateBeams(self_consistent_fields,
                      beam_specifics,
                      beam_species = {'deuterium' : deuterium_specifics},
                     )

##########################
# collisions
##########################

collisions = []
if interactions['fusion']:
    pB_fusion = NuclearFusion(
        name='pb_fusion',
        fusion_multiplier=fusion_multiplier,
        species=[beams['deuterium']['species'], beams['deuterium']['species'], ],
        product_species=[helium3, neutron],
        ndt=interactions['ndt']
    )
    collisions.append(pB_fusion)
    
if interactions['coulomb']:
    cc_species_list = [beams['deuterium']['species']]
    collisions = Coulomb_multi_init(collisions, cc_species_list)


# MCC collisions
# https://github.com/ECP-WarpX/warpx-data/tree/master/MCC_cross_sections
cross_sec_direc = '../../../../warpx-data/MCC_cross_sections/Ar/' 
#Change this to reflect warpx-data location (note, only has He, Ar, and Xe + N2 added by David)

# https://warpx.readthedocs.io/en/latest/usage/python.html#pywarpx.picmi.MCCCollisions:~:text=pywarpx.picmi.MCCCollisions

ionization_d2_process={
    'ionization' : {'cross_section' : cross_sec_direc+'ion_d2_ionization.dat',
                    'energy' : 15.759,
                    'species' : arplus} #the produced ion species from the background
}
d2_colls = picmi.MCCCollisions(
    name='d2_coll',
    species=beams['deuterium']['species'], # species colliding with background
    background_density=n_bkgd,
    background_temperature=t_bkgd,
    ndt=1,
    electron_species=electrons, # the produced electrons
    scattering_processes=ionization_d2_process
)
collisions.append(d2_colls)


#All ion interactions should be turned on for the background ions (N2+)
ar_ion_scattering_processes={
    'elastic' : {'cross_section' : cross_sec_direc+'ion_scattering.dat'},
    'charge_exchange' : {'cross_section' : cross_sec_direc+'charge_exchange.dat'},
}

ar_ion_colls = picmi.MCCCollisions(
    name='ar_coll_ion',
    species=arplus, # species colliding with background
    background_density=n_bkgd,
    background_temperature=t_bkgd,
    ndt=1,
    electron_species=electrons, # the produced electrons
    scattering_processes=ar_ion_scattering_processes
)
collisions.append(ar_ion_colls)


# All electron interactions should be turned on once implemented
electron_scattering_processes={
    'elastic' : {'cross_section' : cross_sec_direc+'electron_scattering.dat'},
    'ionization' : {'cross_section' : cross_sec_direc+'ionization.dat',
                    'energy' : 15.759,
                    'species' : arplus} #the produced ion species from the background
}

electron_colls = picmi.MCCCollisions(
    name='coll_elec',
    species=electrons,
    background_density=n_bkgd,
    background_temperature=t_bkgd,
    background_mass=ar_mass,
    ndt=1,
    electron_species=electrons,
    scattering_processes=electron_scattering_processes
)
collisions.append(electron_colls)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver=solver,
    max_steps=int(run_specifics['tmax']/run_specifics['dt']),
    verbose=0,
    time_step_size=run_specifics['dt'],
    warpx_collisions=collisions if collisions else None,
)

external_mag_field = rsfusion.magnetic_field.migma_cartesian_3d(sim, magnetic_field['B0'], ALPHA)

##########################
# diagnostics
##########################

diagdire = diagnostics['directory']

if diagnostics['HDF5_particle_diagnostic']:
    species_list = [beams['deuterium']['species'], arplus, electrons, helium3]
    part_diag = picmi.ParticleDiagnostic(write_dir = f'./diags/{diagdire}',
                                         warpx_file_prefix = 'particle',
                                         period=beam_specifics['diag_period'],
                                         species=species_list,
                                         warpx_openpmd_backend='h5',
                                         warpx_format='openpmd',
                                         data_list=['x', 'y', 'z', 'ux', 'uy', 'uz', 'weighting'])
    sim.add_diagnostic(part_diag)

if diagnostics['HDF5_field_diagnostic']:
    field_diag = picmi.FieldDiagnostic(write_dir = f'./diags/{diagdire}',
                                       warpx_file_prefix = 'field',
                                       grid = grid,
                                       period=beam_specifics['diag_period'],
                                       warpx_openpmd_backend='h5',
                                       warpx_format='openpmd',
                                       data_list=['B', 'E', 'J', 'rho'])
    sim.add_diagnostic(field_diag)
    
if interactions['fusion']:
    helium_diag = rsfusion.diagnostics.CountDiagnostic(
        sim, helium3, diagnostics['directory'], beam_specifics['diag_period'], install=True)

##########################
# particle initialization
##########################

if interactions['fusion']:
    sim.add_species(helium3, layout=picmi.GriddedLayout(n_macroparticle_per_cell=1))
    sim.add_species(neutron, layout=picmi.GriddedLayout(n_macroparticle_per_cell=1))

sim.add_species(beams['deuterium']['species'], layout=picmi.GriddedLayout(n_macroparticle_per_cell=1))

sim.add_species(electrons, layout = picmi.GriddedLayout(n_macroparticle_per_cell=1))
sim.add_species(arplus, layout = picmi.GriddedLayout(n_macroparticle_per_cell=1))
    
##########################
# particle injection
##########################

injectors = GenerateInjectors(beams=beams,
                              simulation=sim,
                              beam_specifics = beam_specifics,
                              beam_shape = 'circular',
                              dn_array = dn_array,
                              dt = run_specifics['dt'],
                              zfactor = 1.0,
                             )

for key in injectors:
    for beam_index in np.arange(beams[key]['n_beams']):
        callbacks.installbeforestep(injectors[key][f'injector_{beam_index}']._injection)

##########################
# simulation run
##########################

# Write input file that can be used to run with the compiled version
directory = diagnostics['directory']
sim.write_input_file(file_name=f'inputs_{directory}')

sim.step()
