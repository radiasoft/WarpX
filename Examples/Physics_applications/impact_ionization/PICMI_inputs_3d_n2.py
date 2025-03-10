#!/usr/bin/env python3
#
# --- Input file for MCC testing

import os
os.environ["OMP_NUM_THREADS"] = "4"

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

self_consistent_fields = False
n_beams = 4 
fusion_multiplier = 1e20

interactions = {'coulomb' : True, 
                'ndt' : 1, #period to call rxns
                'fusion' : False, 
               }

diagnostics = {'directory' : 'june_impact_fusion_0v', 
               'HDF5_particle_diagnostic' : True,
               'HDF5_field_diagnostic' : False,
              }

beam_specifics = {'injection_period' : 5,
                  'diag_period' : 500,
                  'radius' : 4.0e-3, #m
                  'nmp' : 1, #Num macroparticles to emit per emission step
                  'current_ramp_time': 0.0,
                  'deltav_rms' : 0.0,
                 }

run_specifics = {'nx' : 64,
                 'ny' : 64,
                 'nz' : 32,
                 'xmax' : 1.0, #m
                 'ymax' : 1.0, #m
                 'zmax' : 0.5, #m 
                 'dt' : 1.0e-10, #s
                 'tmax' : 40.0e-6, #s
                }

proton_specifics = {'species_type' : 'hydrogen1',
                    'n_beams' : n_beams,
                    'mass' : 938.27208816e6, #eV/c^2
                    'v_x' : 1.026e7, #m/s
                    'ke' : None, #eV
                    'current' : 5.0e-4, #A
                    'density' : None, #num/m^3
                    'charge' : scipy.constants.e, #C
                    'time_start' : 0.0, #16540*run_specifics['dt'], #s
                    'time_duration' : run_specifics['tmax'], #s 3.1e-6/11.0, #
                    'length' : None, #m
                    # 'injection_radius' : 0.4, #m
                    'injection_offset' : np.pi, #rad
                    'injection_direction' : 0.5,
                   }

boron11_specifics = {'species_type' : 'boron11',
                     'n_beams' : n_beams,
                     'mass' : 1.0255103e10, #eV/c^2
                     'v_x' : 9.389e5, #m/s
                     'ke' : None, #eV
                     'current' : 5.0e-4/11.0, #A
                     'density' : None, #num/m^3
                     'charge' : scipy.constants.e, #C
                     'time_start' : 0.0, #s
                     'time_duration' : run_specifics['tmax'], #s 3.1e-6, #
                     'length' : None, #m
                     # 'injection_radius' : 0.4, #m
                     'injection_offset' : np.pi/2.0, #rad
                     'injection_direction' : 0.5, #-
                    }


##########################
# physics components
##########################

magnetic_field = {'B0' : 0.1863, #0.245, #T 
                  'k' : 0.1, #0.1621
                  'A' : 0.0} #2.0}

def zero_canonical_angular_momentum(r):
    alpha  = magnetic_field['k'] / r**2
    mass   = 1.78266192e-36 * proton_specifics['mass']
    v_inj  = proton_specifics['v_x'] * proton_specifics['injection_direction']
    charge = proton_specifics['charge']
    B0     = magnetic_field['B0']
    return -mass * v_inj + 0.5 * charge * B0 * r * (1.0 - 0.5 * alpha * r**2)

magnetic_field['R'] = brentq(zero_canonical_angular_momentum, 0.01, 3.0)

print("Injection radius is "+str(magnetic_field['R'])+" meters.")

ALPHA = magnetic_field['k'] / (magnetic_field['R'] **2.0) #m^-2

# Now set injection radius equal to the migma R
# deuterium_specifics['injection_radius'] = magnetic_field['R']
boron11_specifics['injection_radius'] = magnetic_field['R']
proton_specifics['injection_radius'] = magnetic_field['R']

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
    required_precision=1e-1,
    warpx_self_fields_verbosity = 0,
)

##########################
# define species
##########################

electrons = picmi.Species(
    particle_type='electron', name='electrons',
    initial_distribution=None,
    warpx_do_not_deposit=not self_consistent_fields,
    warpx_self_fields_verbosity=0
)

n2_mass = 4.6517341e-26
t_bkgd = 300.0
n_bkgd = 1.25e18 # m^-3
n2plus = picmi.Species(
    # particle_type='', 
    name='n2_ions',
    charge=scipy.constants.e, mass=n2_mass,
    initial_distribution=None,
    warpx_do_not_deposit=not self_consistent_fields,
    warpx_self_fields_verbosity=0
)

if interactions['fusion']:
    helium4 = SpeciesWithType(warpx_species_type='helium4',
                              mass=6.64647792508608e-27,
                              name='helium4', initial_distribution=None,
                              warpx_do_not_deposit=not self_consistent_fields,
                              warpx_self_fields_verbosity=0
                             )

beams = GenerateBeams(self_consistent_fields,
                      beam_specifics,
                      beam_species = {'hydrogen1' : proton_specifics, 
                                      'boron11' : boron11_specifics},
                      # beam_species = {'deuterium' : deuterium_specifics},
                     )

##########################
# collisions
##########################

collisions = []
if interactions['fusion']:
    pB_fusion = NuclearFusion(
        name='pb_fusion',
        fusion_multiplier=fusion_multiplier,
        species=[beams['hydrogen1']['species'],beams['boron11']['species']],
        #[beams['deuterium']['species'], beams['deuterium']['species'], ],
        product_species=[helium4, ], #[helium3, neutron],
        ndt=interactions['ndt']
    )
    collisions.append(pB_fusion)
    
if interactions['coulomb']:
    cc_species_list = [beams['hydrogen1']['species'],beams['boron11']['species']]
    collisions = Coulomb_multi_init(collisions, cc_species_list)


# MCC collisions
# https://github.com/ECP-WarpX/warpx-data/tree/master/MCC_cross_sections
cross_sec_direc = '../../../warpx-data/MCC_cross_sections/N2/' #Change this to reflect warpx-data location (note, only has He, Ar, and Xe + N2 added by David)

# https://warpx.readthedocs.io/en/latest/usage/python.html#pywarpx.picmi.MCCCollisions:~:text=pywarpx.picmi.MCCCollisions

ionization_p_process={
    'ionization' : {'cross_section' : cross_sec_direc+'ion_p_ionization.dat',
                    'energy' : 15.578,
                    'species' : n2plus} #the produced ion species from the background
}
p_colls = picmi.MCCCollisions(
    name='p_coll',
    species=beams['hydrogen1']['species'], # species colliding with background
    background_density=n_bkgd,
    background_temperature=t_bkgd,
    ndt=1,
    electron_species=electrons, # the produced electrons
    scattering_processes=ionization_p_process
)
collisions.append(p_colls)

ionization_b11_process={
    'ionization' : {'cross_section' : cross_sec_direc+'ion_b11_ionization.dat',
                    'energy' : 15.578,
                    'species' : n2plus} #the produced ion species from the background
}
b11_colls = picmi.MCCCollisions(
    name='b11_coll',
    species=beams['boron11']['species'], # species colliding with background
    background_density=n_bkgd,
    background_temperature=t_bkgd,
    ndt=1,
    electron_species=electrons, # the produced electrons
    scattering_processes=ionization_b11_process
)
collisions.append(b11_colls)


#All ion interactions should be turned on for the background ions (N2+)
n2_ion_scattering_processes={
    'elastic' : {'cross_section' : cross_sec_direc+'ion_scattering.dat'},
    'charge_exchange' : {'cross_section' : cross_sec_direc+'charge_exchange.dat'},
}

n2_ion_colls = picmi.MCCCollisions(
    name='n2_coll_ion',
    species=n2plus, # species colliding with background
    background_density=n_bkgd,
    background_temperature=t_bkgd,
    ndt=1,
    electron_species=electrons, # the produced electrons
    scattering_processes=n2_ion_scattering_processes
)
collisions.append(n2_ion_colls)


# All electron interactions should be turned on once implemented
electron_scattering_processes={
    'elastic' : {'cross_section' : cross_sec_direc+'electron_scattering.dat'},
    'ionization' : {'cross_section' : cross_sec_direc+'ionization.dat',
                    'energy' : 15.578,
                    'species' : n2plus} #the produced ion species from the background
}

electron_colls = picmi.MCCCollisions(
    name='coll_elec',
    species=electrons,
    background_density=n_bkgd,
    background_temperature=t_bkgd,
    background_mass=n2_mass,
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
    species_list = [beams['hydrogen1']['species'],beams['boron11']['species'], n2plus, electrons] #beams['deuterium']['species']
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
        sim, helium4, diagnostics['directory'], beam_specifics['diag_period'], install=True)

##########################
# particle initialization
##########################

if interactions['fusion']:
    sim.add_species(helium4, layout=picmi.GriddedLayout(n_macroparticle_per_cell=1))

# sim.add_species(beams['deuterium']['species'], layout=picmi.GriddedLayout(n_macroparticle_per_cell=1))
sim.add_species(beams['hydrogen1']['species'], layout=picmi.GriddedLayout(n_macroparticle_per_cell=1))
sim.add_species(beams['boron11']['species'], layout=picmi.GriddedLayout(n_macroparticle_per_cell=1))

sim.add_species(electrons, layout = picmi.GriddedLayout(n_macroparticle_per_cell=1))
sim.add_species(n2plus, layout = picmi.GriddedLayout(n_macroparticle_per_cell=1))
    
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
