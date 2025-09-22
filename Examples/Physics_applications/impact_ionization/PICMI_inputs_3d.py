#!/usr/bin/env python3
#
# --- Input file for MCC testing

import os
os.environ["OMP_NUM_THREADS"] = "4"
# sys.path.append("path_to_rsfusion")

from pywarpx import picmi, particle_containers, callbacks

import numpy as np

import scipy.constants
constants = picmi.constants

from rsbeams.rsstats import kinematic


##########################
# Parameters to Set
##########################

self_consistent_fields = False
n_beams = 1 #only implemented for a single beam

interactions = {'ndt' : 1, #period to call rxns
               }

diagnostics = {'directory' : 'ion_impact_0v', 
               'HDF5_particle_diagnostic' : True,
               'HDF5_field_diagnostic' : False,
              }

beam_specifics = {'injection_period' : 5,
                  'diag_period' : 500,
                  'radius' : 4.0e-3, #m
                  'nmp' : 1, #Num macroparticles to emit per emission step
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

deuterium_specifics = {'species_type' : 'deuterium',
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
                    'injection_radius' : 0.4, #m
                    'injection_offset' : np.pi, #rad
                    'injection_direction' : 0.5,
                   }

##########################
# physics parameters
##########################

N_INERT = 9.64e20 # m^-3
T_INERT = 300.0 # K
M_INERT = 4.65e-26 # kg

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

deuterium = picmi.Species(
    mass=deuterium_specifics['mass'] * constants.q_e / constants.c ** 2,
    charge=deuterium_specifics['charge'],
    name='deuterium', 
    initial_distribution=None,
    warpx_do_not_deposit=not self_consistent_fields,
    warpx_self_fields_verbosity=0
)

deuterium_specifics['nmp'] = beam_specifics['nmp']
deuterium_specifics['cross_sec_area'] = np.pi * beam_specifics['radius']**2.0
if deuterium_specifics['ke'] == None:
    deuterium_specifics['ke'] = kinematic.Converter(velocity=deuterium_specifics['v_x'], mass=deuterium_specifics['mass'])(silent=True)["kenergy"]
if deuterium_specifics['v_x'] == None:
    deuterium_specifics['v_x'] = kinematic.Converter(kenergy=deuterium_specifics['ke'], mass=deuterium_specifics['mass'])(silent=True)['velocity']
if deuterium_specifics['time_duration'] == None:
    deuterium_specifics['time_duration'] = deuterium_specifics['length']/deuterium_specifics['v_x']
if deuterium_specifics['density'] == None:
    deuterium_specifics['density'] = deuterium_specifics['current'] / (np.abs(deuterium_specifics['charge']) * deuterium_specifics['cross_sec_area'] * deuterium_specifics['v_x'])

##########################
# collisions
##########################

collisions = []

# # MCC collisions
# # https://github.com/ECP-WarpX/warpx-data/tree/master/MCC_cross_sections
# cross_sec_direc = '../../../warpx-data/MCC_cross_sections/He/' #Change this to reflect warpx-data location (note, only has He, Ar, and Xe)

# # https://warpx.readthedocs.io/en/latest/usage/python.html#pywarpx.picmi.MCCCollisions:~:text=pywarpx.picmi.MCCCollisions
# mcc_ions = picmi.MCCCollisions(
#     name='coll_ion',
#     species=deuterium,
#     background_density=N_INERT,
#     background_temperature=T_INERT,
#     background_mass=M_INERT,
#     scattering_processes={
#         'ionization' : {
#             'cross_section' : cross_sec_direc+'ionization.dat',
#             'species' : deuterium
#         },
#     }
#     electron_species=electrons
# )

# collisions.append(mcc_ions)

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
    
##########################
# diagnostics
##########################

diagdire = diagnostics['directory']

if diagnostics['HDF5_particle_diagnostic']:
    species_list = [deuterium]
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
    
##########################
# particle initialization
##########################
    
sim.add_species(deuterium, layout=picmi.GriddedLayout(n_macroparticle_per_cell=1))
sim.add_species(electrons, layout = picmi.GriddedLayout(n_macroparticle_per_cell=1))

##########################
# particle injection
##########################

def rotation(a: list[float, float, float], b: list[float, float, float]) -> np.ndarray:
    # Rotate unit vector a to unit vector b
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    v_mat = np.array([[    0, -v[2], v[1]],
                      [ v[2],    0, -v[0]],
                      [-v[1],  v[0],   0]])
    R = np.identity(3) + v_mat + np.dot(v_mat, v_mat) * (1 - c) / s**2

    return R

def deuterium_injection(sim, position, min_step, max_step, period, size, 
                        weight, species, macroparticles, unit_vector, kinetic_energy):
    
    if (sim.extension.warpx.getistep(0) < min_step) or ((sim.extension.warpx.getistep(0) > max_step) and (max_step > 0)):
        return
    if sim.extension.warpx.getistep(0) % period == 0:
        
        species_kinematic = kinematic.Converter(kenergy=kinetic_energy, mass=species.mass, mass_unit='SI')(silent=True)
        w = species_kinematic['velocity'] * sim.time_step_size * period
        
        theta = np.random.uniform(0.0, 2.0*np.pi, size=[macroparticles*period]) 
        radius_1 = np.random.uniform(0.0,size[0]/2.0, size=[macroparticles*period])
        radius_2 = np.random.uniform(0.0,size[1]/2.0, size=[macroparticles*period])
        z = np.random.uniform(w/-2.0,w/2.0, size=[macroparticles*period])
        x = radius_1 * np.cos(theta)
        y = radius_2 * np.sin(theta)
        coordinates = np.vstack((x,y,z)).T.reshape([macroparticles*period, 3, 1])
        coordinates = np.matmul(rotation([0, 0, 1], unit_vector), coordinates).squeeze()
        coordinates += position
        x, y, z = coordinates.T
        
        momenta = (species_kinematic['betagamma'] * constants.c) * np.ones_like(coordinates) * unit_vector
        ux, uy, uz = momenta.T
         
        species_wrapper = particle_containers.ParticleContainerWrapper(species.name)
        # warpx requires that arrays it receives be contiguous
        species_wrapper.add_particles(
            x=np.ascontiguousarray(x),   y=np.ascontiguousarray(y),   z=np.ascontiguousarray(z),
            ux=np.ascontiguousarray(ux), uy=np.ascontiguousarray(uy), uz=np.ascontiguousarray(uz),
            w=np.ascontiguousarray(np.ones_like(ux))*weight, unique_particles=False
        )

theta_slice = (2.0*np.pi) / deuterium_specifics['n_beams']
theta = (theta_slice) + deuterium_specifics['injection_offset']
x_pos = deuterium_specifics['injection_radius']*np.cos(theta)
y_pos = deuterium_specifics['injection_radius']*np.sin(theta)

if deuterium_specifics['injection_direction'].is_integer():
    normal_vector = [y_pos * deuterium_specifics['injection_direction'],
                     -x_pos * deuterium_specifics['injection_direction'],0]
else:
    normal_vector = [-x_pos * np.sign(deuterium_specifics['injection_direction']),
                     -y_pos * np.sign(deuterium_specifics['injection_direction']),0]
    
min_step = np.ceil(deuterium_specifics['time_start'] / run_specifics['dt'])

callbacks.installbeforestep(
    deuterium_injection(
        sim= sim, 
        position= [x_pos, y_pos, 0.0],
        min_step= min_step, 
        max_step= np.ceil(deuterium_specifics['time_duration'] / run_specifics['dt']) +min_step, 
        period= beam_specifics['injection_period'], 
        size= [(2.0*beam_specifics['radius']), (2.0*beam_specifics['radius'])],
        weight= (deuterium_specifics['density'] * deuterium_specifics['v_x'] * run_specifics['dt'] * deuterium_specifics['cross_sec_area']) / deuterium_specifics['nmp'],
        species= deuterium, 
        macroparticles= beam_specifics['nmp'], 
        unit_vector= np.array(normal_vector) / np.sqrt(np.sum(np.array(normal_vector)**2)),
        kinetic_energy= deuterium_specifics['ke']
    )
)

##########################
# simulation run
##########################

# Write input file that can be used to run with the compiled version
directory = diagnostics['directory']
sim.write_input_file(file_name=f'inputs_{directory}')

sim.step()
