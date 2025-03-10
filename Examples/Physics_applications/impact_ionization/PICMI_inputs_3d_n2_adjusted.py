#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "4"
from pywarpx import picmi, callbacks
import pywarpx
import numpy as np
import scipy.constants
constants = picmi.constants


class SpeciesWithType(picmi.Species):
    """
    Extends the picmi.Species class to set the `species_type` attribute.
    This attribute (and not any of the other picmi species attributes)
    is checked by the WarpX NuclearFusion interaction to verify the species for the interaction are
    of the expected type.
    """

    def init(self, kw):
        """See picmi.Species for full information.

            Adds:
                warpx_species_type (str): `species_type` allowed by WarpX.
        """
        super().init(kw)
        self.species_type = kw.pop('warpx_species_type', None)

    def initialize_inputs(self, layout,
                          initialize_self_fields = False,
                          injection_plane_position = None,
                          injection_plane_normal_vector = None):
        if self.species_type:
            self.species.add_new_attr("species_type", self.species_type)
        super().initialize_inputs(layout,
                          initialize_self_fields = False,
                          injection_plane_position = None,
                          injection_plane_normal_vector = None)


##########################
# Parameters to Set
##########################

self_consistent_fields = True
diagnostics = {'directory' : 'test_dual_ion_impact', 
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
                    'mass' : 938.27208816e6, #eV/c^2
                    'charge' : scipy.constants.e, #C
                   }

boron11_specifics = {'species_type' : 'boron11',
                     'mass' : 1.0255103e10, #eV/c^2
                     'charge' : scipy.constants.e, #C
                    }

##########################
# numerics components
##########################

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

n2_mass = 4.6517341e-26
t_bkgd = 300.0
n_bkgd = 1.25e18 # m^-3

n2plus = picmi.Species(
    name='n2_ions',
    charge=scipy.constants.e, mass=n2_mass,
    initial_distribution=None,
    warpx_do_not_deposit=not self_consistent_fields,
    warpx_self_fields_verbosity=0
)

electrons = picmi.Species(
    particle_type='electron', name='electrons',
    initial_distribution=None,
    warpx_do_not_deposit=not self_consistent_fields,
    warpx_self_fields_verbosity=0
)

hydrogen = SpeciesWithType(
    warpx_species_type=proton_specifics['species_type'],
    mass=proton_specifics['mass'] * picmi.constants.q_e / picmi.constants.c ** 2,
    charge=proton_specifics['charge'],
    name='hydrogen1', 
    initial_distribution=None,
    warpx_do_not_deposit=not self_consistent_fields,
    warpx_self_fields_verbosity=0
) 

boron = SpeciesWithType(
    warpx_species_type=boron11_specifics['species_type'],
    mass=boron11_specifics['mass'] * picmi.constants.q_e / picmi.constants.c ** 2,
    charge=boron11_specifics['charge'],
    name='boron11', 
    initial_distribution=None,
    warpx_do_not_deposit=not self_consistent_fields,
    warpx_self_fields_verbosity=0
)

##########################
# collisions
##########################

collisions = []

# MCC collisions: https://github.com/ECP-WarpX/warpx-data/tree/master/MCC_cross_sections
# https://warpx.readthedocs.io/en/latest/usage/python.html#pywarpx.picmi.MCCCollisions:~:text=pywarpx.picmi.MCCCollisions
cross_sec_direc = '../../../warpx-data/MCC_cross_sections/N2/' 


ionization_p_process={
    'ionization' : {'cross_section' : cross_sec_direc+'ion_p_ionization.dat',
                    'energy' : 15.578,
                    'species' : n2plus} #the produced ion species from the background
}
p_colls = picmi.MCCCollisions(
    name='p_coll',
    species=hydrogen, 
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
    species=boron, 
    background_density=n_bkgd,
    background_temperature=t_bkgd,
    ndt=1,
    electron_species=electrons, # the produced electrons
    scattering_processes=ionization_b11_process
)
collisions.append(b11_colls)


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

#########################
# particle initialization
##########################

sim.add_species(hydrogen, layout=picmi.GriddedLayout(n_macroparticle_per_cell=1))
sim.add_species(boron, layout=picmi.GriddedLayout(n_macroparticle_per_cell=1))

sim.add_species(electrons, layout = picmi.GriddedLayout(n_macroparticle_per_cell=1))
sim.add_species(n2plus, layout = picmi.GriddedLayout(n_macroparticle_per_cell=1))

##########################
# simulation run
##########################

# Write input file that can be used to run with the compiled version
directory = diagnostics['directory']
sim.write_input_file(file_name=f'inputs_{directory}')

sim.step()
