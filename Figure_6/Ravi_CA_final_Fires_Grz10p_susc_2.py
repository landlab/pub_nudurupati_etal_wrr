
# -*- coding: utf-8 -*-
##############################################################################
## Authors: Sai Nudurupati and Erkan Istanbulluoglu
##############################################################################
# %%
import numpy as np
import pickle
import os
from landlab import RasterModelGrid as rmg
from funcs_resource_redistribution import (plot_res, plot_veg,
                                           compose_veg_grid,
                                           plot_normalized_res,
                                           plot_vegcover, plot_res_stats)
from landlab.components import ResourceRedistribution, SpatialDisturbance
import warnings
warnings.filterwarnings("ignore")

## Function that checks if a directory exists and creates it if it doesn't
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

current_dir = os.getcwd()
## Declare directory depending on the computer (Manually change things here)
sim_ = 'E:/Research_UW_Sai_PhD/2018_testing_ResourceRedistribution/' # Office desktop
#sim_ = 'C:/Users/saisiddu/Desktop/Buffer/Sims_16May16/'  	 # Thinkpad
#    sim_ = './outputs/' 						# Linux box
print_interval = 20   # Print after every # years.
## Name of the simulation
fires = 1  # 0 - No fires, 1 - fires
grazing = 1 # 0 - No grazing, 1 - Over Grazing
name_simulation = 'Fires_Grz10p_susc_2_pap'
sim = sim_+ name_simulation + '/'
print(name_simulation)
try:
    os.chdir(sim)
except:
    os.mkdir(sim)

## Creating folders to categorize and store plots
sim_R = sim + 'Elev/'
sim_V = sim + 'Veg/'
sim_Rnorm = sim + 'norm_Elev/'
ensure_dir(sim_R)
ensure_dir(sim_V)
ensure_dir(sim_Rnorm)

## Initialize grid
grid = rmg((1000, 1000), 2.5)    # Grid creation
# Create State Variable - R
R = np.ones(grid.number_of_cells)
#R = np.random.random(grid.number_of_cells) * 2 + \
#        (-2)*np.ones(grid.number_of_cells, dtype=float)
print('R_sum() - Initial = ', R.sum())
# Create State Variable - V
V = compose_veg_grid(grid, percent_bare=0.3,
                     percent_grass=0.3, percent_shrub=0.4)  # Organization is random

## Inputs
n_years = 300       # number of years to run model
plot_interval = 50  # interval at which plots are generated
# Fire Inputs
stoc_fire_extent_min = 0.089
stoc_fire_extent_max = 0.09
no_fires_per_year_min = 2
no_fires_per_year_max = 3

# Susceptibility
sh_susc = 0.5
gr_susc = 1.

# Grazing Inputs
if grazing:
    P1 = 0.1        # Grazing pressure P1 @ Over grazing
else:
    P1 = 0.001
e = 0.1             # Erosion rate
Rth_gr = 0.4        # Threshold for establishment of grass
Rth_sh = 0.8       # Threshold for establishment of shrubs
if fires:
    P_gr_rgrwth = 0.5   # Regrowth of grass after fire
else:
    P_gr_rgrwth = 0.25
P_sh_rgrwth = 0.25    # Regrowth of shrubs after fire
Pen = 0.05           # Shrub encroachment prob due to shrub neighbors P2 
if grazing:
    Pgrz = 0.01
else:
    Pgrz = 0.001    # Shrub encroachment due to grazing - seed dispersal P3
P_sh_fire_mor = 0.75     # Shrub mortality post fire
P_gr_fire_mor = 1.       # Grass mortality post fire
if fires:
    P_gr = 0.5 #0.5          # Grass establishment - seed dispersal P5
else:
    P_gr = 0.25
R_high_threshold = 2.    # Resource upper threshold for re-adjustment
R_low_threshold = -2.    # Lower threshold for resource
R_dep_threshold = 2.     # Resource re-adjustment eligibility threshold
sh_max_age = 600         # Shrub max age in years
sh_seedling_max_age = 18 # Max age of shrub seedling
sh_seedling_mor_dis = 0.0
     # Probability for shrub seedling mortality due to disease
sh_mor_dis_low_thresh_age = int(1. * sh_max_age)
     # Threhold age of shrubs until when disease strikes at sh_mor_dis_low_slp
sh_mor_dis_low_slp = 0.01 
     # Constant probability for Shrub mortality due to disease 
     # below sh_mor_dis_low_thresh
     # Probability of shrub mortality due to disease = (multiplier * age)
sh_mor_dis_high_slp = 1. - sh_mor_dis_low_slp
     # Multiplier (slope) for shrub mortality due to disease
     # above sh_mor_dis_low_thresh
sh_mor_ws_thresh = 0.01       # Shrub mortality water stress threshold
gr_mor_ws_thresh = 0.08       # Grass mortality water stress threshold

## Declare necessary parameters
V_age = np.zeros(grid.number_of_cells, dtype=int)
## Plot initial vegetation
In_name = name_simulation + '_00000_'
plot_veg(grid, V, sim_V, name=In_name)
plot_res(grid, R, [R_low_threshold, R_high_threshold], sim_R, name=In_name)
## Initializing Empty/buffer variables for probing
VegType = np.empty( [n_years,grid.number_of_cells], dtype=int )    # Holder for veg
Res_mean = np.empty( n_years, dtype=float )
Res_std = np.empty( n_years, dtype=float )
fr_burnt_locs = np.zeros(n_years)  # To record number of burnt cells every year
n_ignition_cells = np.zeros(n_years)  # To record how many fires every year

## Record all inputs to data so that it can be pickled
data = {'bare_': 0, 'grass': 1, 'shrub': 2, 'n_years': n_years,
        'R_initial':R, 'V_initial':V, 'GrazingPressure__P1':P1,
        'e':e, 'Rth_gr':Rth_gr,
        'Rth_sh':Rth_sh, 'P_gr_regrwth':P_gr_rgrwth,
        'P_sh_regrwth': P_sh_rgrwth,
        'Pen':Pen, 'Pgrz':Pgrz, 'P_gr': P_gr,
        'P_sh_fire_mor':P_sh_fire_mor,
        'P_gr_fire_mor':P_gr_fire_mor,
        'R_threshold':R_high_threshold,
        'R_low_threshold':R_low_threshold,
        'R_dep_threshold': R_dep_threshold,
        'sh_max_age':sh_max_age, 'sh_seedling_max_age': sh_seedling_max_age,
        'sh_seedling_mor_dis': sh_seedling_mor_dis,
        'sh_mor_dis_low_thresh_age': sh_mor_dis_low_thresh_age,
        'sh_mor_dis_low_slp': sh_mor_dis_low_slp,
        'sh_mor_dis_high_slp': sh_mor_dis_high_slp,
        'sh_mor_ws_thresh': sh_mor_ws_thresh,
        'gr_mor_ws_thresh': gr_mor_ws_thresh,
        'sh_susc': sh_susc,
        'gr_susc': gr_susc,
        }

grid.at_cell['vegetation__plant_functional_type'] = V
grid.at_cell['soil__resources'] = R
sd = SpatialDisturbance(grid, pft_scheme='ravi_et_al_2009')
rr = ResourceRedistribution(grid, **data)
V_age = rr.initialize_Veg_age(V_age=V_age)
#susc_rec = np.empty([n_years, grid.number_of_cells])
# %%
## Create Time loop
for time in range(0, n_years):
# Change introduction 1:
#    if time == time_1:
#        per_burn_per_fire = per_burn_per_fire_1
#        grazing = grazing_1
#        if fires_1:
#            rr.initialize()
# Normal Process    
    sim_name = name_simulation + '_' + "%05d"%(time+1) + '_'
#    fires = np.random.randint(0, high=2)
    if fires:
        number_of_fires_per_year = np.random.randint(
                                    no_fires_per_year_min,
                                    high=no_fires_per_year_max+1)
        stoc_fire_extent = (stoc_fire_extent_min +
                            np.random.rand() *
                            (stoc_fire_extent_max -
                             stoc_fire_extent_min))
        per_burn_per_fire = stoc_fire_extent          # Mean Percentage area burnt per fire
        per_dev_fire = (0.3 * per_burn_per_fire)      # Deviation in percentage area burnt from mean

#        susc_rec[time, :] = susc
        (V, burnt_locs,
         ignition_cells) = sd.initiate_fires(
                                V=V, n_fires=number_of_fires_per_year,
                                fire_area_mean=per_burn_per_fire,
                                fire_area_dev=per_dev_fire,
                                sh_susc=sh_susc, gr_susc=gr_susc)
        fr_burnt_locs[time] = len(burnt_locs)/float(grid.number_of_cells)
        n_ignition_cells[time] = len(ignition_cells)
    V, grazed_cells = sd.graze(V=V, grazing_pressure=P1)

    (eroded_soil, eroded_soil_shrub, burnt_shrubs,
     burnt_grass, bare_cells) = rr.erode()
    (burnt_shrubs_neigh, exclusive,
     shrub_exclusive, grass_exclusive,
     bare_exclusive, eroded_soil_part) = rr.deposit(eroded_soil,
                                                    eroded_soil_shrub)
    (resource_adjusted,
     eligible_locs_to_adj_neigh,
     Elig_locs, sed_to_borrow) = rr.re_adjust_resource()
    R = grid.at_cell['soil__resources']
    if time % print_interval == 0:
        print 'time = ', time, ' years'
        print('yearly R.sum = ', R.sum())
    V_age = rr.update_Veg_age(V_age=V_age)
    (V_age, est_1, est_2,
     est_3, est_4, est_5) = rr.establish(V_age)
    (V_age, Pmor_age, Pmor_age_ws) = rr.mortality(V_age=V_age)

## Record vegetation
    VegType[time] = grid.at_cell['vegetation__plant_functional_type']
    Res_mean[time] = np.mean(R)
    Res_std[time] = np.std(R)
    if time%plot_interval==0:
        np.save(sim_V+'vegtype_' + str(time) + '.npy', V)
        np.save(sim_R+'resource_' + str(time) + '.npy', R)
        plot_veg(grid, V, sim_V, name=sim_name)
        plot_res(grid, R, [R_low_threshold, R_high_threshold], \
                        sim_R, name=sim_name)
        Rnorm=plot_normalized_res(grid, R, sim_Rnorm, name=sim_name)
  
bare_cov, grass_cov, shrub_cov = plot_vegcover(VegType, n_years, sim,
                                               ylims=[0, 100])
plot_veg(grid, V, sim_V, name=sim_name)
plot_res(grid, R, [R_low_threshold, R_high_threshold], sim_R, name=sim_name)
Rnorm=plot_normalized_res(grid, R, sim_Rnorm, name=sim_name)
plot_res_stats(Res_mean, Res_std, n_years, sim)


## Save inputs - Dump pickles
output = open(sim+name_simulation+'_data.pkl', 'wb')
pickle.dump(data, output)
output.close()

## Save outputs
veg_cov_output = np.empty([3,n_years])
veg_cov_output[0][:] = bare_cov
veg_cov_output[1][:] = grass_cov
veg_cov_output[2][:] = shrub_cov
np.save(sim+name_simulation+'veg_cov.npy', veg_cov_output)
np.save(sim+name_simulation+'veg_spread.npy', VegType)
np.save(sim+'fr_burnt_locs.npy', fr_burnt_locs)
np.save(sim+'n_ignition_cells.npy', n_ignition_cells)
