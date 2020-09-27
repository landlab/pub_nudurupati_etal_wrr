# Jai Sri Sainath!
"""
Created on 15 May 2017 - for CSDMS 2017
updated on 22 May 2017 - for CSDMS 2017
updated on 24Feb2019 - from ca_flat_change_addWoodyPlants.py
This tutorial is similar to:
landlab/tutorials/ecohydrology/cellular_automaton_vegetation_flat_surface.ipynb

@author: Sai Nudurupati & Erkan Istanbulluoglu
"""
from __future__ import print_function
import os
import sys
import time
import numpy as np
from landlab import RasterModelGrid, load_params
from ecohyd_functions_flat_coupled import (initialize, empty_arrays,
                                           create_pet_lookup, calc_veg_cov,
                                           save_np_files,
                                           add_trees, add_shrubs,
                                           get_edge_cell_ids,
                                           set_pft_at_cells,
                                           plot_marked_cells,
                                           plot_veg_type,
                                           get_fr_seeds)
#from fires_csdms_2017 import create_fires
from landlab.components import SpatialDisturbance
import pickle
import warnings
warnings.filterwarnings("ignore")

# Create dictionary that holds the inputs
#data = load_params('clust_59.yaml')
#output_sub_fldr = 'clust_59'  # Output subfolder name
data = load_params(sys.argv[1]+'.yaml')
output_sub_fldr = sys.argv[1]  # Output subfolder name
print(output_sub_fldr)

# Fires Inputs
fires = data['fires']    # If 1 -> It will execute creat_fires() algorithm
grazing = data['grazing'] # If 1 -> It will execute grazing

n_years = data['n_years']  # Approx number of years for model to run
yr_step = data['yr_step']  # Year step to plot data

# Create RasterModelGrids
nrows = data['nrows']
ncols = data['ncols']
dx = data['dx']
dy = data['dy']
grid1 = RasterModelGrid((nrows, ncols), spacing=(dx, dy))
grid = RasterModelGrid((5, 4), spacing=(dx, dy))

(precip_dry, precip_wet, radiation, pet_tree, pet_shrub,
 pet_grass, soil_moisture, vegetation, vegca) = initialize(data, grid, grid1)

# If initializing with vegtype, read from the file in .npy format
if data['pre_evolved_veg']:
    input_veg = np.load(data['input_veg_filename'])
    if np.shape(input_veg)[0] == grid1.number_of_cells:
        grid1.at_cell['vegetation__plant_functional_type'] = input_veg
    else:
        print('Shape of input vegtype field does not match grid1!')

sd = SpatialDisturbance(grid1, pft_scheme='zhou_et_al_2013')

# Calculate approximate number of storms per year
fraction_wet = (data['doy__end_of_monsoon'] -
                data['doy__start_of_monsoon']) / 365.
fraction_dry = 1 - fraction_wet
no_of_storms_wet = 8760 * fraction_wet / (data['mean_interstorm_wet'] +
                                          data['mean_storm_wet'])
no_of_storms_dry = 8760 * fraction_dry / (data['mean_interstorm_dry'] +
                                          data['mean_storm_dry'])
n = int(n_years * (no_of_storms_wet + no_of_storms_dry))

(precip, inter_storm_dt, storm_dt, daily_pet,
 rad_factor, EP30, pet_threshold) = empty_arrays(n, grid, grid1)

veg_type = np.empty([grid.number_of_cells], dtype=int)
veg_cov = np.zeros([n_years+25, 6])
n_fires_per_year = np.empty(data['n_years'], dtype=int)
per_fire_area = np.empty(data['n_years'], dtype=float)
per_yr_burnt_area_fr = np.zeros(data['n_years'], dtype=float)
per_yr_grazed_area_fr = np.zeros(data['n_years'], dtype=float)

(EP30, daily_pet) = create_pet_lookup(grid, radiation, pet_tree, pet_shrub,
                                      pet_grass, daily_pet, rad_factor, EP30)

precip = np.load('precip_flt_31_lnger.npy')
storm_dt = np.load('storm_dt_flt_31_lnger.npy')
inter_storm_dt = np.load('interstorm_dt_flt_31_lnger.npy')

# Setting Edges - 24Feb19
if data['add_left_edge']:
     lft_cells = get_edge_cell_ids(grid1, edge_width=data['l_edg_wdth'])
     set_pft_at_cells(lft_cells, grid1, pft=data['left_pft'])
     plot_marked_cells(lft_cells, grid1)
if data['add_right_edge']:
     rght_cells = get_edge_cell_ids(grid1, edge_side='right',
                                    edge_width=data['r_edg_wdth'])
     set_pft_at_cells(rght_cells, grid1, pft=data['right_pft'])
     plot_marked_cells(rght_cells, grid1)
if data['add_bottom_edge']:
     b_cells = get_edge_cell_ids(grid1, edge_side='bottom',
                                 edge_width=data['b_edg_wdth'])
     set_pft_at_cells(b_cells, grid1, pft=data['bottom_pft'])
     plot_marked_cells(b_cells, grid1)
if data['add_top_edge']:
     t_cells = get_edge_cell_ids(grid1, edge_side='top',
                                 edge_width=data['t_edg_wdth'])
     set_pft_at_cells(t_cells, grid1, pft=data['top_pft'])
     plot_marked_cells(t_cells, grid1)

# Represent current time in years
current_time = 0  # Start from first day of Jan

# Keep track of run time for simulation - optional
wallclock_start = time.clock()  # Recording time taken for simulation

# declaring few variables that will be used in the storm loop
time_check = 0.  # Buffer to store current_time at previous storm
yrs = 0  # Keep track of number of years passed
water_stress = 0.  # Buffer for Water Stress
Tg = 0  # Growing season in days

# Saving
try:
    os.chdir('E:/Research_UW_Sai_PhD/2019_CATGraSS_SpDist_clusters')
except OSError:
    os.chdir('/data1/sai_projs/2019_CATGraSS_SpDist_clusters')
finally:
    pass

try:
    os.mkdir('output')
except OSError:
    pass
finally:
    os.chdir('output')

sub_fldr = output_sub_fldr
print(sub_fldr)

try:
    os.mkdir(sub_fldr)
except OSError:
    pass
finally:
    os.chdir(sub_fldr)

# Run storm Loop
i = -1
while yrs < n_years:
    i += 1
    # Update objects

    # Calculate Day of Year (DOY)
    julian = np.int(np.floor((current_time - np.floor(current_time)) * 365.))

    # Making sure that minimum storm_dt is 10 min (0.007 hrs) - added: 19Apr18_12:19pm
    if storm_dt[i] < 0.007:
        storm_dt[i] = 0.007
                       
    # Spatially distribute PET and its 30-day-mean (analogous to degree day)
    grid.at_cell['surface__potential_evapotranspiration_rate'] = (
        daily_pet[julian])
    grid.at_cell['surface__potential_evapotranspiration_30day_mean'] = (
        EP30[julian])
    grid.at_cell['surface__potential_evapotranspiration_rate__grass'] = (
        np.full(grid.number_of_cells, daily_pet[julian, 0]))
    # Assign spatial rainfall data
    grid.at_cell['rainfall__daily_depth'] = (
        np.full(grid.number_of_cells, precip[i]))

    # Update soil moisture component
    current_time = soil_moisture.update(current_time, Tr=storm_dt[i],
                                        Tb=inter_storm_dt[i])

    # Decide whether its growing season or not
    if julian != 364:
        if EP30[julian+1, 0] > EP30[julian, 0]:
            pet_threshold = 1
            # 1 corresponds to ETThresholdup (begin growing season)
            if EP30[julian, 0] > vegetation._ETthresholdup:
                growing_season = True
            else:
                growing_season = False
        else:
            pet_threshold = 0
            # 0 corresponds to ETThresholddown (end growing season)
            if EP30[julian, 0] > vegetation._ETthresholddown:
                growing_season = True
            else:
                growing_season = False

    # Update vegetation component
    vegetation.update(PETThreshold_switch=pet_threshold, Tb=inter_storm_dt[i],
                      Tr=storm_dt[i])

    if growing_season:
        # Update yearly cumulative water stress data
        Tg += (storm_dt[i]+inter_storm_dt[i])/24.    # Incrementing growing season storm count
        water_stress += ((grid.at_cell['vegetation__water_stress']) *
                         inter_storm_dt[i]/24.)

    # Update spatial PFTs with Cellular Automata rules
    if (current_time - time_check) >= 1.:
        if yrs < data['ch_yr_1']:
            fires = data['fires']
            no_fires_per_year_min = data['no_fires_per_year_min']
            no_fires_per_year_max = data['no_fires_per_year_max']
            stoc_fire_extent_max = data['stoc_fire_extent_max']
            stoc_fire_extent_min = data['stoc_fire_extent_min']
        elif data['ch_yr_1'] < yrs <= data['ch_yr_2']:
            fires = data['fires_1']
            no_fires_per_year_min = data['no_fires_per_year_min_1']
            no_fires_per_year_max = data['no_fires_per_year_max_1']
            stoc_fire_extent_max = data['stoc_fire_extent_max_1']
            stoc_fire_extent_min = data['stoc_fire_extent_min_1']
        else:
            fires = data['fires_2']
            no_fires_per_year_min = data['no_fires_per_year_min_2']
            no_fires_per_year_max = data['no_fires_per_year_max_2']
            stoc_fire_extent_max = data['stoc_fire_extent_max_2']
            stoc_fire_extent_min = data['stoc_fire_extent_min_2']
        # Introducing Fires
        if fires:
            number_of_fires_per_year = np.random.randint(
                                            no_fires_per_year_min,
                                            high=no_fires_per_year_max+1)
            n_fires_per_year[yrs] = number_of_fires_per_year
            per_burn_per_fire = (stoc_fire_extent_min +
                                 np.random.rand() *
                                 (stoc_fire_extent_max -
                                  stoc_fire_extent_min))
            per_dev_fire = (0.3 * per_burn_per_fire)
            per_fire_area[yrs] = per_burn_per_fire
            (V, burnt_locs, ignition_cells) = sd.initiate_fires(
                    n_fires=number_of_fires_per_year,
                    fire_area_mean=per_burn_per_fire,
                    fire_area_dev=per_dev_fire,
                    sh_susc=data['sh_susc'], tr_susc=data['tr_susc'],
                    gr_susc=data['gr_susc'], sh_seed_susc=data['sh_seed_susc'],
                    tr_seed_susc=data['tr_seed_susc'],
                    gr_vuln=data['gr_vuln'], sh_vuln=data['sh_vuln'],
                    tr_vuln=data['tr_vuln'], sh_seed_vuln=data['sh_seed_vuln'],
                    tr_seed_vuln=data['tr_seed_vuln'])
            per_yr_burnt_area_fr[yrs] = np.float(len(burnt_locs))/np.float(grid1.number_of_cells)
        if grazing:
            (V, grazed_cells) = sd.graze(
                    grazing_pressure=data['gr_pressure'])
            per_yr_grazed_area_fr[yrs] = np.float(len(grazed_cells))/np.float(grid1.number_of_cells)

        if data['add_left_edge']:
            set_pft_at_cells(lft_cells, grid1, pft=data['left_pft'])
        if data['add_right_edge']:
            set_pft_at_cells(rght_cells, grid1, pft=data['right_pft'])
        if data['add_bottom_edge']:
             set_pft_at_cells(b_cells, grid1, pft=data['bottom_pft'])
        if data['add_top_edge']:
             set_pft_at_cells(t_cells, grid1, pft=data['top_pft'])
        veg_type = grid1.at_cell['vegetation__plant_functional_type']
        veg_cov[yrs, :] = calc_veg_cov(grid1, veg_type)
        if yrs % yr_step == 0:
            print('Elapsed time = ', yrs, ' years')
            np.save('vegtype_'+str(yrs)+'.npy', veg_type)
            np.save('veg_cov.npy', veg_cov)

        if yrs % 500 == 0:
            plot_veg_type(grid1, veg_type, yrs, savename=output_sub_fldr)

        cum_water_stress = np.choose(veg_type, water_stress)
        grid1.at_cell['vegetation__cumulative_water_stress'] = (
            cum_water_stress / Tg)
        vegca.update()
        time_check = np.floor(current_time)
        water_stress = 0
        Tg = 0
        yrs += 1
        # Adding woody plants if they vanish - 05Dec18
        if yrs % data['tr_yr_step'] == 0:
            if data['to_add_trees']:
                if data['const_fr_trees']:
                    fr_tree_seeds = data['fr_addition_trees']
                else:
                    fr_tree_seeds = get_fr_seeds(
                            veg_cov[yrs, 2]/100.,
                            x_min=data['min_tr_percent'],
                            x_max=data['max_tr_percent'],
                            y_min=data['min_fr_trees'],
                            y_max=data['max_fr_trees']
                            )
                veg_type = add_trees(grid1,
                                 fraction_addition=fr_tree_seeds,
                                 chck_presence=data['tr_chck_presence'])
        if yrs % data['sh_yr_step'] == 0:    
            if data['to_add_shrubs']:
                if data['const_fr_shrubs']:
                    fr_shrub_seeds = data['fr_addition_shrubs']
                else:
                    fr_shrub_seeds = get_fr_seeds(
                            veg_cov[yrs, 1]/100.,
                            x_min=data['min_sh_percent'],
                            x_max=data['max_sh_percent'],
                            y_min=data['min_fr_shrubs'],
                            y_max=data['max_fr_shrubs']
                            )
                veg_type = add_shrubs(grid1,
                                 fraction_addition=fr_shrub_seeds,
                                 chck_presence=data['sh_chck_presence'])

wallclock_stop = time.clock()
walltime = (wallclock_stop - wallclock_start) / 60.  # in minutes
print('Time_consumed = ', walltime, ' minutes')
pickle_out = open("inputs.pickle","wb")
pickle.dump(data, pickle_out)
pickle_out.close()
np_files = {
        'n_fires_per_year': n_fires_per_year,
        'per_fire_area': per_fire_area,
        'veg_cov': veg_cov,
        'per_yr_burnt_area_fr': per_yr_burnt_area_fr,
        'per_yr_grazed_area_fr': per_yr_grazed_area_fr,
        }
save_np_files(np_files=np_files)
try:
    os.mkdir('../for_comparison')
except OSError:
    pass
finally:
    os.chdir('../for_comparison')
plot_veg_type(grid1, veg_type, yrs, savename=output_sub_fldr)
