
# -*- coding: utf-8 -*-
"""
@author: Sai Nudurupati & Erkan Istanbulluoglu
"""

import os
import time
import sys
import numpy as np
from landlab.io import read_esri_ascii
from landlab import RasterModelGrid, load_params
from ecohyd_functions_dem_bins import (initialize, empty_arrays,
                                       create_pet_lookup,
                                       get_slp_asp_mapping,
                                       get_sm_mapper,
                                       calc_veg_cov_wtrshd,
                                       calc_veg_cov_asp,
                                       plot_veg_cov)

(grid, elevation) = read_esri_ascii('sev5m_nad27.txt')    # Read the DEM
grid.set_nodata_nodes_to_closed(elevation, -9999.)
# Create a grid to hold combination of 6 veg_types X 9 slope_bins X 12 aspect_bins
# therefore, number of cells needed = 648. 83X10 nodal grid - arbitrary
# size to give a grid with 648 cells
sm_grid = RasterModelGrid((83, 10), spacing=(5., 5.))     # Representative grid
# create a grid for radiation factor: 9 slope_bins X 12 aspect_bins = 108 cells
rad_grid = RasterModelGrid((14, 11), spacing=(5., 5.))

sub_fldr_name = 'sm_calib_82_lnger_1'
data = load_params('sm_calib_82.yaml')  # Creates dictionary that holds the inputs
# sub_fldr_name = sys.argv[1] + '_lnger_1'
# data = load_params(sys.argv[1]+'.yaml')  # Creates dictionary that holds the inputs

(precip_dry, precip_wet, radiation, pet_tree, pet_shrub, pet_grass,
 soil_moisture, vegetation, vegca) = initialize(data, grid, sm_grid, rad_grid,
                                                elevation)

n_years = 200       # Approx number of years for model to run
yr_step = 20        # Step for printing current time
# Calculate approximate number of storms per year
fraction_wet = (data['doy__end_of_monsoon']-data['doy__start_of_monsoon'])/365.
fraction_dry = 1 - fraction_wet
no_of_storms_wet = (8760 * (fraction_wet)/(data['mean_interstorm_wet'] +
                    data['mean_storm_wet']))
no_of_storms_dry = (8760 * (fraction_dry)/(data['mean_interstorm_dry'] +
                    data['mean_storm_dry']))
n = int(n_years * (no_of_storms_wet + no_of_storms_dry))

(precip, inter_storm_dt, storm_dt, daily_pet,
 rad_factor, EP30, pet_threshold) = empty_arrays(n, n_years, grid,
                                                 sm_grid, rad_grid)

veg_type = np.empty([grid.number_of_cells], dtype=int)
veg_cov = np.zeros([n_years, 6])   # veg cover fractions
veg_cov_asp = np.zeros([n_years, 13, 6])  # veg cover fractions w.r.t aspect
lai_veg = np.empty([grid.number_of_cells], dtype=int)

(EP30, daily_pet, rad_factor) = create_pet_lookup(radiation, pet_tree, 
                                                  pet_shrub, pet_grass,
                                                  daily_pet, rad_factor,
                                                  EP30, rad_grid)

precip = np.load('precip_flt_31_lnger.npy')
storm_dt = np.load('storm_dt_flt_31_lnger.npy')
inter_storm_dt = np.load('interstorm_dt_flt_31_lnger.npy')

# Get slp_asp_mapper which will then be used to get the mapping
# between sm_grid and grid for every new vegetation__plant_functional_type
slp_asp_mapper = get_slp_asp_mapping(grid)

# # Represent current time in years
current_time = 0            # Start from first day of Jan

# Keep track of run time for simulation—optional
wallclock_start = time.clock()     # Recording time taken for simulation

# declaring few variables that will be used in storm loop
time_check = 0.     # Buffer to store current_time at previous storm
yrs = 0             # Keep track of number of years passed
water_stress = 0.             # Buffer for Water Stress
Tg = 0        # Growing season in days

# Saving
try:
    os.mkdir('E:\pub_wrr_ecohyd\ca_dem')
except OSError:
    pass
finally:
    os.chdir('E:\pub_wrr_ecohyd\ca_dem')

try:
    os.mkdir('output')
except OSError:
    pass
finally:
    os.chdir('output')

try:
    os.mkdir(sub_fldr_name)
except OSError:
    pass
finally:
    os.chdir(sub_fldr_name)

# # Run storm Loop
i = -1
while yrs < n_years:
    i += 1
    # # Update objects
    # Calculate Day of Year (DOY)
    julian = np.int(np.floor((current_time - np.floor(current_time)) * 365.))

    # Making sure that minimum storm_dt is 10 min (0.007 hrs) - added: 19Apr18_12:19pm
    if storm_dt[i] < 0.007:
        storm_dt[i] = 0.007

    # Spatially distribute PET and its 30-day-mean (analogous to degree day)
    sm_grid.at_cell['surface__potential_evapotranspiration_rate'] = (
        (np.choose(sm_grid.at_cell['vegetation__plant_functional_type'],
         daily_pet[julian])) * rad_factor[julian])
    sm_grid.at_cell['surface__potential_evapotranspiration_30day_mean'] = (
        (np.choose(sm_grid.at_cell['vegetation__plant_functional_type'],
         EP30[julian])) * rad_factor[julian])
    sm_grid.at_cell['surface__potential_evapotranspiration_rate__grass'] = (
        daily_pet[julian, 0] * rad_factor[julian])

    # Assign spatial rainfall data
    sm_grid.at_cell['rainfall__daily_depth'] = (precip[i] *
                                             np.ones(sm_grid.number_of_cells))

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
    vegetation.update(PETthreshold_switch=pet_threshold, Tr=storm_dt[i],
                      Tb=inter_storm_dt[i])

    if growing_season:
        Tg += (storm_dt[i]+inter_storm_dt[i])/24.    # Incrementing growing season storm count
        # Update yearly cumulative water stress data
        water_stress += ((sm_grid.at_cell['vegetation__water_stress']) *
                         inter_storm_dt[i]/24.)

    # Cellular Automata
    if (current_time - time_check) >= 1.:
        veg_type = grid.at_cell['vegetation__plant_functional_type']
        veg_cov[yrs, :] = calc_veg_cov_wtrshd(grid, veg_type)
        veg_cov_asp[yrs, :, :] = calc_veg_cov_asp(
                            grid, veg_type, slp_asp_mapper)
        # get mapping between sm_grid and grid
        mapper = get_sm_mapper(grid, slp_asp_mapper)
        lai_veg = np.take(sm_grid.at_cell['vegetation__live_leaf_area_index'],
                          mapper)
        if yrs % yr_step == 0:
            print('Elapsed time = ', yrs, ' years')
            np.save('vegtype_'+str(yrs)+'.npy', veg_type)
            np.save('livelai_'+str(yrs)+'.npy', lai_veg)
            np.save('veg_cover_wtrshd', veg_cov)
            np.save('veg_cover_aspect', veg_cov_asp)
        cum_wtr_str = np.take(water_stress, mapper)
        grid.at_cell['vegetation__cumulative_water_stress'] = (
                                        cum_wtr_str/Tg)
        vegca.update()
        time_check = np.floor(current_time)
        water_stress = 0
        Tg = 0
        yrs += 1

wallclock_stop = time.clock()
walltime = (wallclock_stop - wallclock_start)/60.    # in minutes
print('Time_consumed = ', walltime, ' minutes')
np.save('veg_cover_wtrshd', veg_cov)
np.save('veg_cover_aspect', veg_cov_asp)
plot_veg_cov(veg_cov, yrs=yrs, savename='veg_cov_3sp')