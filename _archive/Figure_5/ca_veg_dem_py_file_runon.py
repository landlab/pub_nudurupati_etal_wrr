
# -*- coding: utf-8 -*-
"""
@author: Sai Nudurupati & Erkan Istanbulluoglu
"""

import os
import time

import numpy as np

from landlab.io import read_esri_ascii
from landlab import RasterModelGrid, load_params
from ecohyd_functions_dem_runon import (initialize, empty_arrays,
                                        create_pet_lookup,
                                        calc_veg_cov_asp,
                                        calc_veg_cov_wtrshd,
                                        plot_veg_cov)

(grid, elevation) = read_esri_ascii('hugo5m_wgs84.txt')    # Read the DEM
grid.set_nodata_nodes_to_closed(elevation, -9999.)
#grid = RasterModelGrid((142, 399), spacing=(10., 10.))
#elevation = np.full(grid.number_of_nodes, 1700.)
grid1 = RasterModelGrid((5, 4), spacing=(5., 5.))     # Representative grid
# grid2 for runon - recognize core cells and close cells outside wtrshd
(grid2, elevation) = read_esri_ascii('hugo5m_wgs84.txt')
#grid2 = RasterModelGrid((142, 399), spacing=(10., 10.))

sub_fldr_name = 'sm_hugo5m_runon_82_lnger_3'   # veg1 flag is for veg compositions
data = load_params('sm_runon_82.yaml')  # Creates dictionary that holds the inputs
print('sub folder = ', sub_fldr_name)
n_years = 20001       # Approx number of years for model to run
yr_step = 200        # Step for printing current time
use_preloaded_pft = 0  # 1 if we want to use an existing PFT map to start with.
if use_preloaded_pft:
    initial_pft_field = np.load('sm_hugo5m_bin_82_pft_19800yrs.npy')

(precip_dry, precip_wet, radiation, rad_pet, pet_tree, pet_shrub, pet_grass,
 soil_moisture, vegetation, vegca, ordered_cells) = initialize(data, grid,
 grid1, grid2, elevation)

if use_preloaded_pft:
    grid.at_cell['vegetation__plant_functional_type'] = initial_pft_field

# Calculate approximate number of storms per year
fraction_wet = (data['doy__end_of_monsoon']-data['doy__start_of_monsoon'])/365.
fraction_dry = 1 - fraction_wet
no_of_storms_wet = (8760 * (fraction_wet)/(data['mean_interstorm_wet'] +
                    data['mean_storm_wet']))
no_of_storms_dry = (8760 * (fraction_dry)/(data['mean_interstorm_dry'] +
                    data['mean_storm_dry']))
n = int(n_years * (no_of_storms_wet + no_of_storms_dry))

(precip, inter_storm_dt, storm_dt, daily_pet,
 rad_factor, EP30, pet_threshold) = empty_arrays(n, n_years, grid, grid1)

veg_type = np.empty([grid.number_of_cells], dtype=int)
veg_cov = np.zeros([n_years, 6])
veg_cov_asp = np.zeros([n_years, 13, 6])  # veg cover fractions w.r.t aspect
lai_veg = np.empty([grid.number_of_cells], dtype=int)

(EP30, daily_pet, rad_factor) = create_pet_lookup(radiation, pet_tree, 
                                                  pet_shrub, pet_grass,
                                                  daily_pet, rad_factor,
                                                  EP30, rad_pet, grid)

precip = np.load('precip_flt_31_lnger.npy')
storm_dt = np.load('storm_dt_flt_31_lnger.npy')
inter_storm_dt = np.load('interstorm_dt_flt_31_lnger.npy')

# # Represent current time in years
current_time = 0            # Start from first day of Jan

# Keep track of run time for simulation—optional
wallclock_start = time.clock()     # Recording time taken for simulation

# declaring few variables that will be used in storm loop
time_check = 0.     # Buffer to store current_time at previous storm
yrs = 0             # Keep track of number of years passed
water_stress = 0.             # Buffer for Water Stress
Tg = 0        # Counter for growing season in number of storms

## Since we will be saving files intermittently, lets change the dir
try:
    os.chdir('E:\Research_UW_Sai_PhD\Jan_2017\ca_dem_inc_runon_apr18')
except OSError:
    os.chdir('/data1/sai_projs/ecohyd_paper_03May18/ca_dem_inc_runon_apr18')
finally:
    pass

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

#    P_annual[yrs] += precip[i]
    # Spatially distribute PET and its 30-day-mean (analogous to degree day)
    grid.at_cell['surface__potential_evapotranspiration_rate'] = (
        (np.choose(grid.at_cell['vegetation__plant_functional_type'],
         daily_pet[julian])) * rad_factor[julian])
    grid.at_cell['surface__potential_evapotranspiration_30day_mean'] = (
        (np.choose(grid.at_cell['vegetation__plant_functional_type'],
         EP30[julian])) * rad_factor[julian])
    grid.at_cell['surface__potential_evapotranspiration_rate__grass'] = (
        daily_pet[julian, 0] * rad_factor[julian])

    # Assign spatial rainfall data
    grid.at_cell['rainfall__daily_depth'] = (precip[i] *
                                             np.ones(grid.number_of_cells))

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
        water_stress += ((grid.at_cell['vegetation__water_stress']) *
                         inter_storm_dt[i]/24.)

    # Cellular Automata
    if (current_time - time_check) >= 1.:
        veg_type = grid.at_cell['vegetation__plant_functional_type']
        veg_cov[yrs, :] = calc_veg_cov_wtrshd(grid, veg_type)
        veg_cov_asp[yrs, :, :] = calc_veg_cov_asp(grid, veg_type)
        lai_veg = grid.at_cell['vegetation__live_leaf_area_index']
        if yrs % yr_step == 0:
            print 'Elapsed time = ', yrs, ' years'
            np.save('vegtype_'+str(yrs)+'.npy', veg_type)
            np.save('livelai_'+str(yrs)+'.npy', lai_veg)
            np.save('veg_cov_wtrshd_intermediate', veg_cov)
            np.save('veg_cov_asp_intermediate', veg_cov_asp)
        grid.at_cell['vegetation__cumulative_water_stress'] = water_stress/Tg
        vegca.update()
        soil_moisture.initialize(ordered_cells=ordered_cells, **data)
        vegetation.initialize(**data)
        time_check = np.floor(current_time)
        water_stress = 0
        Tg = 0
        yrs += 1

wallclock_stop = time.clock()
walltime = (wallclock_stop - wallclock_start)/60.    # in minutes
print 'Time_consumed = ', walltime, ' minutes'
np.save('veg_cover_wtrshd', veg_cov)
np.save('veg_cover_aspect', veg_cov_asp)
plot_veg_cov(veg_cov, yrs=yrs, savename='veg_cov_3sp')
