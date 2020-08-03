
"""
@author: Sai Nudurupati & Erkan Istanbulluoglu
"""

import os
import time
import numpy as np
from landlab import RasterModelGrid, load_params
from ecohyd_functions_flat import (initialize, empty_arrays,
                                   create_pet_lookup, calc_veg_cov,
                                   plot_veg_cov)


grid1 = RasterModelGrid((285, 799), spacing=(5., 5.))
grid = RasterModelGrid((5, 4), spacing=(5., 5.))


results_main_folder = 'E:\pub_wrr_ecohyd\ca_flat'   # Enter the path where you want the results to be written
sub_fldr_name = 'sim_5m_82_lnger_1'
data = load_params('sm_calib_82.yaml')

(precip_dry, precip_wet, radiation, pet_tree, pet_shrub,
 pet_grass, soil_moisture, vegetation, vegca) = initialize(data, grid, grid1)

n_years = 200 # Approx number of years for model to run
yr_step = 20

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

(EP30, daily_pet) = create_pet_lookup(grid, radiation, pet_tree, pet_shrub,
                                      pet_grass, daily_pet, rad_factor, EP30)

precip = np.load('precip_flt_31_lnger.npy')
storm_dt = np.load('storm_dt_flt_31_lnger.npy')
inter_storm_dt = np.load('interstorm_dt_flt_31_lnger.npy')

# Represent current time in years
current_time = 0 # Start from first day of Jan

# Keep track of run time for simulation - optional
wallclock_start = time.clock() # Recording time taken for simulation

# declaring few variables that will be used in the storm loop
time_check = 0. # Buffer to store current_time at previous storm
yrs = 0 # Keep track of number of years passed
water_stress = 0. # Buffer for Water Stress
Tg = 0 # Growing season in days

# Saving
try:
    os.mkdir(results_main_folder)
except OSError:
    pass
finally:
    os.chdir(results_main_folder)


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

## Run storm Loop
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
    vegetation.update(PETthreshold_switch=pet_threshold, Tb=inter_storm_dt[i],
                      Tr=storm_dt[i])

    if growing_season:
        # Update yearly cumulative water stress data
        Tg += (storm_dt[i]+inter_storm_dt[i])/24.    # Incrementing growing season storm count
        water_stress += ((grid.at_cell['vegetation__water_stress']) *
                         inter_storm_dt[i]/24.)

    # Update spatial PFTs with Cellular Automata rules
    if (current_time - time_check) >= 1.:
        veg_type = grid1.at_cell['vegetation__plant_functional_type']
        veg_cov[yrs, :] = calc_veg_cov(grid1, veg_type)
        if yrs % yr_step == 0:
            print('Elapsed time = ', yrs, ' years')
            np.save('vegtype_'+str(yrs)+'.npy', veg_type)
            np.save('vegetation_cover', veg_cov)
        cum_water_stress = np.choose(veg_type, water_stress)
        grid1.at_cell['vegetation__cumulative_water_stress'] = (
            cum_water_stress/Tg)
        vegca.update()
        time_check = np.floor(current_time)
        water_stress = 0
        Tg = 0
        yrs += 1

wallclock_stop = time.clock()
walltime = (wallclock_stop - wallclock_start) / 60. # in minutes
print('Time_consumed = ', walltime, ' minutes')
np.save('vegetation_cover', veg_cov)
plot_veg_cov(veg_cov, yrs=yrs, savename='veg_cov_3sp')
