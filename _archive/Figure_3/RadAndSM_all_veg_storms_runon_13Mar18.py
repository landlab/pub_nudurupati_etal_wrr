

# Authors Sai Nudurupati & Erkan Istanbulluoglu

import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
#from landlab import RasterModelGrid
from landlab.io import read_esri_ascii
from landlab.components import Radiation
from landlab.components import PotentialEvapotranspiration
from landlab.components import SoilMoisture
from landlab.components import Vegetation
from landlab import load_params
from landlab.plot import imshow_grid
import datetime as dt
import pickle
#from landlab.plot import imshow_grid
from funcs_for_including_runon_in_soil_moisture import (
    get_ordered_cells_for_soil_moisture)
#from ecohyd_functions_dem_runon_vegcom import (initialize, empty_arrays,
#                                               create_pet_lookup)
import warnings
warnings.filterwarnings("ignore")

# Function to compose spatially distribute PFT
def compose_veg_grid(grid, percent_bare=0.4, percent_grass=0.2,
                     percent_shrub=0.2, percent_tree=0.2):
    no_cells = grid.number_of_cells
    veg_grid = 3 * np.ones(grid.number_of_cells, dtype=int)
    shrub_point = int(percent_bare * no_cells)
    tree_point = int((percent_bare + percent_shrub) * no_cells)
    grass_point = int((1 - percent_grass) * no_cells)
    veg_grid[shrub_point:tree_point] = 1
    veg_grid[tree_point:grass_point] = 2
    veg_grid[grass_point:] = 0
    np.random.shuffle(veg_grid)
    return veg_grid

# Point to the input DEM
_DEFAULT_INPUT_FILE_1 = os.path.join(os.path.dirname(__file__),
                                 'hugo10mws.txt')

""" Importing Grid and Elevations from DEM """
## For modeled Radiation model
(grid,elevation) = read_esri_ascii(_DEFAULT_INPUT_FILE_1)
(grid2, elevation) = read_esri_ascii(_DEFAULT_INPUT_FILE_1)
grid.at_node['topographic__elevation'] = elevation
grid2.at_node['topographic__elevation'] = elevation

## Name of the folder you want to store the outputs
sub_fldr_name = 'trial21_runon_strms_wtrshd_veg12_pap'   # veg1 flag is for veg compositions
data = load_params('trial21_veg12.yaml')  # Creates dictionary that holds the inputs
runon_switch = 1

#days_n = 1266
nfs_data = pickle.load(open("nfs_daily_storm_analysis.p", "rb"))
P_NFS = np.array(nfs_data['storm_depth'])
P_NFS = ma.fix_invalid(P_NFS, fill_value=0.).data
Tr_NFS = np.array(nfs_data['storm_duration'])
Tb_NFS = np.array(nfs_data['inter_storm_duration'])
strm_days = nfs_data['storm_start_time']
days_n = int(P_NFS.shape[0])

sfs_data = pickle.load(open("sfs_daily_storm_analysis.p", "rb"))
P_SFS = np.array(sfs_data['storm_depth'])
P_SFS = ma.fix_invalid(P_SFS, fill_value=0.).data
Tr_SFS = np.array(sfs_data['storm_duration'])
Tb_SFS = np.array(sfs_data['inter_storm_duration'])

ctrl_data = pickle.load(open("ctrl_daily_storm_analysis.p", "rb"))
P_ctrl = np.array(ctrl_data['storm_depth'])

# Temperature and other inputs
Tavg_NFS_ = np.array(nfs_data['tavg_nfs'])
Tmax_NFS_ = np.array(nfs_data['tmin_nfs'])
Tmin_NFS_ = np.array(nfs_data['tmax_nfs'])
Tavg_NFS = ma.fix_invalid(Tavg_NFS_, fill_value = 0.).data
Tmax_NFS = ma.fix_invalid(Tmax_NFS_, fill_value = 0.).data
Tmin_NFS = ma.fix_invalid(Tmin_NFS_, fill_value = 0.).data
RH_NFS = np.array(nfs_data['rh_nfs'])  # Load RelativeHumidity %
RH_NFS = ma.fix_invalid(RH_NFS, fill_value=0.).data
RH_NFS[RH_NFS<0.] = 0.
WS_NFS = np.array(nfs_data['windspeed_nfs'])  # Load Wind Speed in m/s
WS_NFS = ma.fix_invalid(WS_NFS, fill_value=0.).data
WS_NFS[WS_NFS<0.] = 0.
Rad_obs_NFS_daily = np.array(nfs_data['rad_obs_nfs'])  # W/m^2

Tavg_SFS_ = np.array(sfs_data['tavg_sfs'])
Tmax_SFS_ = np.array(sfs_data['tmin_sfs'])
Tmin_SFS_ = np.array(sfs_data['tmax_sfs'])
Tavg_SFS = ma.fix_invalid(Tavg_SFS_, fill_value = 0.).data
Tmax_SFS = ma.fix_invalid(Tmax_SFS_, fill_value = 0.).data
Tmin_SFS = ma.fix_invalid(Tmin_SFS_, fill_value = 0.).data
RH_SFS = np.array(sfs_data['rh_sfs'])  # Load RelativeHumidity %
RH_SFS = ma.fix_invalid(RH_SFS, fill_value=0.).data
RH_SFS[RH_SFS<0.] = 0.
WS_SFS = np.array(sfs_data['windspeed_sfs'])  # Load Wind Speed in m/s
WS_SFS = ma.fix_invalid(WS_SFS, fill_value=0.).data
WS_SFS[WS_SFS<0.] = 0.
Rad_obs_SFS_daily = np.array(sfs_data['rad_obs_sfs'])  # W/m^2

Tavg_ctrl = np.array(ctrl_data['tavg_ctrl'])
Tmax_ctrl = np.array(ctrl_data['tmin_ctrl'])
Tmin_ctrl = np.array(ctrl_data['tmax_ctrl'])
RH_ctrl = np.array(ctrl_data['rh_ctrl'])  # Load RelativeHumidity %
RH_ctrl[RH_ctrl<0.] = 0.
WS_ctrl = np.array(ctrl_data['windspeed_ctrl'])  # Load Wind Speed in m/s
WS_ctrl[WS_ctrl<0.] = 0.
Rad_obs_ctrl_daily = np.array(ctrl_data['rad_obs_ctrl'])  # W/m^2

# Calculating watershed inputs (simple watershed average observations) 
Tavg_obs = (Tavg_NFS + Tavg_SFS + Tavg_ctrl)/3.
Tmax_obs = (Tmax_NFS + Tmax_SFS + Tmax_ctrl)/3.
Tmin_obs = (Tmin_NFS + Tmin_SFS + Tmin_ctrl)/3.
rad_obs = (Rad_obs_NFS_daily + Rad_obs_SFS_daily + Rad_obs_ctrl_daily)/3.
rh_obs = (RH_NFS + RH_SFS + RH_ctrl)/3.
ws_obs = (WS_NFS + WS_SFS + WS_ctrl)/3.
precip = (P_NFS + P_SFS + P_ctrl)/3.

## Loading N3 observations from figure 7 Hugo2013 paper
#print 'current dir: ', os.getcwd()
data_loaded = pickle.load(open("N3_obs_Hugo13_26mar18.p", "rb"))
sm_obs10cm_N3_int = data_loaded['sm_obs10cm_N3_int']
sm_obs20cm_N3_int = data_loaded['sm_obs20cm_N3_int']
sm_obs10cm_N3_und = data_loaded['sm_obs10cm_N3_und']
sm_obs20cm_N3_und = data_loaded['sm_obs20cm_N3_und']
date_obs10cm_N3_int = data_loaded['date_obs10cm_N3_int']
date_obs20cm_N3_int = data_loaded['date_obs20cm_N3_int']
date_obs10cm_N3_und = data_loaded['date_obs10cm_N3_und']
date_obs20cm_N3_und = data_loaded['date_obs20cm_N3_und']
date_et_nfs_r = np.array(data_loaded['date_et_nfs'])   # From figure 11 Hugo2013 paper
et_nfs_r = np.array(data_loaded['et_nfs'])      # From figure 11 Hugo2013 paper
date_et_nfs = np.unique(date_et_nfs_r)
et_nfs = np.array([np.mean(et_nfs_r[np.where(date_et_nfs_r==x)[0]]) for x in date_et_nfs])
strt_n = np.where(date_et_nfs == dt.datetime(2009, 03, 20, 0, 0))[0][0]
stp_n = np.where(date_et_nfs == dt.datetime(2009, 9, 01, 0, 0))[0][0]
to_del_n = np.where(et_nfs[strt_n:stp_n] < 0.2)[0]
date_et_nfs = np.delete(date_et_nfs, strt_n+to_del_n)
et_nfs = np.delete(et_nfs, strt_n+to_del_n)

date_et_sfs_r = np.array(data_loaded['date_et_sfs'])
et_sfs_r = np.array(data_loaded['et_sfs'])
date_et_sfs = np.unique(date_et_sfs_r)
et_sfs = np.array([np.mean(et_sfs_r[np.where(date_et_sfs_r==x)[0]]) for x in date_et_sfs])
strt_s = np.where(date_et_sfs == dt.datetime(2009, 07, 27, 0, 0))[0][0]
stp_s = np.where(date_et_sfs == dt.datetime(2009, 9, 01, 0, 0))[0][0]
to_del_s = np.where(et_sfs[strt_s:stp_s] < 0.1)[0]
date_et_sfs = np.delete(date_et_sfs, strt_s+to_del_s)
et_sfs = np.delete(et_sfs, strt_s+to_del_s)

## 30Mar18 - adding two precipitation multipliers
#nfs_mon1_end = np.where(np.array(strm_days) == dt.datetime(2008, 10, 01, 0, 0))[0][0]
# 04Apr18 - extending precip_mult_1 to datetime(2009, 5, 21, 0, 0)
nfs_mon1_end = np.where(np.array(strm_days) == dt.datetime(2009, 5, 21, 0, 0))[0][0]
precip[:nfs_mon1_end] *= data['precip_mult_1']   # introduced 30Mar18
precip[nfs_mon1_end:] *= data['precip_mult_2']

# Input veg composition
grid['cell']['vegetation__plant_functional_type'] = compose_veg_grid(
        grid, percent_bare=data['percent_bare_initial'],
        percent_grass=data['percent_grass_initial'],
        percent_shrub=data['percent_shrub_initial'],
        percent_tree=data['percent_tree_initial'])
#grid.at_cell['vegetation__plant_functional_type'][[1748, 1692, 1523]] = 2   # All Trees - 2 means Tree
# Create radiation, PET and soil moisture objects
## For modeled Radiation model
Rad = Radiation(grid)
pet_grass = PotentialEvapotranspiration(grid, method='PenmanMonteith',
                albedo=data['albedo_grass'], rl=data['rl_grass'],
                zveg=data['zveg_grass'], LAI=data['LAI_grass'],
                zm=data['zm_grass'], zh=data['zh_grass'])
pet_shrub = PotentialEvapotranspiration(grid, method='PenmanMonteith',
                albedo=data['albedo_shrub'], rl=data['rl_shrub'],
                zveg=data['zveg_shrub'], LAI=data['LAI_shrub'],
                zm=data['zm_shrub'], zh=data['zh_shrub'])
pet_tree = PotentialEvapotranspiration(grid, method='PenmanMonteith',
                albedo=data['albedo_tree'], rl=data['rl_tree'],
                zveg=data['zveg_tree'], LAI=data['LAI_tree'],
                zm=data['zm_tree'], zh=data['zh_tree'])

if runon_switch:
    (ordered_cells, grid2) = get_ordered_cells_for_soil_moisture(
            grid2, outlet_id=1449-(2*58)-2)
    grid.at_node['flow__receiver_node'] = (
            grid2.at_node['flow__receiver_node'])

SM = SoilMoisture(grid, ordered_cells=ordered_cells, **data)

VEG = Vegetation(grid, **data)    # Vegetation object

# Create arrays to store modeled data
## For modeled Radiation model
Rad_mod_NFS = np.zeros([days_n, grid.number_of_cells], dtype=float) # Total Short Wave Radiation
PET_mod_NFS = np.zeros([days_n, grid.number_of_cells], dtype=float) # Potential Evapotranspiration
AET_mod_NFS = np.zeros([days_n, grid.number_of_cells], dtype=float) # Actual Evapotranspiration
SM_mod_NFS = np.zeros([days_n, grid.number_of_cells], dtype=float)  # Soil Moisture Saturation Fraction
Rad_Factor = np.zeros([days_n, grid.number_of_cells], dtype=float)
AET_annual = np.zeros([4, grid.number_of_cells], dtype=float)  # mm - For annual AET ((actual)evapotranspiration)
P_annual = np.zeros([4, grid.number_of_cells], dtype=float)  # mm - For annual P (precipitation)
drainage_annual = np.zeros([4, grid.number_of_cells], dtype=float)
AET_over_P = np.zeros([4, grid.number_of_cells], dtype=float)  # AET/P ratio - annual
EP30 = np.zeros([days_n, grid.number_of_cells], dtype=float)  # 30 day moving average of PET - only for target cell
pet_thresh_n = np.zeros(days_n)  # record PET_threshold - growing season trigger
LAI_live_mod_NFS = np.zeros([days_n, grid.number_of_cells], dtype=float)
LAI_dead_mod_NFS = np.zeros([days_n, grid.number_of_cells], dtype=float)
biomass_live_NFS = np.zeros([days_n, grid.number_of_cells], dtype=float)
biomass_dead_NFS = np.zeros([days_n, grid.number_of_cells], dtype=float)
drainage_NFS = np.zeros([days_n, grid.number_of_cells], dtype=float)
runon_annual = np.zeros([4, grid.number_of_cells], dtype=float)
runoff_annual = np.zeros([4, grid.number_of_cells], dtype=float)

target_NFS = 1692  # N3 location NFS
target_SFS = 2028  # S3 location SFS
Rad_mod_NFS_target = np.zeros(days_n, dtype = float)
SM_mod_NFS_target = np.zeros(days_n, dtype = float)
PET_mod_NFS_target = np.zeros(days_n, dtype = float)
AET_mod_NFS_target = np.zeros(days_n, dtype = float)
Rad_FLAT = np.zeros(days_n, dtype = float)
PET_FLAT = np.zeros(days_n, dtype = float)

# Loading stored variables from Hugo Site
SM_obs_N1_und = np.zeros(days_n) # np.load('SM_obs_S1_und.npy')
SM_obs_N2_und = np.array(nfs_data['sm_obs_N2_und'])
SM_obs_N3_und = np.array(nfs_data['sm_obs_N3_und'])
SM_obs_N4_und = np.array(nfs_data['sm_obs_N4_und'])
SM_obs_N1_und /= (SM._soil_pc[target_NFS]*100)
SM_obs_N2_und /= (SM._soil_pc[target_NFS]*100)
SM_obs_N3_und /= (SM._soil_pc[target_NFS]*100)
SM_obs_N4_und /= (SM._soil_pc[target_NFS]*100)
sm_obs_nfs = (SM_obs_N2_und + SM_obs_N3_und + SM_obs_N4_und)/3.

SM_obs_S1_und = np.zeros(days_n) # np.load('SM_obs_S1_und.npy')
SM_obs_S2_und = np.array(sfs_data['sm_obs_S2_und'])
SM_obs_S3_und = np.array(sfs_data['sm_obs_S3_und'])
SM_obs_S4_und = np.array(sfs_data['sm_obs_S4_und'])
SM_obs_S1_und /= SM._soil_pc[target_SFS]
SM_obs_S2_und /= SM._soil_pc[target_SFS]
SM_obs_S3_und /= SM._soil_pc[target_SFS]
SM_obs_S4_und /= SM._soil_pc[target_SFS]
sm_obs_sfs = (SM_obs_S2_und + SM_obs_S3_und + SM_obs_S4_und)/3.

# Inputs for Soil Moisture object are initialized
VegCover = np.zeros( grid.number_of_cells, dtype = float )
LiveLAI = np.zeros( grid.number_of_cells, dtype = float )
LiveLAI[:] = 0.5
#RUNON = 1
# For modeled Radiation model
grid.at_cell['soil_moisture__initial_saturation_fraction'] = (
        0.2 * np.ones(grid.number_of_cells))
grid['cell']['vegetation__cover_fraction'] = VegCover
grid['cell']['vegetation__live_leaf_area_index'] = LiveLAI

# Calculate current time in years such that Julian time can be calculated
## For modeled Radiation model
current_time = 195/365.25
PET_threshold = 0  # Initial value of PET_threshold = 0
yrs = 0
time_check = 0

# Temporal loop to calculate Potential Evapotranspiration
for i in range(0, days_n):
    ## For modeled Radiation model
    Rad.update(current_time)
    rad_factor = grid.at_cell['radiation__ratio_to_flat_surface']
    pet_grass.update(Tavg=Tavg_obs[i],
                     obs_radiation=rad_obs[i],
                     relative_humidity=rh_obs[i],
                     wind_speed=ws_obs[i], precipitation=precip[i])
    pet_shrub.update(Tavg=Tavg_obs[i],
                     obs_radiation=rad_obs[i],
                     relative_humidity=rh_obs[i],
                     wind_speed=ws_obs[i], precipitation=precip[i])
    pet_tree.update(Tavg=Tavg_obs[i],
                    obs_radiation=rad_obs[i],
                    relative_humidity=rh_obs[i],
                    wind_speed=ws_obs[i], precipitation=precip[i])
    pet_all = [pet_grass._PET_value, pet_shrub._PET_value,
               pet_tree._PET_value, 0., pet_shrub._PET_value,
               pet_tree._PET_value]
    grid.at_cell['surface__potential_evapotranspiration_rate'] = (
        (np.choose(grid.at_cell['vegetation__plant_functional_type'],
         pet_all)) * rad_factor)
    grid.at_cell['surface__potential_evapotranspiration_rate__grass'] = (
        pet_grass._PET_value * rad_factor)
    if i < 30:
        if i == 0:
            EP30[0] = (
                grid['cell']['surface__potential_evapotranspiration_rate'])
        else:
            EP30[i] = np.mean(PET_mod_NFS[:i, :], axis=0)
    else:
        EP30[i] = np.mean(PET_mod_NFS[i-30:i, :], axis=0)

    if i > 1:
        if EP30[i, target_SFS] > EP30[i-1, target_SFS]:
                PET_threshold = 1
                # 1 corresponds to ETThresholdup (begin growing season)
        else:
            PET_threshold = 0
    grid.at_cell['rainfall__daily_depth'] = (precip[i] *
                                             np.ones(grid.number_of_cells))
    grid['cell']['surface__potential_evapotranspiration_30day_mean'] = EP30[i]
    current_time = SM.update(current_time, Tr=Tr_NFS[i], Tb=Tb_NFS[i])
    VEG.update(PETthreshold_switch=PET_threshold)

    # Collect values for analysis
    ## For modeled Radiation model
    Rad_mod_NFS[i][:] = grid['cell']['radiation__incoming_shortwave_flux']
    #Rad_FLAT[i] = PET._Rs#flat
    Rad_Factor[i][:] = grid['cell']['radiation__ratio_to_flat_surface']
    PET_mod_NFS[i][:] = grid['cell']['surface__potential_evapotranspiration_rate']
    AET_mod_NFS[i][:] = grid['cell']['surface__evapotranspiration']
    SM_mod_NFS[i][:] = grid['cell']['soil_moisture__saturation_fraction']
    LAI_live_mod_NFS[i] = grid.at_cell['vegetation__live_leaf_area_index']
    LAI_dead_mod_NFS[i] = grid.at_cell['vegetation__dead_leaf_area_index']
    biomass_live_NFS[i] = grid.at_cell['vegetation__live_biomass']
    biomass_dead_NFS[i] = grid.at_cell['vegetation__dead_biomass']
    drainage_NFS[i] = grid.at_cell['soil_moisture__root_zone_leakage']
    pet_thresh_n[i] = PET_threshold
    AET_annual[yrs, :] += AET_mod_NFS[i]
    P_annual[yrs, :] += grid.at_cell['rainfall__daily_depth']
    drainage_annual[yrs, :] += drainage_NFS[i]
    runoff_annual[yrs] += SM._runoff
    runon_annual[yrs] += SM._runon
    if (current_time - time_check) >= 1.:
        yrs += 1
        time_check = current_time
AET_over_P = AET_annual/P_annual
drain_over_P = drainage_annual/P_annual
runoff_over_P = runoff_annual[:, target_NFS]/P_annual[:, target_NFS]
runon_over_P = runon_annual[:, target_NFS]/P_annual[:, target_NFS]

## 11Mar18: defining NFS and SFS slopes to take ET average
slope, aspect = np.degrees(Rad._slope), np.degrees(Rad._aspect)
# NFS:: cell_rows: 23:33  - cell_ids: 1248:1848
nfs_cells = np.where(np.logical_or(
        np.logical_and(aspect[1288:1848]>0, aspect[1288:1848]<45),
        aspect[1288:1848]>315))[0]
nfs_cells = np.arange(1288, 1848)[nfs_cells]

## SFS:: cell_rows: 32:43  cell_cols: 5:35
sfs_t = []
for row in range(32, 43):
    start_cell = row*56 + 5
    end_cell = row*56 + 35
    sfs_t += range(start_cell, end_cell)
sfs_cells=np.where(np.logical_and(aspect[sfs_t]>135, aspect[sfs_t]<225))[0]
sfs_cells = np.array(sfs_t)[sfs_cells]


## Folder manangement for storing outputs
sim = 'allVeg_aspect_15Mar18'
try:
    os.chdir('E:/Research_UW_Sai_PhD/Jan_2017/Landlab_re_validation_with_Xiaochi13_25Feb17/validation_to_Sevilleta_obs')
except OSError:
    pass

try:
    os.chdir('NFS')
except OSError:
    pass

try:
    os.mkdir('output')
except OSError:
    pass
finally:
    os.chdir('output')

sub_fldr = sub_fldr_name
print sub_fldr

try:
    os.mkdir(sub_fldr)
except OSError:
    pass
finally:
    os.chdir(sub_fldr)


## Saving
np.save('VegType', grid.at_cell['vegetation__plant_functional_type'])

## Calculate Nash-Sutcliffe for ET
# NFS
storm_days = np.array(strm_days)
mod_ids_nfs = np.where(np.in1d(storm_days, date_et_nfs)==True)[0]
mq_et_nfs = np.mean(et_nfs)
Rns2_ET_nfs = (1 - np.sum(np.square(et_nfs -
          np.mean(AET_mod_NFS[:, nfs_cells][mod_ids_nfs], axis=1)))/np.sum(np.square(
          et_nfs - mq_et_nfs)))
#print('Rns2 for Evapotranspiration: ', Rns2_ET_nfs)
# SFS
mod_ids_sfs = np.where(np.in1d(storm_days, date_et_sfs)==True)[0]
mq_et_sfs = np.mean(et_sfs)
Rns2_ET_sfs = (1 - np.sum(np.square(et_sfs -
          np.mean(AET_mod_NFS[:, sfs_cells][mod_ids_sfs], axis=1)))/np.sum(np.square(
          et_sfs - mq_et_sfs)))
#print('Rns2 for Evapotranspiration: ', Rns2_ET_sfs)

## Calculate Nash-Sutcliffe for Soil Moisture
# NFS
SM_mod_NFS_target = SM_mod_NFS[:, target_NFS]
B_nfs = np.logical_and(sm_obs_nfs[:]>0., sm_obs_nfs[:]<1.)
obs_S_locs_nfs = np.where(B_nfs)[0]  # locations where ObsS is valid
mq_s_nfs = np.mean(sm_obs_nfs[obs_S_locs_nfs])
Rns2_S_nfs = (1 - np.sum(np.square(sm_obs_nfs[obs_S_locs_nfs] -
          SM_mod_NFS_target[obs_S_locs_nfs]))/np.sum(np.square(
          sm_obs_nfs[obs_S_locs_nfs] - mq_s_nfs)))
# SFS
SM_mod_SFS_target = SM_mod_NFS[:, target_SFS]
B = np.logical_and(sm_obs_sfs[:]>0., sm_obs_sfs[:]<1.)
obs_S_locs_sfs = np.where(B)[0]  # locations where ObsS is valid
mq_s_sfs = np.mean(sm_obs_sfs[obs_S_locs_sfs])
Rns2_S_sfs = (1 - np.sum(np.square(sm_obs_sfs[obs_S_locs_sfs] -
          SM_mod_SFS_target[obs_S_locs_sfs]))/np.sum(np.square(
          sm_obs_sfs[obs_S_locs_sfs] - mq_s_sfs)))

## Drainage
drainage_area = grid2.at_node['drainage_area'][grid.node_at_cell]
draining_cells = drainage_area/grid.cellarea + 1
sp_mean_runoffOp = np.zeros(yrs)
sp_max_runoffOp = np.zeros(yrs)
for i in range(0, yrs):
    sp_mean_runoffOp[i] = (
            np.mean(runoff_annual[i,:]/(draining_cells * P_annual[i])))
    sp_max_runoffOp[i] = (
            np.max(runoff_annual[i,:]/(draining_cells * P_annual[i])))


## Plotting

## Plot 1:1 Mod S vs Obs S
plt.figure(figsize=(10, 8))
plt.scatter(np.mean(AET_mod_NFS[:, nfs_cells][mod_ids_nfs], axis=1), et_nfs)
plt.plot(np.arange(0, 4.1, 0.1), np.arange(0, 4.1, 0.1), 'k')
plt.xlim(xmin=0, xmax=4)
plt.ylim(ymin=0, ymax=4)
plt.xlabel('Modeled ET (mm/d)')
plt.ylabel('Obs ET (mm/d)')
txt_in = 'Rns2_ET_nfs = '+str(round(Rns2_ET_nfs, 2))
plt.text(2.5, 3.5, txt_in, bbox=dict(facecolor='none', alpha=0.5))
plt.legend('NFS')
plt.savefig('1_1_plot_ET_nfs')

plt.figure(figsize=(10, 8))
plt.scatter(np.mean(AET_mod_NFS[:, sfs_cells][mod_ids_sfs], axis=1), et_sfs)
plt.plot(np.arange(0, 4.1, 0.1), np.arange(0, 4.1, 0.1), 'k')
plt.xlim(xmin=0, xmax=4)
plt.ylim(ymin=0, ymax=4)
plt.xlabel('Modeled ET (mm/d)')
plt.ylabel('Obs ET (mm/d)')
txt_in = 'Rns2_ET_sfs = '+str(round(Rns2_ET_sfs, 2))
plt.text(2.5, 3.5, txt_in, bbox=dict(facecolor='none', alpha=0.5))
plt.legend('SFS')
plt.savefig('1_1_plot_ET_sfs')

## For observed Radiation model
days = range(0,days_n)

## ET plots
# NFS
fig, ax1 = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(10)
label_prop = {'fontsize': 12, 'weight': 'bold'}
legend_prop = {'size': 12, 'weight': 'bold'}
linewidth = 3
markersize = 10
ax1.plot(strm_days, np.mean(AET_mod_NFS[:, nfs_cells], axis=1),
         'k', label='Mod NFS ET',
         linewidth=linewidth)
ax1.plot(date_et_nfs, et_nfs, 'ro', label = 'Obs NFS ET',
                markersize=markersize, markerfacecolor='none')
plt.gcf().autofmt_xdate()
ax1.set_xlabel('Day', **label_prop)
ax1.set_ylabel('Evapotranspiration (mm/d)', **label_prop)
#ax1.set_title('Soil Moisture at NFS - Obs vs Mod')
ax1.set_xlim(left=date_et_nfs[0], right=strm_days[-1])
ax1.set_ylim(bottom=0)
ax1.legend(loc='upper right', prop=legend_prop)
plt.savefig('act_evapotranspiration_nfs.png')

# SFS
fig, ax1 = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(10)
label_prop = {'fontsize': 12, 'weight': 'bold'}
legend_prop = {'size': 12, 'weight': 'bold'}
linewidth = 3
markersize = 10
ax1.plot(strm_days, np.mean(AET_mod_NFS[:, sfs_cells], axis=1),
         'k', label='Mod SFS ET',
         linewidth=linewidth)
ax1.plot(date_et_sfs, et_sfs, 'ro', label = 'Obs SFS ET',
                markersize=markersize, markerfacecolor='none')
plt.gcf().autofmt_xdate()
ax1.set_xlabel('Day', **label_prop)
ax1.set_ylabel('Evapotranspiration (mm/d)', **label_prop)
#ax1.set_title('Soil Moisture at NFS - Obs vs Mod')
ax1.set_xlim(left=date_et_sfs[0], right=strm_days[-1])
ax1.set_ylim(bottom=0)
ax1.legend(loc='upper right', prop=legend_prop)
plt.savefig('act_evapotranspiration_sfs.png')

# plotting NFS cells
plt.figure(figsize=(10, 8))
imshow_grid(grid, values=aspect, values_at='cell', cmap='jet')
plt.plot(grid.x_of_cell[nfs_cells], grid.y_of_cell[nfs_cells],
         'xk', label='NFS cells')
plt.plot(grid.x_of_cell[sfs_cells], grid.y_of_cell[sfs_cells],
         'xr', label='SFS cells')
plt.savefig('sampled_locations_for_ET')

# AET over P
plt.figure(figsize=(10, 8))
plt.plot(np.mean(AET_over_P, axis=1), '--*r')
plt.xlim([0, yrs-1])
plt.ylim([0, 1.1])
plt.xlabel('Years')
plt.ylabel('Annual ET/P')
plt.title('Watershed average annual ET/P')
plt.savefig('annual_AET_over_P_wtrshd.png')

# AET and PET
fig, (ax3, ax1) = plt.subplots(nrows=2, ncols=1)
fig.set_figheight(8)
fig.set_figwidth(10)
label_prop = {'fontsize': 12, 'weight': 'bold'}
legend_prop = {'size': 12, 'weight': 'bold'}
linewidth = 3
markersize = 10
ax1.plot(strm_days, np.mean(AET_mod_NFS[:, nfs_cells], axis=1),
         'k', label='Mod NFS ET',
         linewidth=linewidth)
ax1.plot(date_et_nfs, et_nfs, '--ro', label = 'Obs NFS ET',
                markersize=markersize, markerfacecolor='none')
plt.gcf().autofmt_xdate()
ax1.set_xlabel('Day', **label_prop)
ax1.set_ylabel('Evapotranspiration (mm/d)', **label_prop)
#ax1.set_title('Soil Moisture at NFS - Obs vs Mod')
ax1.set_xlim(left=date_et_nfs[0], right=strm_days[-1])
ax1.set_ylim(bottom=0)
ax1.legend(loc=7, prop=legend_prop)

ax2 = ax1.twinx()
ax2.bar(strm_days, precip, color='b', width=2)
ax1.set_xlim(left=date_et_nfs[0], right=strm_days[-1])
ax2.set_ylim(bottom=0, top=50)
ax2.invert_yaxis()
ax2.set_ylabel('Preciptation (mm)', **label_prop)
fig.tight_layout()

ax3.plot(strm_days, np.mean(PET_mod_NFS[:, nfs_cells], axis=1), 'b', label='Mod NFS PET', linewidth=linewidth)
plt.gcf().autofmt_xdate()
ax3.set_xlabel('Day', **label_prop)
ax3.set_ylabel('Evapotranspiration (mm/d)', **label_prop)
ax3.set_xlim(left=date_et_nfs[0], right=strm_days[-1])
ax3.set_ylim(bottom=0)
ax3.legend(loc=0, prop=legend_prop)
plt.savefig('act_pet_evapotranspiration_nfs.png', bbox_inches = 'tight')


fig, (ax3, ax1) = plt.subplots(nrows=2, ncols=1)
fig.set_figheight(8)
fig.set_figwidth(10)
label_prop = {'fontsize': 12, 'weight': 'bold'}
legend_prop = {'size': 12, 'weight': 'bold'}
linewidth = 3
markersize = 10
ax1.plot(strm_days, np.mean(AET_mod_NFS[:, sfs_cells], axis=1),
         'k', label='Mod SFS ET',
         linewidth=linewidth)
ax1.plot(date_et_sfs, et_sfs, '--ro', label = 'Obs SFS ET',
                markersize=markersize, markerfacecolor='none')
plt.gcf().autofmt_xdate()
ax1.set_xlabel('Day', **label_prop)
ax1.set_ylabel('Evapotranspiration (mm/d)', **label_prop)
ax1.set_xlim(left=date_et_sfs[0], right=strm_days[-1])
ax1.set_ylim(bottom=0)
ax1.legend(loc=7, prop=legend_prop)

ax2 = ax1.twinx()
ax2.bar(strm_days, precip, color='b', width=2)
ax1.set_xlim(left=date_et_sfs[0], right=strm_days[-1])
ax2.set_ylim(bottom=0, top=50)
ax2.invert_yaxis()
ax2.set_ylabel('Preciptation (mm)', **label_prop)
fig.tight_layout()

ax3.plot(strm_days, np.mean(PET_mod_NFS[:, sfs_cells], axis=1), 'b',
         label='Mod SFS PET', linewidth=linewidth)
plt.gcf().autofmt_xdate()
ax3.set_xlabel('Day', **label_prop)
ax3.set_ylabel('Evapotranspiration (mm/d)', **label_prop)
ax3.set_xlim(left=date_et_sfs[0], right=strm_days[-1])
ax3.set_ylim(bottom=0)
ax3.legend(loc=0, prop=legend_prop)
plt.savefig('act_pet_evapotranspiration_sfs.png', bbox_inches = 'tight')

# Plot Max Annual Drainage Ratio
plt.figure(figsize=(10, 8))
plt.plot(range(0, yrs), sp_max_runoffOp, '--*k', linewidth=3)
plt.xlim(xmin=0, xmax=yrs-1)
plt.ylim(ymin=0)
plt.xlabel('Years')
plt.ylabel('Max Annual Runoff Ratio (-)')
plt.savefig('max_annual_runoff_ratio.png')

plt.show()

metadata = 'Outputs_entire watershed simulations 22Mar18'

data_stored = {'sub_fldr_name': sub_fldr_name,
               'metadata': metadata,
               'Rns_et_nfs': Rns2_ET_nfs,
               'AET_tot_nfs': np.sum(np.mean(AET_mod_NFS[:, nfs_cells][mod_ids_nfs], axis=1)),
               'PET_tot_nfs': np.sum(np.mean(PET_mod_NFS[:, nfs_cells][mod_ids_nfs], axis=1)),
               'et_nfs_tot': np.sum(et_nfs),
               'precip_nfs': np.sum(precip[mod_ids_nfs]),
               'Rns_S_nfs': Rns2_S_nfs,
               'Rns_et_sfs': Rns2_ET_sfs,
               'AET_tot_sfs': np.sum(np.mean(AET_mod_NFS[:, sfs_cells][mod_ids_sfs], axis=1)),
               'PET_tot_sfs': np.sum(np.mean(PET_mod_NFS[:, sfs_cells][mod_ids_sfs], axis=1)),
               'et_sfs_tot': np.sum(et_sfs),
               'precip_sfs': np.sum(precip[mod_ids_sfs]),
               'Rns_S_sfs': Rns2_S_sfs,
               'AET_over_P': AET_over_P,
               'drain_over_P': drain_over_P,
               'sp_mean_runoffOp': sp_mean_runoffOp,
               'sp_max_runoffOp': sp_max_runoffOp,
               }
pickle.dump(data_stored, open("outputs_sim.p", "wb"))

if 1:
    print('Rns_et_nfs = ', Rns2_ET_nfs)
    print('AET_tot_nfs = ', np.sum(np.mean(AET_mod_NFS[:, nfs_cells][mod_ids_nfs], axis=1)))
    print('PET_tot_nfs = ', np.sum(np.mean(PET_mod_NFS[:, nfs_cells][mod_ids_nfs], axis=1)))
    print('et_nfs_tot = ', np.sum(et_nfs))
    print('precip_nfs = ', np.sum(precip[mod_ids_nfs]))
    print('Rns_S_nfs = ', Rns2_S_nfs)
    
    print('Rns_et_sfs = ', Rns2_ET_sfs)
    print('AET_tot_sfs = ', np.sum(np.mean(AET_mod_NFS[:, sfs_cells][mod_ids_sfs], axis=1)))
    print('PET_tot_sfs = ', np.sum(np.mean(PET_mod_NFS[:, sfs_cells][mod_ids_sfs], axis=1)))
    print('et_sfs_tot = ', np.sum(et_sfs))
    print('precip_sfs = ', np.sum(precip[mod_ids_sfs]))
    print('Rns_S_sfs = ', Rns2_S_sfs)