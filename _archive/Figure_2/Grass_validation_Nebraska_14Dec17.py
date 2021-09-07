
# Authors: Sai Nudurupati & Erkan Istanbulluoglu, 12Dec14


import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from landlab.io import read_esri_ascii
from landlab import RasterModelGrid as rmg
from landlab import load_params
#from landlab.components import PrecipitationDistribution
from landlab.components import Radiation
from landlab.components import PotentialEvapotranspiration
from landlab.components import SoilMoisture
from landlab.components import Vegetation
from landlab.components import VegCA
import matplotlib as mpl
#from matplotlib.axes import *
import time

# GRASS = 0; SHRUB = 1; TREE = 2; BARE = 3;
# SHRUBSEEDLING = 4; TREESEEDLING = 5


## Point to the input DEM
_DEFAULT_INPUT_FILE_1 = os.path.join(os.path.dirname(__file__), 'DEM_10m.asc')

data = load_params('trial_17.yaml') # Create dictionary that holds the inputs

# Name of the folder you want to store the outputs
sub_fldr_name = 'trial_17_copy_1'

## Importing Grid and Elevations from DEM
(grid1,elevation) = read_esri_ascii(_DEFAULT_INPUT_FILE_1)
grid1['node']['topographic__elevation'] = elevation
grid = rmg(5,4,5)
grid['node']['topographic__elevation'] = 1700. * np.ones(grid.number_of_nodes)
grid1['cell']['vegetation__plant_functional_type'] = (
        np.random.randint(0,6,grid1.number_of_cells))
grid['cell']['vegetation__plant_functional_type'] = np.arange(0,6)

mat_data = scipy.io.loadmat('Sailow.mat')
ObsYearsB = mat_data['YearsB']   # Observed LAI low years
ObsLAIlow = mat_data['LAIlow']   # Observed LAI low
ObsMODIS_Years = mat_data['MODIS_Years'] # Observed MODIS years
ObsMODIS_LAI = mat_data['MODIS_LAI']   # Observed MODIS LAI
ObsT = mat_data['T']     # years for observed biomass
ObsBioMlow = mat_data['BioMlow']  # Observed live Biomass low
ObsDeadBlow = mat_data['DeadBlow'] # Observed dead Biomass low
ObsS = mat_data['ObsS']
BGMBiol = mat_data['BGMBiol']
BGMBiot = mat_data['BGMBiot']

PClim = mat_data['P']
TbClim = mat_data['Tb']
TrClim = mat_data['Tr']
PET_grass = mat_data['Ep']   # Penman Monteith data from file

# Create radiation, soil moisture and Vegetation objects
Rad = Radiation(grid)
PET_Tree = PotentialEvapotranspiration(grid1, method=data['PET_method'],
                                       MeanTmaxF=data['MeanTmaxF_tree'],
                                       DeltaD=data['DeltaD'])
PET_Shrub = PotentialEvapotranspiration(grid1, method=data['PET_method'],
                                        MeanTmaxF=data['MeanTmaxF_shrub'],
                                        DeltaD=data['DeltaD'])
PET_Grass = PotentialEvapotranspiration(grid1, method = data['PET_method'],
                                        MeanTmaxF=data['MeanTmaxF_grass'],
                                        DeltaD=data['DeltaD'])

SM = SoilMoisture(grid, runon_switch=0, **data)   # Soil Moisture object
VEG = Vegetation(grid, **data)    # Vegetation object
vegca = VegCA(grid1, **data)      # Cellular automaton object

##########
n = 3275 #data['n_short']   # Defining number of storms the model will be run
##########

## Create arrays to store modeled data
P = np.empty(n)    # Record precipitation
Tb = np.empty(n)    # Record inter storm duration
Tr = np.empty(n)    # Record storm duration
Time = np.empty(n) # To record time elapsed from the start of simulation

CumWaterStress = np.empty([n/50, grid1.number_of_cells]) # Cum Water Stress
CumWS = np.empty([n/50, grid.number_of_cells]) # Cum Water Stress of different plant types
VegType = np.empty([n/50, grid1.number_of_cells],dtype = int)
PETn_ = np.empty([n, grid.number_of_cells])
EP30_ = np.empty([n, grid.number_of_cells])
AET_ = np.empty([n, grid.number_of_cells])
SM_ = np.empty([n, grid.number_of_cells])
SMini_ = np.empty([n, grid.number_of_cells])
SMavg_ = np.empty([n, grid.number_of_cells])
LAI = np.empty([n, grid.number_of_cells])
Blive = np.empty([n, grid.number_of_cells])
Btotal = np.empty([n, grid.number_of_cells])
WaterStress = np.empty([n, grid.number_of_cells])
ETmax = np.empty([n, grid.number_of_cells])
LAI_fr = np.empty([n, grid.number_of_cells])
veg_cov = np.empty([n, grid.number_of_cells])
MAP = np.zeros(n)
Julian_ = np.empty(n)

PET_ = np.zeros([n, grid.number_of_cells])
Rad_Factor = np.empty([365, grid.number_of_cells])
EP30 = np.empty([n, grid.number_of_cells]) # 30 day average PET to determine season
PET_threshold = 0  # Initializing PET_threshold to ETThresholddown

## Initializing inputs for Soil Moisture object
grid['cell']['vegetation__live_leaf_area_index'] = (1.6 *
                np.ones(grid.number_of_cells))
SM._SO = (0.12 * np.ones(grid.number_of_cells)) # Initializing Soil Moisture
Tg = 365.    # Growing season in days


## Calculate current time in years such that Julian time can be calculated
current_time = 0                   # Start from first day of June

for i in range(0,n):
    PET_[i] = (PET_grass[i] * np.ones(grid.number_of_cells))
    if i < 30:
        if i == 0:
            EP30[0] = PET_[0]
        else:
            EP30[i] = np.mean(PET_[:i],axis = 0)
    else:
        EP30[i] = np.mean(PET_[i-30:i], axis = 0)


time_check = 0.               # Buffer to store current_time at previous storm
i_check = 0                   #
yrs = 0

Start_time = time.clock()     # Recording time taken for simulation
WS = 0.

## Run Time Loop
for i in range(0, n):
    P[i] = PClim[i]
    Tr[i] = TrClim[i]
    Tb[i] = TbClim[i]
    grid.at_cell['rainfall__daily_depth'] = (P[i] *
                                             np.ones(grid.number_of_cells))
    grid['cell']['surface__potential_evapotranspiration_rate'] = PET_[i]
    grid['cell']['surface__potential_evapotranspiration_30day_mean'] = EP30[i]
    grid.at_cell['surface__potential_evapotranspiration_rate__grass'] = PET_[i]
    current_time = SM.update(current_time, Tr = Tr[i], Tb = Tb[i])
    PETn_[i] = SM._PET

    if i != n-1:
        if EP30[i + 1,0] > EP30[i,0]:
            PET_threshold = 1  # 1 corresponds to ETThresholdup
        else:
            PET_threshold = 0  # 0 corresponds to ETThresholddown

    VEG.update(PETthreshold_switch=PET_threshold, Tr=Tr[i], Tb=Tb[i])

    WS += (grid['cell']['vegetation__water_stress'])*Tb[i]/24.
    PETn_[i] = grid['cell']['surface__potential_evapotranspiration_rate']
    AET_[i] = grid['cell']['surface__evapotranspiration']
    SM_[i] = grid['cell']['soil_moisture__saturation_fraction']
    SMini_[i] = SM._Sini
    SMavg_[i] = (SMini_[i]+SM_[i])/2.
    Time[i] = current_time
    WaterStress[i] = grid['cell']['vegetation__water_stress']
    MAP[yrs] += P[i]
    #Julian_[i] = Julian
    LAI[i] = grid['cell']['vegetation__live_leaf_area_index']
    Blive[i] = grid['cell']['vegetation__live_biomass']
    Btotal[i] = Blive[i]+grid['cell']['vegetation__dead_biomass']
    veg_cov[i] = grid.at_cell['vegetation__cover_fraction']
    ETmax[i] = SM._ETmax
    LAI_fr[i] = SM._fr

    # Cellular Automata
    if (current_time - time_check) >= 1.:
        VegType[yrs] = grid1['cell']['vegetation__plant_functional_type']
        WS_ = np.choose(VegType[yrs],WS)
        CumWaterStress[yrs] = WS_/Tg
        grid1['cell']['vegetation__cumulative_water_stress'] = (
                CumWaterStress[yrs])
        vegca.update()
        CumWS[yrs] = WS/Tg   # Record of Cumulative Water Stress of different plant types
        time_check = current_time
        WS = 0
        i_check = i
        yrs += 1


Final_time = time.clock()

VegType[yrs] = grid1['cell']['vegetation__plant_functional_type']
Time_Consumed = ((Final_time - Start_time)/60.)    # in minutes

## Evaluating Cumulative WaterStress of each plant functional type
CWS = CumWS.round(decimals=2)
# Separate arrays according to their PFTs
CWS_G = CWS[0:yrs,0]
CWS_S = CWS[0:yrs,1]
CWS_T = CWS[0:yrs,2]
CWS_SS = CWS[0:yrs,4]
CWS_TS = CWS[0:yrs,5]
# Sort them in ascending order
CWS_G.sort()
CWS_S.sort()
CWS_T.sort()
CWS_SS.sort()
CWS_TS.sort()
# Weibull probability of exceedance (Ref. Erkan's class notes - Hw1 Physical Hydrology)
Q_G = [(1. - (m/(yrs+1.))) for m in range(1,len(CWS_G)+1)]
Q_S = [(1. - (m/(yrs+1.))) for m in range(1,len(CWS_S)+1)]
Q_T = [(1. - (m/(yrs+1.))) for m in range(1,len(CWS_T)+1)]
Q_SS = [(1. - (m/(yrs+1.))) for m in range(1,len(CWS_SS)+1)]
Q_TS = [(1. - (m/(yrs+1.))) for m in range(1,len(CWS_TS)+1)]


## Plotting
cmap = mpl.colors.ListedColormap(   \
                    [ 'green', 'red', 'black', 'white', 'red', 'black' ] )
bounds = [-0.5,0.5,1.5,2.5,3.5,4.5,5.5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


sim = 'Grass_validation_NE_15Dec17'

try:
    os.chdir('E:/Research_UW_Sai_PhD/Jan_2017/Landlab_re_validation_with_Xiaochi13_25Feb17/Match_inputs_with_BGM_paper')
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

## Final plot for NSH Validation - marked on 29Dec17

fig, (ax1, ax3, ax4) = plt.subplots(3)
fig.set_figheight(12)
fig.set_figwidth(10)
ax1.plot(ObsT, SMavg_[:, 0],'-k', label = 'Modeled S', linewidth=2)
ax1.plot(ObsT[1917:2828], ObsS[1917:2828, 0], 'ro', markerfacecolor='none',
         label = 'Observed S')
#ax1.set_xlabel('Day', fontsize=14, weight='bold')
ax1.set_ylabel('Saturation Fraction', fontsize=14, weight='bold')
#ax1.set_title('Soil Moisture - Obs vs Mod', weight='bold', fontsize=14)
ax1.set_xlim(left=ObsT[0], right=ObsT[3274])
ax1.set_ylim(bottom=0.07, top=1.)
legend_properties = {'weight': 'bold'}
ax1.legend(fontsize=10, bbox_to_anchor=(0.78, 0.64),
    prop=legend_properties)

ax2 = ax1.twinx()
ax2.bar(ObsT, PClim[:], color='b', width=0.03)
ax2.set_xlim(left=ObsT[0], right=ObsT[3274])
ax2.set_ylim(bottom=0, top=150)
ax2.invert_yaxis()
ax2.set_ylabel('Preciptation (mm)', fontsize=14, weight='bold')
fig.tight_layout()


ax3.plot(ObsT, LAI[:,0], label='Modeled live LAI', linewidth=2)
#plt.hold(True)
ax3.plot(ObsYearsB, ObsLAIlow, 'r^', label='Observed live LAI')
#plt.hold(True)
ax3.plot(ObsMODIS_Years, ObsMODIS_LAI, 'go', label='MODIS live LAI')
ax3.set_xlim(left=ObsT[0], right=ObsT[3274])
#locs,labels = plt.xticks()
#ax3.xticks(locs, map(lambda x: "%g" % x, locs))
ax3.set_ylabel('LAI', {'weight':'bold','size':14})
ax3.legend(loc=2, fontsize=10, ncol=3, prop=legend_properties)
#plt.title( 'Live-LAI',{'weight':'bold','size':14})

ax4.plot(ObsT, Blive[:, 0], label='Modeled live Biomass', linewidth=2)
#plt.hold(True)
ax4.plot(ObsYearsB, ObsBioMlow, 'g^', label='Observed live Biomass')
#plt.hold(True)
ax4.plot(ObsT, Btotal[:, 0], 'g', label='Modeled total Biomass', linewidth=2)
#locs,labels = plt.xticks()
#ax4.xticks(locs, map(lambda x: "%g" % x, locs))
#plt.hold(True)
ax4.plot(ObsYearsB, ObsBioMlow+ObsDeadBlow, 'ro',
         linewidth=2, label='Observed Total Biomass')
ax4.set_xlim(left=ObsT[0], right=ObsT[3274])
ax4.set_ylim([0.0, 1.15*max(Btotal[:, 0])])
ax4.set_ylabel('Biomass', {'weight':'bold','size':14})
ax4.legend(ncol=2, loc=2, fontsize=10, prop=legend_properties)

fontdict = {'fontsize': 14, 'weight': 'heavy', 'bbox': {'facecolor': 'none'}}
fig.text(0.01, 0.68, 'a)', fontdict=fontdict)
fig.text(0.01, 0.34, 'b)', fontdict=fontdict)
fig.text(0.01, 0.01, 'c)', fontdict=fontdict)
plt.savefig('Nebraska_calibration_final_figure')


## Code to calculate Nash-Sutcliffe coefficient for S, LAI and Biomasslive
# Ref: Dingman 2nd edition - Pg. 580
# isolate observations that are not zero and from a particular year
# bcoz mq should be calculated separately for each year
B = np.logical_and(ObsS[:, 0]>0., ObsS[:, 0]<1.)
obs_S_locs = np.where(B==True)[0]  # locations where ObsS is valid
mq_s = np.mean(ObsS[obs_S_locs, 0])
Rns2_S = (1 - np.sum(np.square(ObsS[obs_S_locs, 0] - SMavg_[obs_S_locs, 0]))/
        np.sum(np.square(ObsS[obs_S_locs, 0] - mq_s)))
print('Rns2 for Soil Moisture Saturation Fraction: ', Rns2_S)

## Calculate Rns2_mean_LAI
ObsT_d = np.round(ObsT*365.25, decimals=0)
ObsYearsB_d = np.round(ObsYearsB*365.25, decimals=0)
obs_L_locs = np.where(np.in1d(ObsT_d, ObsYearsB_d)==True)[0]
mq_l = np.mean(ObsLAIlow[:, 0])
mq_b = np.mean(ObsBioMlow[:, 0])
Rns2_L = (1 - np.sum(np.square(ObsLAIlow[:, 0] - LAI[obs_L_locs, 0]))/
        np.sum(np.square(ObsLAIlow[:, 0] - mq_l)))
Rns2_B = (1 - np.sum(np.square(ObsBioMlow[:, 0] - Blive[obs_L_locs, 0]))/
        np.sum(np.square(ObsBioMlow[:, 0] - mq_b)))
print('Rns2 for leaf Area Index: ', Rns2_L)
print('Rns2 for Live Biomass: ', Rns2_B)

