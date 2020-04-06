
# Authors: Sai Nudurupati & Erkan Istanbulluoglu, 21May15

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from landlab.plot import imshow_grid
from landlab.components import PrecipitationDistribution
from landlab.components import Radiation
from landlab.components import PotentialEvapotranspiration
from landlab.components import SoilMoisture
from landlab.components import Vegetation
from landlab.components import VegCA

GRASS = 0
SHRUB = 1
TREE = 2
BARE = 3
SHRUBSEEDLING = 4
TREESEEDLING = 5


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


def initialize(data, grid, sm_grid, rad_grid, elevation):
    """Initialize random plant type field.

    Plant types are defined as the following:

    *  GRASS = 0
    *  SHRUB = 1
    *  TREE = 2
    *  BARE = 3
    *  SHRUBSEEDLING = 4
    *  TREESEEDLING = 5
    """
    grid.at_cell['vegetation__plant_functional_type'] = compose_veg_grid(
        grid, percent_bare=data['percent_bare_initial'],
        percent_grass=data['percent_grass_initial'],
        percent_shrub=data['percent_shrub_initial'],
        percent_tree=data['percent_tree_initial'])
    # Assign plant type for representative ecohydrologic simulations
    sm_grid.at_cell['vegetation__plant_functional_type'] = np.repeat(
                                                            np.arange(6), 108)
    sm_grid.at_node['topographic__elevation'] = np.full(
                                    sm_grid.number_of_nodes, 1700.)
    grid.at_node['topographic__elevation'] = elevation
    rad_grid.at_node['topographic__elevation'] = np.full(
                                            rad_grid.number_of_nodes, 1700.)    
    precip_dry = PrecipitationDistribution(
        mean_storm_duration=data['mean_storm_dry'],
        mean_interstorm_duration=data['mean_interstorm_dry'],
        mean_storm_depth=data['mean_storm_depth_dry'],
        random_seed=None)
    precip_wet = PrecipitationDistribution(
        mean_storm_duration=data['mean_storm_wet'],
        mean_interstorm_duration=data['mean_interstorm_wet'],
        mean_storm_depth=data['mean_storm_depth_wet'],
        random_seed=None)
    radiation = Radiation(rad_grid)
    # set slope and aspect to each bin
    set_slopes_aspects(radiation)
    pet_tree = PotentialEvapotranspiration(sm_grid, method=data['PET_method'],
                                           MeanTmaxF=data['MeanTmaxF_tree'],
                                           delta_d=data['DeltaD'])
    pet_shrub = PotentialEvapotranspiration(sm_grid, method=data['PET_method'],
                                            MeanTmaxF=data['MeanTmaxF_shrub'],
                                            delta_d=data['DeltaD'])
    pet_grass = PotentialEvapotranspiration(sm_grid, method=data['PET_method'],
                                            MeanTmaxF=data['MeanTmaxF_grass'],
                                            delta_d=data['DeltaD'])
    soil_moisture = SoilMoisture(sm_grid, **data)   # Soil Moisture object
    vegetation = Vegetation(sm_grid, **data)    # Vegetation object
    vegca = VegCA(grid, **data)      # Cellular automaton object

    # # Initializing inputs for Soil Moisture object
    sm_grid.at_cell['vegetation__live_leaf_area_index'] = (
        1.6 * np.ones(sm_grid.number_of_cells))
    sm_grid.at_cell['soil_moisture__initial_saturation_fraction'] = (
        0.59 * np.ones(sm_grid.number_of_cells))
    # Initializing Soil Moisture
    return (precip_dry, precip_wet, radiation, pet_tree, pet_shrub,
            pet_grass, soil_moisture, vegetation, vegca)


def empty_arrays(n, n_years, grid, sm_grid, rad_grid):
    precip = np.empty(n)    # Record precipitation
    inter_storm_dt = np.empty(n)    # Record inter storm duration
    storm_dt = np.empty(n)    # Record storm duration
#    CumWaterStress = np.empty(grid.number_of_cells)  # Cum Water Stress
    daily_pet = np.zeros([365, 6])
    rad_Factor = np.empty([365, sm_grid.number_of_cells])
    EP30 = np.empty([365, 6])
    # 30 day average PET to determine season
    pet_threshold = 0  # Initializing daily_pet threshold to ETThresholddown
    return (precip, inter_storm_dt, storm_dt,
            daily_pet, rad_Factor, EP30, pet_threshold)


def create_pet_lookup(radiation, pet_tree, pet_shrub, pet_grass, daily_pet,
                      rad_factor, EP30, rad_grid):
    for i in range(0, 365):
#        rad_pet.update(float(i)/365.25)
        pet_tree.update(float(i)/365.25)
        pet_shrub.update(float(i)/365.25)
        pet_grass.update(float(i)/365.25)
        daily_pet[i] = [pet_grass._PET_value, pet_shrub._PET_value,
                   pet_tree._PET_value, 0., pet_shrub._PET_value,
                   pet_tree._PET_value]
        radiation.update(float(i)/365.25)
        rad_factor[i] = np.tile(
                rad_grid.at_cell['radiation__ratio_to_flat_surface'], 6)
        if i < 30:
            if i == 0:
                EP30[0] = daily_pet[0]
            else:
                EP30[i] = np.mean(daily_pet[:i], axis=0)
        else:
            EP30[i] = np.mean(daily_pet[i-30:i], axis=0)
    return (EP30, daily_pet, rad_factor)

def save(sim, inter_storm_dt, storm_dt, precip, veg_type, yrs, walltime,
         time_elapsed):
    np.save(sim+'inter_storm_dt', inter_storm_dt)
    np.save(sim+'storm_dt', storm_dt)
    np.save(sim+'P', precip)
    np.save(sim+'veg_type', veg_type)
#    np.save(sim+'CumWaterStress', CumWaterStress)
    np.save(sim+'Years', yrs)
    np.save(sim+'Time_Consumed_minutes', walltime)
    np.save(sim+'CurrentTime', time_elapsed)


def plot(sim, grid, veg_type, yrs, yr_step=10):
    elevation = np.copy(grid.at_node['topographic__elevation'])
    elevation[elevation<0.] = elevation.max()
    elev_raster = grid.node_vector_to_raster(elevation)
    elev_grid = np.zeros([elev_raster.shape[0]-2, elev_raster.shape[1]-2])
    xx = (grid.dx * (np.arange(elev_grid.shape[1])))
    yy = (grid.dy * (np.arange(elev_grid.shape[0])))
    for ii in range(1, int(elev_raster.shape[0]-1)):
        for jj in range(1, int(elev_raster.shape[1]-1)):
            elev_grid[ii-1][jj-1] = elev_raster[ii][jj]

    pic = 0
    years = range(0, yrs)
    cmap = mpl.colors.ListedColormap(
                        ['green', 'red', 'black', 'white', 'red', 'black'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    print 'Plotting cellular field of Plant Functional Type'
    print 'Green - Grass; Red - Shrubs; Black - Trees; White - Bare'
    # # Plot images to make gif.
    for year in range(0, yrs, yr_step):
        filename = 'year_' + "%05d" % year
        pic += 1
        plt.figure(pic, figsize=(10, 8))
        imshow_grid(grid, veg_type[year], values_at='cell', cmap=cmap,
                    grid_units=('m', 'm'), norm=norm, limits=[0, 5],
                    allow_colorbar=False)
        clt = plt.contour(xx, yy, elev_grid, colors='b',
                          figsize=(10, 8), linewidths=3)
        plt.clabel(clt, inline=1, fmt='%4.0f', fontsize=12)
        plt.title(filename, weight='bold', fontsize=14)
        plt.savefig(sim+filename)

    grass_cov = np.empty(yrs)
    shrub_cov = np.empty(yrs)
    tree_cov = np.empty(yrs)
    shseed_cov = np.empty(yrs)
    trseed_cov = np.empty(yrs)
    grid_size = float(veg_type.shape[1])

    for x in range(0, yrs):
        grass_cov[x] = ((veg_type[x][veg_type[x] == GRASS].size / grid_size) *
                         100)
        shrub_cov[x] = ((veg_type[x][veg_type[x] == SHRUB].size / grid_size) *
                        100)
        shseed_cov[x] = ((veg_type[x][veg_type[x] == SHRUBSEEDLING].size /
                        grid_size) * 100)
        tree_cov[x] = ((veg_type[x][veg_type[x] == TREE].size / grid_size) *
                       100)
        trseed_cov[x] = ((veg_type[x][veg_type[x] == TREESEEDLING].size /
                       grid_size) * 100)

    pic += 1
    plt.figure(pic, figsize=(10, 8))
    plt.plot(years, grass_cov, '-g', label='Grass', linewidth=4)
    plt.plot(years, shrub_cov, '-r', label='Shrub', linewidth=4)
    plt.plot(years, tree_cov, '-k', label='Tree', linewidth=4)
    plt.plot(years, shseed_cov, '-m', label='ShSeed', linewidth=4)
    plt.plot(years, trseed_cov, '-b', label='TrSeed', linewidth=4)
    plt.ylabel('% Area Covered by Plant Type', weight='bold', fontsize=18)
    plt.xlabel('Time in years', weight='bold', fontsize=18)
    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')
    plt.legend(loc=0, prop={'size': 16, 'weight': 'bold'})
    plt.savefig(sim+'_percent_cover')
    # plt.show()


def set_slopes_aspects(radiation):
    """ This function will set up bins with combinations of
    slopes (0-45 deg in 5 deg increments) and aspects (0-360 in 30 deg
    increments). Final slopes and aspects (in radians) will be embedded into
    radiation object as radiation._slope and radiation._aspect fields.
    """
    slp_bins_deg = np.arange(0, 45, 5)
    asp_bins_deg = np.arange(0, 360, 30)
    slp_deg_r = slp_bins_deg.repeat(12)
    asp_deg_r = np.tile(asp_bins_deg, 9)
    radiation._slope = np.radians(slp_deg_r)
    radiation._aspect = np.radians(asp_deg_r)


def get_slp_asp_mapping(grid):
    """This function will map each cell in the *grid* to the
    slope_aspect bin it belongs to. The bins are identified by an
    id ranging from 0 to 107 (108 bins). An id will be assigned to
    each cell in the watershed. Tested for 'sev10m_nad83.txt'
    """
    (slope, aspect) = (
           grid.calculate_slope_aspect_at_nodes_burrough(vals='topographic__elevation'))
    slope[slope > 1.4] = 0.     # mannualy edit walls (tested for sev10m_nad83)
#    core_cells = grid.core_cells
#    neg_asp_cr = np.where(aspect[core_cells]<0.)[0]
    neg_asp_cr = np.where(aspect<0.)[0]
#    aspect[core_cells[neg_asp_cr]] += (2*np.pi)
    aspect[neg_asp_cr] += (2*np.pi)
    slp_deg = np.degrees(slope)
    asp_deg = np.degrees(aspect)
    slp_bins = np.concatenate((np.array([0.]), np.arange(2.5, 45, 5)))
    slp_ind = np.digitize(slp_deg, slp_bins) - 1  # -1 in the end gives an index starting with 0
    asp_bins = np.concatenate((np.array([0.]), np.arange(15., 380., 30.)))
    asp_ind = np.digitize(asp_deg, asp_bins) - 1
    asp_ind[asp_ind == 12] = 0   # To adjust asp>345deg fall in -15<asp<15 limit (ind 0)
    slp_asp_mapper = (slp_ind * 12 + asp_ind)
    return slp_asp_mapper


def get_sm_mapper(grid, slp_asp_mapper):
    """This function outputs a mapper that maps every cell
    in the *grid* to *sm_grid* based on vegetation_type, slope and aspect.
    Range of the mapper - 0 to 647.
    """
    veg_type = grid.at_cell['vegetation__plant_functional_type']
    mapper = (veg_type * 108 + slp_asp_mapper)
    return mapper

def calc_veg_cov(grid, veg_type):
    """This function calculates the percentage of the area covered
    by a PFT. 
    """
    grid_size = float(grid.number_of_cells)
    grass_cov = ((veg_type[veg_type == GRASS].size/grid_size) *
                         100)
    shrub_cov = ((veg_type[veg_type == SHRUB].size/grid_size) *
                        100)
    shseed_cov = ((veg_type[veg_type == SHRUBSEEDLING].size /
                        grid_size) * 100)
    tree_cov = ((veg_type[veg_type == TREE].size/grid_size) *
                       100)
    trseed_cov = ((veg_type[veg_type == TREESEEDLING].size /
                       grid_size) * 100)
    bare_cov = 100 - np.sum([grass_cov, shrub_cov, tree_cov, shseed_cov,
                            trseed_cov])
    return np.array([grass_cov, shrub_cov, tree_cov, bare_cov,
                     shseed_cov, trseed_cov])


def calc_veg_cov_wtrshd(grid, veg_type):
    """This function calculates the percentage of the watershed area
    covered by a PFT. 
    """
    grid_size = float(grid.number_of_core_cells)
    veg_type = veg_type[grid.core_cells]
    grass_cov = ((veg_type[veg_type == GRASS].size/grid_size) *
                         100)
    shrub_cov = ((veg_type[veg_type == SHRUB].size/grid_size) *
                        100)
    shseed_cov = ((veg_type[veg_type == SHRUBSEEDLING].size /
                        grid_size) * 100)
    tree_cov = ((veg_type[veg_type == TREE].size/grid_size) *
                       100)
    trseed_cov = ((veg_type[veg_type == TREESEEDLING].size /
                       grid_size) * 100)
    bare_cov = 100 - np.sum([grass_cov, shrub_cov, tree_cov, shseed_cov,
                            trseed_cov])
    return np.array([grass_cov, shrub_cov, tree_cov, bare_cov,
                     shseed_cov, trseed_cov])


def calc_cov_percent(grid_size, veg_type):
    """This function calculates the percentage of the 'grid_size', each
    PFT covers. This is a generalized version oc 'calc_veg_cov_..' functions.
    """
    grid_size = float(grid_size)
    return np.array([veg_type[veg_type == GRASS].size/grid_size * 100,
                     veg_type[veg_type == SHRUB].size/grid_size * 100,
                     veg_type[veg_type == TREE].size/grid_size * 100,
                     veg_type[veg_type == BARE].size/grid_size * 100,
                     veg_type[veg_type == SHRUBSEEDLING].size/grid_size * 100,
                     veg_type[veg_type == TREESEEDLING].size/grid_size * 100])


def calc_veg_cov_asp(grid, veg_type, slp_asp_mapper):
    """This function calculates the percentage of area covered by each
    PFT for each aspect. Aspect bins are from 0 thru 11. 12 stands for
    flat. Aspect bins start from 0 and are at increments of 30 deg.
    """
    slp_asp_mapper_core = slp_asp_mapper[grid.core_cells]
    asp_mapper = (slp_asp_mapper % 12)
    asp_mapper_core = asp_mapper[grid.core_cells]
    veg_type_core = veg_type[grid.core_cells]
    cov_asp = np.zeros([13, 6])
    for i in range(0, 12):
        asp_cells = np.where(np.logical_and(
                slp_asp_mapper_core > 11, asp_mapper_core == i))[0]
        cov_asp[i, :] = calc_cov_percent(
                asp_cells.size, veg_type_core[asp_cells])
    flt_cells = np.where(slp_asp_mapper_core<12)[0]
    cov_asp[12, :] = calc_cov_percent(flt_cells.size, veg_type_core[flt_cells])
    return cov_asp


## for plotting veg_cov obtained from calc_veg_cov
def plot_veg_cov(veg_cov, yrs=None, plot_seeds=False, savename='veg_cover'):
   plt.figure(figsize=(10, 8))
   if yrs == None:
       yrs = veg_cov.shape[0]
   years = np.arange(yrs)
   lw = 2
   if plot_seeds:
       plt.plot(years, veg_cov[:yrs, 0], '-g', label='Grass', linewidth=lw)
       plt.plot(years, veg_cov[:yrs, 1], '-r', label='Shrub', linewidth=lw)
       plt.plot(years, veg_cov[:yrs, 2], '-k', label='Tree', linewidth=lw)
       plt.plot(years, veg_cov[:yrs, 4], '-m', label='ShSeed', linewidth=lw)
       plt.plot(years, veg_cov[:yrs, 5], '-b', label='TrSeed', linewidth=lw)
   else:
       plt.plot(years, veg_cov[:yrs, 0], '-g', label='Grass', linewidth=lw)
       plt.plot(years, (veg_cov[:yrs, 1] + veg_cov[:yrs, 4]),
                '-r', label='Shrub', linewidth=lw)
       plt.plot(years, (veg_cov[:yrs, 2] + veg_cov[:yrs, 5]),
                '-k', label='Tree', linewidth=lw)
   plt.xlabel('Years')
   plt.ylabel('Vegetation Cover %')
   plt.xlim(xmin=0, xmax=np.max(years))
   plt.ylim(ymin=0, ymax=100)
   plt.legend()
   plt.savefig(savename)
