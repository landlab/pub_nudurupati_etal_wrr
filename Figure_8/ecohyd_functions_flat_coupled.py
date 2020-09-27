# Jai Sri Sainath!
# Authors: Sai Nudurupati & Erkan Istanbulluoglu, 21May15
# Edited: 15Jul16 - to conform to Landlab version 1.
# Updated: 22May17 - for CSDMS 2017
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from landlab.plot import imshow_grid
from landlab.components import (PrecipitationDistribution, Radiation,
                                PotentialEvapotranspiration, SoilMoisture,
                                Vegetation, VegCA)

GRASS = 0
SHRUB = 1
TREE = 2
BARE = 3
SHRUBSEEDLING = 4
TREESEEDLING = 5


def compose_veg_grid(grid, percent_bare=0.4, percent_grass=0.2,
                     percent_shrub=0.2, percent_tree=0.2):
    """Compose spatially distribute PFT."""
    no_cells = grid.number_of_cells
    shrub_point = int(percent_bare * no_cells)
    tree_point = int((percent_bare + percent_shrub) * no_cells)
    grass_point = int((1 - percent_grass) * no_cells)

    veg_grid = np.full(grid.number_of_cells, BARE, dtype=int)
    veg_grid[shrub_point:grass_point] = SHRUB
    veg_grid[tree_point:grass_point] = TREE
    veg_grid[grass_point:] = GRASS

    np.random.shuffle(veg_grid)
    return veg_grid


def initialize(data, grid, grid1):
    """Initialize random plant type field.

    Plant types are defined as the following:

    *  GRASS = 0
    *  SHRUB = 1
    *  TREE = 2
    *  BARE = 3
    *  SHRUBSEEDLING = 4
    *  TREESEEDLING = 5
    """
    grid1.at_cell['vegetation__plant_functional_type'] = compose_veg_grid(
        grid1, percent_bare=data['percent_bare_initial'],
        percent_grass=data['percent_grass_initial'],
        percent_shrub=data['percent_shrub_initial'],
        percent_tree=data['percent_tree_initial'])

    # Assign plant type for representative ecohydrologic simulations
    grid.at_cell['vegetation__plant_functional_type'] = np.arange(6)
    grid1.at_node['topographic__elevation'] = np.full(grid1.number_of_nodes,
                                                      1700.)
    grid.at_node['topographic__elevation'] = np.full(grid.number_of_nodes,
                                                     1700.)
    precip_dry = PrecipitationDistribution(
        mean_storm_duration=data['mean_storm_dry'],
        mean_interstorm_duration=data['mean_interstorm_dry'],
        mean_storm_depth=data['mean_storm_depth_dry'])
    precip_wet = PrecipitationDistribution(
        mean_storm_duration=data['mean_storm_wet'],
        mean_interstorm_duration=data['mean_interstorm_wet'],
        mean_storm_depth=data['mean_storm_depth_wet'])

    radiation = Radiation(grid)
    pet_tree = PotentialEvapotranspiration(grid, method=data['PET_method'],
                                           MeanTmaxF=data['MeanTmaxF_tree'],
                                           delta_d=data['DeltaD'])
    pet_shrub = PotentialEvapotranspiration(grid, method=data['PET_method'],
                                            MeanTmaxF=data['MeanTmaxF_shrub'],
                                            delta_d=data['DeltaD'])
    pet_grass = PotentialEvapotranspiration(grid, method=data['PET_method'],
                                            MeanTmaxF=data['MeanTmaxF_grass'],
                                            delta_d=data['DeltaD'])
    soil_moisture = SoilMoisture(grid, **data)  # Soil Moisture object
    vegetation = Vegetation(grid, **data)  # Vegetation object
    vegca = VegCA(grid1, **data)  # Cellular automaton object

    # Initializing inputs for Soil Moisture object
    grid.at_cell['vegetation__live_leaf_area_index'] = (
        1.6 * np.ones(grid.number_of_cells))
    grid.at_cell['soil_moisture__initial_saturation_fraction'] = (
        0.59 * np.ones(grid.number_of_cells))

    return (precip_dry, precip_wet, radiation, pet_tree, pet_shrub,
            pet_grass, soil_moisture, vegetation, vegca)


def empty_arrays(n, grid, grid1):
    precip = np.empty(n) # Record precipitation
    inter_storm_dt = np.empty(n) # Record inter storm duration
    storm_dt = np.empty(n) # Record storm duration
    daily_pet = np.zeros([365, grid.number_of_cells])
    rad_factor = np.empty([365, grid.number_of_cells])
    EP30 = np.empty([365, grid.number_of_cells])
    # 30 day average PET to determine season
    pet_threshold = 0  # Initializing pet_threshold to ETThresholddown
    return (precip, inter_storm_dt, storm_dt,
            daily_pet, rad_factor, EP30, pet_threshold)


def create_pet_lookup(grid, radiation, pet_tree, pet_shrub, pet_grass,
                      daily_pet, rad_factor, EP30):
    for i in range(0, 365):
        pet_tree.update(float(i) / 365.25)
        pet_shrub.update(float(i) / 365.25)
        pet_grass.update(float(i) / 365.25)
        daily_pet[i] = [pet_grass._PET_value, pet_shrub._PET_value,
                        pet_tree._PET_value, 0., pet_shrub._PET_value,
                        pet_tree._PET_value]
        radiation.update(float(i) / 365.25)
        rad_factor[i] = grid.at_cell['radiation__ratio_to_flat_surface']

        if i < 30:
            if i == 0:
                EP30[0] = daily_pet[0]
            else:
                EP30[i] = np.mean(daily_pet[:i], axis=0)
        else:
            EP30[i] = np.mean(daily_pet[i - 30:i], axis=0)
    return (EP30, daily_pet)


def save(sim, inter_storm_dt, storm_dt, precip, veg_type, yrs,
         walltime, time_elapsed):
    np.save(sim + '_Tb', inter_storm_dt)
    np.save(sim + '_Tr', storm_dt)
    np.save(sim + '_P', precip)
    np.save(sim + '_VegType', veg_type)
    np.save(sim + '_Years', yrs)
    np.save(sim + '_Time_Consumed_minutes', walltime)
    np.save(sim + '_CurrentTime', time_elapsed)


def save_np_files(np_files={}):
    # np_files - dict
    #    np_files.keys() - list of variable names
    #    np_files.values() - list of corresponding values
    if np_files != {}:
        for file_name, a_file in zip(np_files.keys(), np_files.values()):
            np.save(file_name+'.npy', a_file)
    else:
        pass


def plot(sim, grid, veg_type, yrs, yr_step=10):
    pic = 0
    years = range(0, yrs)
    cmap = mpl.colors.ListedColormap(
        ['green', 'red', 'black', 'white', 'red', 'black'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    print 'Plotting cellular field of Plant Functional Type'
    print 'Green - Grass; Red - Shrubs; Black - Trees; White - Bare'

    # Plot images to make gif.
    for year in range(0, yrs, yr_step):
        filename = 'year_' + "%05d" % year
        pic += 1
        plt.figure(pic, figsize=(10, 8))
        imshow_grid(grid, veg_type[year], values_at='cell', cmap=cmap,
                    grid_units=('m', 'm'), norm=norm, limits=[0, 5],
                    allow_colorbar=False)
        plt.title(filename, weight='bold', fontsize=22)
        plt.xlabel('X (m)', weight='bold', fontsize=18)
        plt.ylabel('Y (m)', weight='bold', fontsize=18)
        plt.xticks(fontsize=14, weight='bold')
        plt.yticks(fontsize=14, weight='bold')
        plt.savefig(sim + '_' + filename)

    grass_cov = np.empty(yrs)
    shrub_cov = np.empty(yrs)
    tree_cov = np.empty(yrs)
    grid_size = float(veg_type.shape[1])

    for x in range(0, yrs):
        grass_cov[x] = ((veg_type[x][veg_type[x] == GRASS].size / grid_size) *
                        100)
        shrub_cov[x] = ((veg_type[x][veg_type[x] == SHRUB].size / grid_size) *
                        100 + (veg_type[x][veg_type[x] == SHRUBSEEDLING].size /
                        grid_size) * 100)
        tree_cov[x] = ((veg_type[x][veg_type[x] == TREE].size / grid_size) *
                       100 + (veg_type[x][veg_type[x] == TREESEEDLING].size /
                       grid_size) * 100)

    pic += 1
    plt.figure(pic, figsize=(10, 8))
    plt.plot(years, grass_cov, '-g', label='Grass', linewidth=4)
    plt.hold(True)
    plt.plot(years, shrub_cov, '-r', label='Shrub', linewidth=4)
    plt.hold(True)
    plt.plot(years, tree_cov, '-k', label='Tree', linewidth=4)
    plt.ylabel('% Area Covered by Plant Type', weight='bold', fontsize=18)
    plt.xlabel('Time in years', weight='bold', fontsize=18)
    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')
    plt.xlim((0, yrs))
    plt.ylim(ymin=0, ymax=100)
    plt.legend(loc=1, prop={'size': 16, 'weight': 'bold'})
    plt.savefig(sim + '_percent_cover')


def calc_veg_cov(grid, veg_type):
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

## 06Dec18: func to add woody plants if none in the system
# this func checks if there are any woody plants and adds few
# if there are none. Adds fr_addition of bottom two rows of shrub seedlings.
# Adds fr_addition of top two rows of tree seedlings. 
def check_and_add_woody_plants(grid, fraction_addition=0.8):
    veg_type = grid.at_cell['vegetation__plant_functional_type']
    c_rows = grid.number_of_cell_rows
    c_cols = grid.number_of_cell_columns
    t_cells = grid.number_of_cells
    first_two_rows = np.arange(2*c_cols)
    last_two_rows = np.arange((c_rows-2)*c_cols, t_cells)
    if np.logical_and(
            np.where(veg_type == SHRUB)[0].shape[0] == 0,
            np.where(veg_type == SHRUBSEEDLING)[0].shape[0] == 0):
        dice = np.random.rand(first_two_rows.shape[0])
        locs_to_add = first_two_rows[np.where(dice<fraction_addition)[0]]
        veg_type[locs_to_add] = SHRUBSEEDLING
    if np.logical_and(
            np.where(veg_type == TREE)[0].shape[0] == 0,
            np.where(veg_type == TREESEEDLING)[0].shape[0] == 0):
        dice = np.random.rand(last_two_rows.shape[0])
        locs_to_add = last_two_rows[np.where(dice<fraction_addition)[0]]
        veg_type[locs_to_add] = TREESEEDLING
    return veg_type

def add_trees(grid, fraction_addition=0.1, chck_presence=False):
    veg_type = grid.at_cell['vegetation__plant_functional_type']
    if chck_presence:
        if np.logical_or(
                check_if_plants_around(veg_type, PFT=TREE),
                check_if_plants_around(veg_type, PFT=TREESEEDLING)):
            return veg_type
    dice = np.random.rand(grid.number_of_cells)
    elig_locs_to_add = np.where(dice<fraction_addition)[0]
    locs_to_add = elig_locs_to_add[
            np.where(veg_type[elig_locs_to_add]==BARE)[0]]
    veg_type[locs_to_add] = TREESEEDLING
    return veg_type

def add_shrubs(grid, fraction_addition=0.1, chck_presence=False):
    veg_type = grid.at_cell['vegetation__plant_functional_type']
    if chck_presence:
        if np.logical_or(
                check_if_plants_around(veg_type, PFT=SHRUB),
                check_if_plants_around(veg_type, PFT=SHRUBSEEDLING)):
            return veg_type
    dice = np.random.rand(grid.number_of_cells)
    elig_locs_to_add = np.where(dice<fraction_addition)[0]
    locs_to_add = elig_locs_to_add[
            np.where(veg_type[elig_locs_to_add]==BARE)[0]]
    veg_type[locs_to_add] = SHRUBSEEDLING
    return veg_type

def check_if_plants_around(veg_type, PFT=TREE):
    if np.where(veg_type == PFT)[0].shape[0] == 0:
        return False
    else:
        return True

# Added 24Feb19 - get cells that belong to an edge designed
# by edge_side and edge_width for a RasterModelGrid
def get_edge_cell_ids(grid, edge_side='left', edge_width=3):
    cols = grid.number_of_cell_columns
    rows = grid.number_of_cell_rows
    cells = []
    if edge_side == 'bottom':
        cells = range(0, edge_width*cols)
    elif edge_side == 'top':
        cells = range((rows-edge_width)*cols, grid.number_of_cells)
    else:
        for i in range(edge_width):
            if edge_side == 'left':            
                cells += range(i, grid.number_of_cells, cols)
            elif edge_side == 'right':
                cells += range(cols-i-1, grid.number_of_cells, cols)
            else:
                print('Edge side not recognized!')
    return np.array(cells)

# Added 24Feb19 - mark cells on RasterGrid
def plot_marked_cells(cells, grid, values=None, marker='x',
                      marker_color='k'):
#    if import_packages==True:
#        import matplotlib.pyplot as plt
#        from landlab.plot import imshow_grid
#        import numpy as np
    nodes = grid.node_at_cell[cells]
    if values == None:
         values = np.zeros(grid.number_of_nodes)
    plt.figure(figsize=(10, 8))
    imshow_grid(grid, values=values, values_at='node', cmap='jet')
    plt.plot(grid.x_of_node[nodes], grid.y_of_node[nodes],
             marker+marker_color)

# Added 24Feb19 - set vegtype at cells
def set_pft_at_cells(cells, grid, pft):
    """
    cells: int  
        cell ids at which PFT needs to be updated
    grid: RasterModelGrid
        Landlab's raster grid instance
    pft: int
        PFT to be set at the above mentioned cells (Xiaochi's convention)
    """
    grid.at_cell['vegetation__plant_functional_type'][cells] = pft

# Added 17Mar19 - Getting fraction of seeds to add based on
# current PFT coverage - using a linear model
def get_fr_seeds(x, x_min=0., x_max=0., y_min=0., y_max=0.):
    if x_min <= x <= x_max:
        y = (float((x-x_min)*(y_max-y_min))/(x_max-x_min) + y_min)
    elif x < x_min:
        y = y_min
    else:
        y = y_max
    return y
