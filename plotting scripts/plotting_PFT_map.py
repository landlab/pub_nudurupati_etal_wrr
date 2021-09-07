
"""
Created on Fri May 04 10:11:48 2018
@author: saisiddu

- function to map PFT on the modeled domain from saved files. when reploting delete vegtype_000.png sometimes error occurs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from landlab.plot import imshow_grid


elev_provided = True  # elevation of topography
elev_contour = False  # elevation of topography
plot_seeds = False   # When False: both seeds and mature plants are same color#introduced on 29Nov18
parent_dir = r"E:\Figure_3\output"


files = ['sample_run_DEM']



# Below give the DEM asc or txt file to read if elev_provided is True
# else give the size of the flat domain (nrows, ncols) being plotted

for a_file in files:
    print('Working on ' + a_file)
    file_dir = os.path.join(parent_dir, a_file)
    if elev_provided:
        from landlab.io import read_esri_ascii
        (grid, elevation) = read_esri_ascii('sev5m_nad27.txt')    # Read the DEM
        grid.set_nodata_nodes_to_closed(elevation, -9999.)
    else:
        from landlab import RasterModelGrid #
        grid = RasterModelGrid((285, 799), spacing=(5., 5.))  # (285, 799) (98, 148) (49 58)
        elevation = None
    
    os.chdir(file_dir)
    prefixed = [
        filename for filename in os.listdir('.') if filename.startswith("vegtype")]
    
    if elev_provided:
        elevation[elevation<0] = np.max(elevation)   # To avoid issues with contour
        elev_raster = grid.node_vector_to_raster(elevation)
        elev_grid = np.zeros([elev_raster.shape[0]-2, elev_raster.shape[1]-2])
        xx = (grid.dx * (np.arange(elev_grid.shape[1])))
        yy = (grid.dy * (np.arange(elev_grid.shape[0])))
        for ii in range(1, int(elev_raster.shape[0]-1)):
            for jj in range(1, int(elev_raster.shape[1]-1)):
                 elev_grid[ii-1][jj-1] = elev_raster[ii][jj]
    
    print('Plotting cellular field of Plant Functional Type')
    #print('Green - Grass; Red - Shrubs; Black - Trees; White - Bare')
    #print('ShrubSeedling - Maroon; TreeSeedling - Blue')
    print('BluishGreen - Grass; Vermilion - Shrubs; Black - Trees; White - Bare')
    print('ShrubSeedling - Vermilion; TreeSeedling - Black')
    
    for filename in prefixed:
        yr = int(filename.replace('vegtype_', '').replace('.npy', ''))
        veg_type = np.load(filename)
        if plot_seeds:
    #        cmap = mpl.colors.ListedColormap(
    #                ['green', 'red', 'black', 'white', 'maroon', 'blue'])
    ## For colors to be visible to color blind, colors are chosen in the 
    ## following order: ['Bluish Green', 'Vermilion', 'Black',
    ##                   'White', 'Yellow', 'reddish purple']
            cmap = mpl.colors.ListedColormap(
                    [(0, 0.6, 0.5), (0.8, 0.4, 0), (0, 0, 0),
                     (1, 1, 1), (0.95, 0.9, 0.25), (0.8, 0.6, 0.7)])
    
        else:
    #        cmap = mpl.colors.ListedColormap(
    #        ['green', 'red', 'black', 'white', 'red', 'black'])
    ## For colors to be visible to color blind, colors are chosen in the 
    ## following order: ['Bluish Green', 'Vermilion', 'Black',
    ##                   'White', 'Vermilion', 'Black']
            cmap = mpl.colors.ListedColormap(
                    [(0, 0.6, 0.5), (0.8, 0.4, 0), (0, 0, 0),
                     (1, 1, 1), (0.8, 0.4, 0), (0, 0, 0)])
    
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
        plt.figure(figsize=(10, 5))
        imshow_grid(grid, veg_type, values_at='cell', cmap=cmap,
                    grid_units=('m', 'm'), norm=norm, limits=[0, 5],
                    allow_colorbar=False, color_for_closed='white')
        
        if elev_contour:
            clt = plt.contour(xx, yy, elev_grid, colors='b',
                              figsize=(10, 8), linewidths=2)
            plt.clabel(clt, inline=1, fmt='%4.0f', fontsize=10)
            
        name = 'PFT at Year = ' + "%05d" % yr
        plt.title(name, weight='bold', fontsize=10)
        plt.xlabel('X (m)', weight='bold', fontsize=10)
        plt.ylabel('Y (m)', weight='bold', fontsize=10)
        plt.xticks(fontsize=10, weight='bold')
        plt.yticks(fontsize=10, weight='bold')
        savename = 'vegtype_' + "%05d" % yr
        plt.savefig(savename, bbox_inches='tight', dpi=600)
        plt.close()