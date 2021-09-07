
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 19:01:38 2018
@author: saisiddu

- code to plot box-whisker plots of aspect specific veg_cover fractions. 
this code renders through vegetation output fields withon given year start and end dates and calculates fractions of PFTs in each aspect class.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from landlab.io import read_esri_ascii
import matplotlib.font_manager as fm

GRASS = 0
SHRUB = 1
TREE = 2
BARE = 3
SHRUBSEEDLING = 4
TREESEEDLING = 5

(grid, elevation) = read_esri_ascii('hugo5m_wgs84.txt')    # Read the DEM
grid.set_nodata_nodes_to_closed(elevation, -9999.)

# Gather data for Hugo's basin with Runon simulations
file_dir = r"E:\Figure_3\output\sample_run_DEM"
os.chdir(file_dir)
yr_start = 100
yr_end = 500 
plot_lidar_data = False    # if observations are available from lidar 
veg_cov_asp = np.load('veg_cover_aspect.npy')   #veg_cover_aspect.npy

## splitting data into 8 aspect bins and 1 flat bin
n_mat = veg_cov_asp[:, 0, :]   # ndarray_size = [yrs, 6 PFTs]
ne_mat = (veg_cov_asp[:, 1, :] + veg_cov_asp[:, 2, :])/2.   # NE 22.5 to 67.5 deg
e_mat = veg_cov_asp[:, 3, :]
se_mat = (veg_cov_asp[:, 4, :] + veg_cov_asp[:, 5, :])/2.
s_mat = veg_cov_asp[:, 6, :]
sw_mat = (veg_cov_asp[:, 7, :] + veg_cov_asp[:, 8, :])/2.
w_mat = veg_cov_asp[:, 9, :]
nw_mat = (veg_cov_asp[:, 10, :] + veg_cov_asp[:, 11, :])/2.
flat_mat = veg_cov_asp[:, 12, :]

# Lidar data (if available)
lidar_data = [20, 12, 4, 2, 1, 1.5, 9.5, 20, 2.5]  # For Trees - Xiaochi (Zhou et al. 2013)

# Functions for plotting
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
# Set ticks for the plots
ticks = ['N', 'NE', 'E', 'SE', 'S',
         'SW', 'W', 'NW', 'Flat']
color_2 = (0.8, 0.4, 0)  # RGB - Vermilion
color_1 = (0, 0.6, 0.5)  # RGB - Bluish Green
color_3 = (0, 0, 0)    # RGB - Black
color_4 = (0, 0.45, 0.7)   # RGB - Blue
boxprops = dict(linewidth=3)
whiskerprops = dict(linewidth=2)
capprops = dict(linewidth=1.5)

# Run the loop to plot boxplots for each PFT
pft = 0
data = [n_mat[yr_start:yr_end, pft],
        ne_mat[yr_start:yr_end, pft],
        e_mat[yr_start:yr_end, pft],
        se_mat[yr_start:yr_end, pft],
        s_mat[yr_start:yr_end, pft],
        sw_mat[yr_start:yr_end, pft],
        w_mat[yr_start:yr_end, pft],
        nw_mat[yr_start:yr_end, pft],
        flat_mat[yr_start:yr_end, pft]]
pft = 1
data_1 = [n_mat[yr_start:yr_end, pft],
          ne_mat[yr_start:yr_end, pft],
          e_mat[yr_start:yr_end, pft],
          se_mat[yr_start:yr_end, pft],
          s_mat[yr_start:yr_end, pft],
          sw_mat[yr_start:yr_end, pft],
          w_mat[yr_start:yr_end, pft],
          nw_mat[yr_start:yr_end, pft],
          flat_mat[yr_start:yr_end, pft]]
pft = 2
data_2 = [n_mat[yr_start:yr_end, pft],
          ne_mat[yr_start:yr_end, pft],
          e_mat[yr_start:yr_end, pft],
          se_mat[yr_start:yr_end, pft],
          s_mat[yr_start:yr_end, pft],
          sw_mat[yr_start:yr_end, pft],
          w_mat[yr_start:yr_end, pft],
          nw_mat[yr_start:yr_end, pft],
          flat_mat[yr_start:yr_end, pft]]

# Plot boxplot
plt.figure(figsize=(10, 6))
d1 = plt.boxplot(data, showfliers=False,
            positions=np.array(np.arange(len(data)))*2.0-0.5,
            sym='', widths=0.3,
            whiskerprops=whiskerprops, capprops=capprops,
            boxprops=boxprops, medianprops=boxprops)
d2 = plt.boxplot(data_1, showfliers=False,
            positions=np.array(np.arange(len(data_1)))*2.0-0.15,
            sym='', widths=0.3,
            whiskerprops=whiskerprops, capprops=capprops,
            boxprops=boxprops, medianprops=boxprops)
d3 = plt.boxplot(data_2, showfliers=False,
                 positions=np.array(np.arange(len(data)))*2.0+0.2,
                 sym='', widths=0.3,
                 whiskerprops=whiskerprops, capprops=capprops,
                 boxprops=boxprops, medianprops=boxprops)
if plot_lidar_data:
    marker_style = dict(linestyle='-', color=color_3, markersize=10)
    font = fm.FontProperties(weight='bold', size=12)
    plt.plot(np.array(np.arange(len(data)))*2.0+0.2, lidar_data,
             marker="D", label='Lidar_Tree', **marker_style)
    plt.legend(prop=font)
set_box_color(d1, color_1) # colors are from http://colorbrewer2.org/
set_box_color(d2, color_2)
set_box_color(d3, color_3)
plt.ylabel('Vegetation cover [%]', fontsize=16,
           fontweight='bold')
plt.xlabel('Aspect', fontsize=16,
           fontweight='bold')
plt.xlim(-1, len(ticks)*2-1)
plt.ylim(ymin=0, ymax=50)
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(range(0, len(ticks) * 2, 2), ticks,
           fontsize=16, fontweight='bold')
# draw temporary red and blue lines and use them to create a legend
font = fm.FontProperties(weight='bold', size=14)
plt.plot([], c=color_1, label='Grass')
plt.plot([], c=color_2, label='Shrub')
plt.plot([], c=color_3, label='Tree')
plt.legend(prop=font)
plt.savefig('asp_vegcov_three_pfts' + '_' + str(pft) +
            '_yrs_' + str(yr_start) + '_to_' + str(yr_end),
            dpi=600)

