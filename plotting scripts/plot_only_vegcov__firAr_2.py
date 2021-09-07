
# -*- coding: utf-8 -*-
"""
Created on 17Aug19
Similar to plot_vegcov_vegtype_frm_saved_files.py
Written to incorporate Average % of fire area over 
every consecutive 100 years with time.

@author: Sai Nudurupati
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from landlab.plot import imshow_grid
import matplotlib.font_manager as fm


elev_provided = False
plot_seeds = False   # introduced on 29Nov18
file_dir = r"E:\Disturbance_results\output\sample_disturbance_run"
#file_dir = r"E:\Disturbance_results\output\Control_all_PFT"
savename = 'TS_veg_fire_area.jpg' # File will be saved as savename
vegcov_file = 'veg_cov.npy'     # filename for vegcover
vc_xlimmax = 100   # xlim max for vegcover plots



os.chdir(file_dir)
veg_cov = np.load(vegcov_file)

#if veg_cov == None:
 #   print('Vegetation Cover (veg_cov) not provided!')

## 17Aug19: Adding plot with mean fire area with time
per_yr_burnt = np.load('per_yr_burnt_area_fr.npy')
fire_x_a = 1   # Starting x-axis for fire plot
fire_x_b = vc_xlimmax   # ending x-axis for fire plot
fire_x_spacing = 10    # Spacing for x axis-- averaging duration
fire_y_max = 8    # Maximum y axis for fire plot
ave_spacing = fire_x_spacing      # (yrs) Averaging fires over ave_spacing years
l = int(per_yr_burnt.shape[0])
ave = np.zeros(int(l/ave_spacing))
k = 0
for i in np.arange(int(ave.shape[0])-1):
   # sum = np.sum(per_yr_burnt[k:(i+1)*ave_spacing])
    sum = np.sum(per_yr_burnt[k:(i+1)*ave_spacing])
    ave[i] = sum/ave_spacing
    k = (i+1)*ave_spacing

## Plot figure
fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
yrs = veg_cov.shape[0]
years = np.arange(yrs)
lw = 3
## For colors to be visible to color blind, colors are chosen in the 
## following order: ['Bluish Green', 'Vermilion', 'Black',
##                   'White', 'Vermilion', 'Black']

### FOR PLOTTING SHRUB DETAILS

#plt.plot(years, (veg_cov[:yrs, 1]),
#           color=(0.8, 0.4, 0), label='Shrub', linewidth=lw)
#plt.plot(years, (veg_cov[:yrs, 4]),
#            color=(0, 0.6, 0.5), label='Shrub seedlings', linewidth=lw)

# FOR PLOTTING TREE DETAILS

#plt.plot(years, (veg_cov[:yrs, 5]),
#           color=(0, 0.6, 0.5), label='Tree Seedlings', linewidth=lw)
#plt.plot(years, (veg_cov[:yrs, 2]),
#           color=(0, 0, 0), label='Tree', linewidth=lw)

## FOR PLOTTING ALL PFT
plt.plot(years, veg_cov[:yrs, 0],
         color=(0, 0.6, 0.5), label='Grass', linewidth=lw)
plt.plot(years, (veg_cov[:yrs, 1] + veg_cov[:yrs, 4]),
          color=(0.8, 0.4, 0), label='Shrub', linewidth=lw)
plt.plot(years, (veg_cov[:yrs, 2] + veg_cov[:yrs, 5]),
          color=(0, 0, 0), label='Tree', linewidth=lw)

plt.xlabel('Years', weight='bold', fontsize=16)
plt.ylabel('Vegetation Cover (%)', weight='bold', fontsize=16)
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.xlim(xmin=0, xmax=vc_xlimmax)
plt.ylim(ymin=0, ymax=100)
font = fm.FontProperties(weight='bold', size=12)
plt.legend(prop=font, loc=1)

ax1 = ax.twinx()
ax1.plot(np.arange(fire_x_a, fire_x_b, fire_x_spacing), ave*100,
         color='gray', linewidth=lw,
         alpha=0.3)
ax1.fill_between(np.arange(fire_x_a, fire_x_b, fire_x_spacing),
                 0, ave*100, color='gray', alpha=0.3)
plt.xlim(xmin=0, xmax=vc_xlimmax)
plt.ylim(ymin=0, ymax=fire_y_max)
plt.ylabel('Fire Area (%)',
           weight='bold', fontsize=16,
           color='black')
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.plot([], [], color='gray', marker='s', linestyle='None',
         markersize=12, label='10-year Mean fire area (%)')
plt.legend(prop=font, loc='upper left')

plt.savefig(savename, dpi=600) #, bbox_inches='tight')