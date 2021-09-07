
# -*- coding: utf-8 -*-
# """
# Created on Mon Jan 28 12:09:10 2019 
# @author: saisiddu
# Added code from C:\Users\saisiddu\Documents\GitHub\Sai_Oct14\Research_13Jan16\2017_Jan_Test_SM_runon\ecohydrology_flat_surface\ecohyd_functions_flat.py
# Changed to a color blind friendly pallettes.
# """
#plots time series of spatially averaged vegetation cover fraction for each PFT

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def main():
    plot_seeds = False   # this can plot seed as separate time series
    parent_dir = r"E:\Figure_3\output\sample_run_DEM"
   
   # vegcov_file = 'vegetation_cover.npy' #'veg_cov.npy'   #'veg_cover_wtrshd.npy' 'vegetation_cover.npy'
    vegcov_file ='veg_cover_wtrshd.npy' # see output veg_cover_X files different model output slightly different file names
    years_to_plot=501
   

    file_dir = parent_dir
    os.chdir(file_dir)
    veg_cov = np.load(vegcov_file)
    plot_veg_cov(veg_cov, yrs=years_to_plot, plot_seeds=plot_seeds,
                 savename='vegetation_cover', xmax=years_to_plot,
                 ymax=70, lbl_fnt=18)

def plot_veg_cov(veg_cov, yrs=None, plot_seeds=False, savename='veg_cover',
                 xmax=None, ymax=80, legend_col=None, leg_loc=0,
                 leg_col=1, figwidth=10, figheight=5,
                 lbl_fnt=18):
   plt.figure(figsize=(figwidth, figheight))
   if yrs == None:
       yrs = veg_cov.shape[0]
   years = np.arange(yrs)
   lw = 3 
   if plot_seeds:
       plt.plot(years, veg_cov[:yrs, 0],
                color=(0, 0.6, 0.5), label='Grass', linewidth=lw)
       plt.plot(years, veg_cov[:yrs, 1],
                color=(0.8, 0.4, 0), label='Shrub', linewidth=lw)
       plt.plot(years, veg_cov[:yrs, 2],
                color=(0, 0, 0), label='Tree', linewidth=lw)
       plt.plot(years, veg_cov[:yrs, 4],
                color=(0.95, 0.9, 0.25), label='ShSeed', linewidth=lw)
       plt.plot(years, veg_cov[:yrs, 5],
                color=(0.8, 0.6, 0.7), label='TrSeed', linewidth=lw)
   else:
       plt.plot(years, veg_cov[:yrs, 0],
                color=(0, 0.6, 0.5), label='Grass', linewidth=lw)
       plt.plot(years, (veg_cov[:yrs, 1] + veg_cov[:yrs, 4]),     #seedlings are added to mature plants
                color=(0.8, 0.4, 0), label='Shrub', linewidth=lw)
       plt.plot(years, (veg_cov[:yrs, 2] + veg_cov[:yrs, 5]),
                color=(0, 0, 0), label='Tree', linewidth=lw)

   plt.xlabel('Years', weight='bold', fontsize=lbl_fnt)
   plt.ylabel('Vegetation Cover (%)', weight='bold', fontsize=lbl_fnt)
   plt.xticks(fontsize=16, weight='bold')
   plt.yticks(fontsize=16, weight='bold')
   if xmax == None:
       xmax = np.max(years)
   plt.xlim(xmin=0, xmax=xmax)
   plt.ylim(ymin=0, ymax=ymax)
   font = fm.FontProperties(weight='bold', size=12)
   plt.legend(prop=font, loc=leg_loc, ncol=leg_col)
   plt.savefig(savename, bbox_inches='tight',dpi=600)

if __name__ == '__main__':
    main()
