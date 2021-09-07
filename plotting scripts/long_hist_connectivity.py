
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 21:51:30 2020
plots connectivity values and writes out the CI for each time slice.

@author: saisiddu
"""
import os
# import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.font_manager as fm

GRASS = 0 # 0 indicates grass category. 1 is shrub

dem_path = r"C:\Users\saisiddu\Documents\GitHub\Sai_Oct14\Research_13Jan16\2019_paleo_research\Catgrass_SpDisturbance\paleo_Plot_scripts"
elev_provided = False
dem_fname = os.path.join(dem_path, 'sev5m_nad27.txt')
    
parent_dir =r"E:\Disturbance_results\output"
# pal_years = np.load(os.path.join(r"C:\Users\saisiddu\Documents\GitHub\Sai_Oct14\Research_13Jan16\2019_paleo_research\Catgrass_SpDisturbance",
#                                  "paleo_yrs_hall_2013.npy"))  ## Stephen Hall 2013 Years

files = ['sample_disturbance_run']

#bp_years = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285]  # Years BP to be considered
bp_years = [0, 20, 40, 60, 80, 100]  # Years BP to be considered
#bp_years = [12093, 10093, 7993, 3993, 1993]  # Years BP to be considered
extension = "_3"


for a_file in files:
    print('Working on ' + a_file)
    file_dir = os.path.join(parent_dir, a_file)
    if elev_provided:
        from landlab.io import read_esri_ascii
        (grid, elevation) = read_esri_ascii(dem_fname)    # Read the DEM
        grid.set_nodata_nodes_to_closed(elevation, -9999.)
    else:
        from landlab import RasterModelGrid
        grid = RasterModelGrid((182,182), spacing=(5., 5.))
        elevation = None
    
    os.chdir(file_dir)
    # wallclock_start = time.clock()     # Recording time taken for simulation
#    if os.path.exists("percent_con_hist.npy"):
#        con_histogram = np.load("connectivity_hist.npy")
#        per_con_hist = np.load("percent_con_hist.npy")
#    else:
    data = pickle.load(open("inputs.pickle", "rb"))
    # pal_yrs = int((np.max(pal_years) + 1) + data['init_ext'])  ## Initial Spinup
    prefixed = [
        "vegtype_" + str(bp_year) + ".npy" for bp_year in bp_years]
    con_histogram = np.empty([len(prefixed), 9], dtype=int)
    per_con_hist = np.empty([len(prefixed), 9], dtype=float)
    for i, filename in enumerate(prefixed):
#        yr = bp_years[i]
        veg_type = np.load(filename)
        gr_cells = np.where(veg_type == GRASS)[0]
        VEG_COV = np.size(gr_cells)/np.size(veg_type)
        first_ring = grid.looped_neighbors_at_cell[gr_cells]
       # gr_neigh = np.count_nonzero(veg_type[first_ring] == 0, axis=1)
        gr_neigh = np.count_nonzero(veg_type[first_ring] == GRASS, axis=1)
        con_histogram[i, :] = np.histogram(gr_neigh, bins=(np.arange(10) - 0.5))[0]
        #per_con_hist[i, :] = ((con_histogram[i, :]
                             #  / float(gr_cells.shape[0]))
                             #  * 100.)
        per_con_hist[i, :] = ((con_histogram[i, :]
                               / float(np.size(veg_type)))
                               )

        x=[0, 1, 2,3, 4, 5, 6, 7, 8]
        
        p=per_con_hist[i, :]
        
        CI=sum(x*p)/8
        
        print(CI)
        
    fig = plt.figure(figsize=(10, 5))
    lw = 2
    for i in np.arange(per_con_hist.shape[0]):        
        plt.plot(np.arange(9), per_con_hist[i, :], "-*",
                 linewidth=lw, markersize=12,
                 label="Year " + str(bp_years[i]) + " ")
    plt.xticks(fontsize=14, weight='bold')
    plt.yticks(fontsize=14, weight='bold')
    plt.xlabel("Number of connected grass neighbors",
               weight='bold', fontsize=16)
    plt.ylabel("Fraction of area",
               weight='bold', fontsize=16)
    plt.xlim(xmin=0, xmax=8)
    plt.ylim(ymin=0, ymax=0.8)
    font = fm.FontProperties(weight='bold', size=14)
    plt.legend(prop=font)
    plt.savefig("fire_connectivity_box"+extension, dpi=600)

    if not os.path.exists("percent_con_hist.npy"):
        np.save("connectivity_hist"+extension, con_histogram)
        np.save("percent_con_hist"+extension, per_con_hist)
        np.save("bp_years"+extension, np.array(bp_years))
