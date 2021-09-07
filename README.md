# pub_nudurupati_etal_wrr
Running Landlab Cellular Automaton (CA) ecohydrology model:
 

Folder: DEM_model_driver

 

CA_flat.py: Driver for a flat surface Cellular Automaton model that uses rainfall input files. Use the following function in the same folder: ecohyd_functions_flat.py

 

CA_DEM_input_storms.py: Driver for an actual topography Cellular Automaton that uses rainfall input files. Use the following function in the same folder: ecohyd_functions_dem_bins.py

 

Dem used: sev5m_nad27.txt

 

Storm files used are:

'precip_flt_31_lnger.npy'

'storm_dt_flt_31_lnger.npy'

'interstorm_dt_flt_31_lnger.npy'

 

Input file used: input_file.yaml. This can be used for topography and flat simulations with/without runon. If used with runon the runon driver should be used, configured to run the model spatially distributed.

 

Folder: Disturbance_model_driver

CA_disturbance.py

Ecohyd_functions_flat_coupled.py

Input_disturbances_new.yaml

 

Folder: Plotting scripts

Plotting:

Plotting scripts can be stored in a separate folder to keep them more organized. They need to reference to the directory where outputs are stored. Except for the aspect box whisker plot, the rest of the plots can be used for both DEM and flat simulations.

 

plotting_PFT_map.py: Maps Plant Functional Type in space. Options include plotting seedlings separately. If False then PFT include seedling within a type (plot_seeds = True). You will need to include the DEM of watershed modeled as txt file in the folder if this script is used to plot watershed PFT data. IF DEM is not used (elev_provided= False) Elevation contour lines can also be plotted (elev_contour = True). If a flat surface vegetation plot is made, please make sure to use correct row and columns of the modeled surface.  

 

plotting_PFT_cover_time_series.py: spatial fraction of each PFT plotted as a time series. Seeds can be included (plot_Seeds = True). If seeds are not included seedlings are added to the PFT cover fraction. 

 

plt_aspect_bxwhskr_PFT.py: This plots the box whisker plot of PFT cover for each 8 aspect class.

 

plot_only_vegcov_firAr_2.py:  plots veg cover time series and fire area averaged over a given averaging duration.

 

long_hist_connectivity.py: this plots the connectivity distribution of grass field and writes out the connectivity integral (index), CI.
