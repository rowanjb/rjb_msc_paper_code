# Produces heat content and temperature 
# Rowan Brown
# 17 May 2023

import numpy as np 
import pandas as pd
import xarray as xr
import os

def heat(run,mask_choice): 
    """Calculates heat content"""

    # Mask for land, bathymetry, etc. and horiz. grid dimensions
    with xr.open_dataset('masks/ANHA4_mesh_mask.nc') as DS:
        tmask = DS.tmask[0,:,:,:].rename({'z': 'deptht', 'y': 'y_grid_T', 'x': 'x_grid_T'}) #DataArray with dims (t: 1, z: 50, y: 800, x: 544) 
        e1t = DS.e1t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
        e2t = DS.e2t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})

    if mask_choice == 'LS2k': #mask for 2000m depth interior area
        mask = xr.open_dataarray('masks/mask_LS_2k.nc').astype(int)
    elif mask_choice == 'LS3k': #mask for 3000m depth interior area
        mask = xr.open_dataarray('masks/mask_LS_3000.nc').astype(int)
    elif mask_choice == 'LS': #mask for entire LS region
        mask = xr.open_dataarray('masks/mask_LS.nc').astype(int)
    elif mask_choice == 'LSCR': #mask for LS convection region
        mask = xr.open_dataset('masks/ARGOProfiles_mask.nc').tmask.astype(int).rename({'x':'x_grid_T','y':'y_grid_T'})
    else: 
        print("Y'all didn't choose a mask")
        quit()

    ##################################################################################################################
    #OPENING AND INITIAL PROCESSING OF THE NETCDF MODEL OUTPUT FILES

    #text file of paths to non-empty model output
    gridT_txt = run + '_filepaths/' + run + '_gridT_filepaths_Jan2024.txt'

    #open the text files and get lists of the .nc output filepaths
    with open(gridT_txt) as f: lines = f.readlines()
    filepaths_gridT = [line.strip() for line in lines]

    #open the files and look at e3t and votemper
    preprocess_gridT = lambda ds: ds[['e3t','votemper']]
    DS = xr.open_mfdataset(filepaths_gridT,preprocess=preprocess_gridT)

    #add horizontal cell dims
    DS[['e1t','e2t']] = e1t,e2t #add T cell dimensions as variables

    #apply tmask (ie masking bathy)
    DS = DS.where(tmask == 1)

    #apply mask (if there is one)
    if mask_choice == 'LSCR' or mask_choice == 'LS2k' or mask_choice == 'LS' or mask_choice == 'LS3k':
        DS.coords['mask'] = mask
        DS = DS.where(DS.mask == 1, drop=True)

    ##################################################################################################################
    #CALCULATIONS

    #constants needed for heat content calculations
    refT = -1.8 #reference temperature [C]                              (Value from Paul's email)
    rho_0 = 1026#1025 #density reference (const. value?( [kg/m^3]       (Value from Gillard paper)
    C_p = 3992#3850 #specific heat capacity (const. value?) [J/kgK]     (Value from Gillard paper)

    #loop through the 4 depths and save .nc files
    for d in [200]: #[50, 200, 1000, 2000]: #loop through depths
        DS_d = DS.where(DS.deptht < d, drop=True) #drop values below specified depth

        #note: there are two main ideas below: "col" refers to the idea that we're looking at water-columnwise averages, ie so we can make maps later. On 
        #the other hand, "region" refers to regionwise averages, so that we can make time plots later.

        #masking shelves
        #NOTE: bathy is masked to avoid skewed understandings/results from the on-shelf values this section could be commented out if needed 
        bottom_slice = DS_d.votemper.isel(deptht = -1).isel(time_counter = 0) 
        bottom_slice_bool = bottom_slice.notnull()
        shelf_mask, temp = xr.broadcast(bottom_slice_bool, DS_d.votemper.isel(time_counter=0))
        DS_d = DS_d.where(shelf_mask) 

        ###temperature averaged through time 
        ###cell weights (col): divide cell volume by average cell volume in each column
        volumes = DS_d.e1t*DS_d.e3t*DS_d.e2t #volume of each cell
        ##avg_col_vol = volumes.mean(dim='deptht') #average cell volume in each column 
        ##weights = volumes/avg_col_vol #dataarray of weights 
        ##weights = weights.fillna(0)
        ##votemper_col_weighted = DS_d.votemper.weighted(weights)
        ##votemper_avg_col = votemper_col_weighted.mean(dim='deptht') #NOTE: skipna should be True if not blocking shelves
        ##votemper_avg_col_time = votemper_avg_col.mean(dim='time_counter')
        ##votemper_avg_col_time.to_netcdf(run + '_heat/' + run + '_votemper_timeAvg_' + mask_choice + str(d) + '.nc') #.nc with time-avg temp in each column (ie for making maps)
        ##
        ###temperature averaged in space
        ###cell weights (region): divide cell volume by average cell volume in the whole masked region
        ##avg_cell_vol = volumes.mean(dim=['deptht','y_grid_T','x_grid_T'])
        ##weights = volumes/avg_cell_vol
        ##weights = weights.fillna(0)
        ##votemper_region_weighted = DS_d.votemper.weighted(weights)
        ##votemper_avg_region = votemper_region_weighted.mean(dim=['deptht','y_grid_T','x_grid_T'],skipna=True)
        ##votemper_avg_region.to_netcdf(run + '_heat/' + run + '_votemper_spaceAvg_' + mask_choice + str(d) + '.nc') 

        #heat content
        HC = rho_0 * C_p * 10**(-12) * (DS_d.votemper - refT) * volumes  
        HC_sum_deptht = HC.sum(dim='deptht')
        #HC_avg_time = HC_sum_deptht.mean(dim='time_counter')
        #HC_avg_time.to_netcdf(run + '_heat/' + run + '_HC_timeAvg_' + mask_choice + str(d) + '.nc') 
        HC_sum_space = HC_sum_deptht.sum(dim=['y_grid_T','x_grid_T'])
        HC_sum_space.to_netcdf(run + '_heat/' + run + '_HC_spaceSum_' + mask_choice + str(d) + '.nc')

    print('completed: ' + run + ', ' + mask_choice)

if __name__ == '__main__':
    for run in ['EPM151','EPM152','EPM155','EPM156','EPM157','EPM158']:#'EPM158' #specify the run
        mask_choice = 'LS3k' #choose which mask; options are 'LSCR', 'LS2k', or 'LS'
        heat(run,mask_choice)
