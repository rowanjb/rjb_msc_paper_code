#Produces MLD (average and maximum)
#Rowan Brown
#17 May 2023

import numpy as np
import pandas as pd
import xarray as xr
import os

def MLD(run,mask_choice,movie=False):

    #== creating directory if doesn't already exist ==#
    dir = run + '_MLD/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    #== masks ==#
    with xr.open_dataset('masks/ANHA4_mesh_mask.nc') as DS: #mask for land, bathymetry, etc. and horiz. grid dimensions
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

    #== opening model output ==# 
    gridT_txt = run + '_filepaths/' + run + '_gridT_filepaths.txt' #text file of paths to non-empty model output
    with open(gridT_txt) as f: lines = f.readlines() #open the text files
    filepaths_gridT = [line.strip() for line in lines] #get lists of the .nc output filepaths
    num_files = len(filepaths_gridT)
    preprocess_gridT = lambda ds: ds[['e3t','somxlts']]#,'vooxy']] #specify veriables to retrieve  
    DS = xr.open_mfdataset(filepaths_gridT,preprocess=preprocess_gridT) #open the files (and look at e3t and sohmld)
    
    #== applying masks ==#
    DS[['e1t','e2t']] = e1t,e2t #add T cell dimensions as variables
    DS = DS.where(tmask == 1) #apply tmask (ie masking bathy)
    if mask_choice == 'LSCR' or mask_choice == 'LS2k' or mask_choice == 'LS': #apply mask
        DS.coords['mask'] = mask
        DS = DS.where(DS.mask == 1, drop=True)
        DS = DS.drop_vars(['mask','time_centered'])

    #== selecting only one depth slice (since MLD is constant throughout the water column) ==#
    MLD = DS.somxlts.isel(deptht = 0)#.somxl010.isel(deptht = 0)

    ##masking shelves
    ##NOTE: bathy is masked to avoid skewed understandings/results from the on-shelf values this section could be commented out if needed 
    #bottom_slice = DS_d.vosaline.isel(deptht = -1).isel(time_counter = 0)
    #bottom_slice_bool = bottom_slice.notnull()
    #shelf_mask, temp = xr.broadcast(bottom_slice_bool, DS_d.vosaline.isel(time_counter=0))
    #DS_d = DS_d.where(shelf_mask)

    #== movie ==#
    if movie==True:
        dir2 = run + '_MLD/movie_NCs'
        if not os.path.exists(dir2):
            os.makedirs(dir2)
        for i in range(num_files):
            date = str(MLD.time_counter[i].to_numpy())[0:10]
            MLD.isel(time_counter=i).to_netcdf(dir2 + '/' + run + 'MLD_map_' + mask_choice + '_' + date + '.nc')
        return

    #== non-movie plots ==#
    if movie==False:
  
        #max MLD
        maxMLD_col = MLD.max(dim=['time_counter'], skipna=True) #max MLD in each column during the whole period (i.e., for mapping reasons)
        maxMLD_region = MLD.max(dim=['y_grid_T','x_grid_T'], skipna=True) #max MLD in the masked region for each time-step (i.e., for time-plotting reasons)

        #average MLD
        areas = DS.e1t*DS.e2t
        areas = areas.isel(deptht = 0)
        avgArea = areas.mean(dim=['y_grid_T','x_grid_T'])
        weights = areas/avgArea 
        weights = weights.fillna(0)
        MLD = MLD.weighted(weights)
        avgMLD_col = MLD.mean(dim='time_counter',skipna=True) #average MLD in each column during the whole period
        avgMLD_region = MLD.mean(dim=['y_grid_T','x_grid_T'],skipna=True) #average MLD in the masked region for each time-step 

        #saving
        maxMLD_col.to_netcdf(run + '_MLD/' + run + '_max_MLD_somxlts_map_' + mask_choice + '.nc')
        maxMLD_region.to_netcdf(run + '_MLD/' + run + '_max_MLD_somxlts_time_plot_' + mask_choice + '.nc')
        avgMLD_col.to_netcdf(run + '_MLD/' + run + '_avg_MLD_somxlts_map_' + mask_choice + '.nc')
        avgMLD_region.to_netcdf(run + '_MLD/' + run + '_avg_MLD_somxlts_time_plot_' + mask_choice + '.nc')
    
    print(mask_choice)

if __name__ == '__main__':
    for mask in ['LS2k','LS','LSCR','LS3k']:
        MLD(run='EPM155',mask_choice=mask,movie=False)
