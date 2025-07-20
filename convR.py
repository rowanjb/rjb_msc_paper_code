#Produces convective resistance
#Rowan Brown
#10 Jul 2023

import xarray as xr
import os
import numpy as np
import density

def convR(run, mask_choice, movie=False):

    #== creating directory if doesn't already exist ==#
    dir = run + '_convR/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    #== masks ==#
    with xr.open_dataset('masks/ANHA4_mesh_mask.nc') as DS: #mask for land, bathymetry, etc. and horiz. grid dimensions
        tmask = DS.tmask[0,:,:,:].rename({'z': 'deptht', 'y': 'y_grid_T', 'x': 'x_grid_T'}) #DataArray with dims (t: 1, z: 50, y: 800, x: 544) 
        e1t = DS.e1t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
        e2t = DS.e2t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
    if mask_choice == 'LS2k': #mask for 2000m depth interior area
        mask = xr.open_dataarray('masks/mask_LS_2k.nc').astype(int)
    elif mask_choice == 'LS': #mask for entire LS region
        mask = xr.open_dataarray('masks/mask_LS.nc').astype(int)
    elif mask_choice == 'LSCR': #mask for LS convection region
        mask = xr.open_dataset('masks/ARGOProfiles_mask.nc').tmask.astype(int).rename({'x':'x_grid_T','y':'y_grid_T'})
    else:
        print("Y'all didn't choose a mask")
        quit()

    #== opening the output files ==#
    gridT_txt = run + '_filepaths/' + run + '_gridT_filepaths.txt' #text file of paths to non-empty model output
    with open(gridT_txt) as f: lines = f.readlines() #open the text files and get lists of the .nc output filepaths
    filepaths_gridT = [line.strip() for line in lines]
    preprocess_gridT = lambda ds: ds[['e3t','votemper','vosaline','somxl010']] #open files and get desired variables only
    DS = xr.open_mfdataset(filepaths_gridT,preprocess=preprocess_gridT)
    DS[['e1t','e2t']] = e1t,e2t #add T cell dimensions as variables

    #== apply masks ==#
    DS = DS.where(tmask == 1) #tmask (ie masking bathy)
    if mask_choice == 'LSCR' or mask_choice == 'LS2k' or mask_choice == 'LS':
        DS.coords['mask'] = mask
        DS = DS.where(DS.mask == 1, drop=True)

    #== calculations (note: these are based on the Introducing LAB60... paper from Clark ==#
    g = 9.80665 #gravity
    for d in [50, 200, 1000, 2000]: #loop through the 4 depths  
        DS_d = DS.where(DS.deptht < d, drop=True) #drop values below specified depth

        #note: there are two main ideas below: "col" refers to the idea that we're looking at water-columnwise averages, ie so we can make maps later. On 
        #the other hand, "region" refers to regionwise averages, so that we can make time plots later.

        #masking shelves
        #NOTE: bathy is masked to avoid skewed understandings/results from the on-shelf values this section could be commented out if needed 
        bottom_slice = DS_d.votemper.isel(deptht = -1).isel(time_counter = 0)
        bottom_slice_bool = bottom_slice.notnull()
        shelf_mask, temp = xr.broadcast(bottom_slice_bool, DS_d.votemper.isel(time_counter=0))
        DS_d = DS_d.where(shelf_mask)

        #pre-calculations
        areas = DS_d.e1t*DS_d.e2t
        areas = areas.isel(deptht=0) #only need one slice
        area = areas.sum(dim=['x_grid_T','y_grid_T'])
        Th = DS_d.votemper.isel(deptht=-1) #temperature values in the deepest level
        Sh = DS_d.vosaline.isel(deptht=-1) #salinity values in the deepest level
        refDens = density.density(Sh,Th) #potential densities at (around) h
        dens = density.density(DS_d.vosaline,DS_d.votemper) #densities in each cell from surface to h

        #following the equation from "Introducing LAB60..."
        term1 = refDens*(DS_d.e3t.sum(dim='deptht')) #used e3t.sum instead of depth[-1] because term2 is integrated using e3t too
        term2 = dens*DS_d.e3t
        term2 = term2.sum(dim='deptht')
        integrand = term1 - term2
        convR_col = integrand*g #this is the representative value in each column, units are J/m3 or kg/m s2 (didn't both multiplying and dividing by area)
        convR_timePlot = convR_col*areas*DS_d.deptht.isel(deptht=-1) #multiply the J/m3 value by column area and depth to get the total J  #/area #multiplying by cell area and dividing by total area (i.e., essentially weighting the values)
        convR_timePlot = convR_timePlot.sum(dim=['x_grid_T','y_grid_T']) #summing to get the total J in in the masked area #summing the weighted values spatially, which gives the mean convR in the mask
        convR_col = convR_col.mean(dim='time_counter') #taking the mean in time, which gives the mean convR in each column throughout the run
        
        #dropping deptht dim
        convR_timePlot = convR_timePlot.drop_vars('deptht')
        convR_col = convR_col.drop_vars('deptht')

        #saving
        convR_col.to_netcdf(run + '_convR/' + run + '_convR_map_TEST_loop_' + mask_choice + str(d) + '.nc')
        convR_timePlot.to_netcdf(run + '_convR/' + run + '_sumConvR_plot_TEST_loop_' + mask_choice + str(d) + '.nc') 

        #== movie ==#
        #if movie==True:
           # dir2 = run + '_MLD/movie_NCs'
           # if not os.path.exists(dir2):
          #      os.makedirs(dir2)
         #   for i in range(num_files):
         #       date = str(MLD.time_counter[i].to_numpy())[0:10]
         #       MLD.isel(time_counter=i).to_netcdf(dir2 + '/' + run + 'MLD_map_' + mask_choice + '_' + date + '.nc')
         #   return

        #== non-movie plots ==#
        #if movie==False:

            ##max MLD
            #maxMLD_col = MLD.max(dim=['time_counter'], skipna=True) #max MLD in each column during the whole period (i.e., for mapping reasons)
            #maxMLD_region = MLD.max(dim=['y_grid_T','x_grid_T'], skipna=True) #max MLD in the masked region for each time-step (i.e., for time-plotting reasons)

            ##average MLD
            #areas = DS.e1t*DS.e2t
            #areas = areas.isel(deptht = 0)
            #avgArea = areas.mean(dim=['y_grid_T','x_grid_T'])
            #weights = areas/avgArea #CHECK THAT THIS IS RIGHT!!!!!!!!!!!!!!!!!!!!!!!!!!
            #weights = weights.fillna(0)
            #MLD = MLD.weighted(weights)
            #avgMLD_col = MLD.mean(dim='time_counter',skipna=True) #average MLD in each column during the whole period
            #avgMLD_region = MLD.mean(dim=['y_grid_T','x_grid_T'],skipna=True) #average MLD in the masked region for each time-step 

            ##saving
            #maxMLD_col.to_netcdf(run + '_MLD/' + run + '_max_MLD_map_' + mask_choice + '.nc')
           # maxMLD_region.to_netcdf(run + '_MLD/' + run + '_max_MLD_time_plot_' + mask_choice + '.nc')
          #  avgMLD_col.to_netcdf(run + '_MLD/' + run + '_avg_MLD_map_' + mask_choice + '.nc')
         #   avgMLD_region.to_netcdf(run + '_MLD/' + run + '_avg_MLD_time_plot_' + mask_choice + '.nc')

        #print('test')

if __name__ == '__main__':
    convR(run='EPM158',mask_choice='LS',movie=False)
