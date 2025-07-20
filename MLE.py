#Looks at the MLE streamfunction
#Currently makes maps; doesn't seem useful at this stage to look at time series
#Rowan Brown
#9 Feb 2024

import numpy as np
import pandas as pd
import xarray as xr
import os
import LSmap
from datetime import datetime

def MLE(run,mask_choice,movie=False):

    ##################################################################################################################
    #== creating directory if doesn't already exist ==#
    dir = run + '_MLE/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    ##################################################################################################################
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
    
    ##################################################################################################################
    #== opening model output ==# 
    gridT_txt = run + '_filepaths/' + run + '_gridT_filepaths.txt' #text file of paths to non-empty model output
    with open(gridT_txt) as f: lines = f.readlines() #open the text files
    filepaths_gridT = [line.strip() for line in lines]#[850:870:2]#[818]#[870:873] #get lists of the .nc output filepaths
    filepaths_gridT = [fp for fp in filepaths_gridT if (datetime(2012,1,1) <= datetime.strptime(fp[-20:-9],'y%Ym%md%d') <= datetime(2017,12,31))]
    num_files = len(filepaths_gridT)
    preprocess_gridT = lambda ds: ds[['e3t','MLE Lf','i-mle streamfunction','j-mle streamfunction']] #specify veriable(s) to retrieve
    DS = xr.open_mfdataset(filepaths_gridT,preprocess=preprocess_gridT) #open the files 

    ##################################################################################################################
    #== calculating the magnitude of the streamfunction ==#
    iMLE = DS['i-mle streamfunction'].interp(x_grid_U=DS.x_grid_U-0.5).drop_vars(['x_grid_U','nav_lat_grid_U','nav_lon_grid_U']).rename({'depthu':'deptht','x_grid_U':'x_grid_T','y_grid_U':'y_grid_T'})
    jMLE = DS['j-mle streamfunction'].interp(y_grid_V=DS.y_grid_V-0.5).drop_vars(['y_grid_V','nav_lat_grid_V','nav_lon_grid_V']).rename({'depthv':'deptht','x_grid_V':'x_grid_T','y_grid_V':'y_grid_T'})
    DS['Psi'] = (iMLE**2 + jMLE**2)**0.5
    DS = DS.drop_vars(['i-mle streamfunction','j-mle streamfunction','nav_lon_grid_U','nav_lat_grid_U','nav_lon_grid_V','nav_lat_grid_V','depthu','depthv'])
    
    #FOR CHECKING COORDS
    #print(iMLE.nav_lat_grid_U[300,200].to_numpy())
    #print(DS.nav_lat_grid_T[300,200].to_numpy())
    #print(DS.nav_lat_grid_T[300,200].to_numpy())
    #print(DS.nav_lon_grid_T[300,200].to_numpy())
    #print(DS.nav_lat_grid_V[300,200].to_numpy())
    #print(DS.nav_lat_grid_V[299,200].to_numpy())
    #print(DS.nav_lon_grid_V[300,200].to_numpy())

    ##################################################################################################################
    #== applying masks ==#
    DS[['e1t','e2t']] = e1t,e2t #add T cell dimensions as variables; PROBABLY NOT NECESSARY FOR PSI, Lf
    DS = DS.where(tmask == 1) #apply tmask (ie masking bathy); ALSO PROBABLY NOT NECESSARY FOR PSI AND Lf
    if mask_choice == 'LSCR' or mask_choice == 'LS2k' or mask_choice == 'LS' or mask_choice == 'LS3k': #apply mask
        DS.coords['mask'] = mask
        DS = DS.where(mask == 1, drop=True)
        DS = DS.drop_vars(['mask','time_centered'])
   
    #just lookin' at data, for testing purposes
    #print(filepaths_gridT)
    #print(DS['i-mle streamfunction'])#[1,2,200:205,300:305].to_numpy()) #MLE Lf
    ##iMLE = DS['i-mle streamfunction'][0,:,:,:].rename({'y_grid_U':'y_grid_T','x_grid_U':'x_grid_T','depthu':'deptht'}).drop_vars(['nav_lon_grid_U','nav_lat_grid_U']).max(dim='deptht')
    ##jMLE = DS['j-mle streamfunction'][0,:,:,:].rename({'y_grid_V':'y_grid_T','x_grid_V':'x_grid_T','depthv':'deptht'}).drop_vars(['nav_lon_grid_V','nav_lat_grid_V']).max(dim='deptht')
    ##MLE = (iMLE**2 + jMLE**2)**0.5
    ##MLD = DS['somxlts'][0,:,:]
    ##Lf = DS['MLE Lf'][0,:,:]

    ##masking shelves
    ##NOTE: bathy is masked to avoid skewed understandings/results from the on-shelf values this section could be commented out if needed 
    #bottom_slice = DS_d.vosaline.isel(deptht = -1).isel(time_counter = 0)
    #bottom_slice_bool = bottom_slice.notnull()
    #shelf_mask, temp = xr.broadcast(bottom_slice_bool, DS_d.vosaline.isel(time_counter=0))
    #DS_d = DS_d.where(shelf_mask)

    ##################################################################################################################
    #== Psi calcs ==#
    DS['Psi max'] = DS['Psi'].max(dim='deptht')
   
    #Curious about the minimum in Lf (def shouldn't ever be below 200 m, and even this is very small)
    #print(DS['MLE Lf'].min().to_numpy())
    #quit()

    #== movie ==#
    if movie==True:
        dir2 = run + '_MLE/movie_NCs'
        if not os.path.exists(dir2):
            os.makedirs(dir2)
        for i in range(num_files):
            date = str(DS.time_counter[i].to_numpy())[0:10]
            DS['Psi max'].isel(time_counter=i).to_netcdf(dir2 + '/' + run + 'MLE_map_' + mask_choice + '_' + date + '.nc')
        return

    #     EKE['EKE_per_m2_in_J'] = EKE.EKE.sum(dim='depth',skipna=True)/(EKE.areas.isel(depth=0).drop_vars('depth'))
    #     EKE_max_EKE_per_m2_in_J_each_year_AS_OCT = EKE.EKE_per_m2_in_J.resample(time_counter='AS-OCT').max(dim='time_counter',skipna=True)#.drop_vars('mask')
    #     #print(EKE)
    #     #print(EKE['EKE_per_m2_in_J'])
    #     #print(EKE_max_EKE_per_m2_in_J_each_year_AS_OCT)
    #     #
    #     #quit()
    #     #EKE_max_EKE_per_m2_in_J_each_year_AS_OCT.to_netcdf(run + '_EKE/' + run + '_EKE_map_THESIS_' + mask_choice + str(d) + '_window'+str(window)+'.nc') 
    #     EKE['EKE_in_region_in_J'] = EKE.EKE.sum(dim=['depth','x','y'],skipna=True)
    #     EKE.EKE_in_region_in_J.to_netcdf(run + '_EKE/' + run + '_EKE_timePlot_THESIS_' + mask_choice + str(d) + '_window'+str(window)+'.nc')

    #== non-movie plots ==#
    if movie==False: 
        DS['vol'] = DS.e1t*DS.e2t*DS.e3t
        #DS['weights'] = DS['vol']/DS['vol'].mean(dim=['x_grid_T','y_grid_T','deptht'])
        DS['Psi_weighted'] = DS['Psi']/DS['vol'] #units become m3/s/m3
        DS['Psi_weighted, max in depth'] = DS['Psi_weighted'].max(dim='deptht')
        psi_yearly_max_map = DS['Psi_weighted, max in depth'].resample(time_counter='AS-OCT').max(dim='time_counter',skipna=True)#.drop_vars('mask')
        psi_yearly_max_map.to_netcdf(run + '_MLE/' + run + '_MLE_yearly_max_map_THESIS_' + mask_choice + '_perVol.nc')
        DS['Psi_weighted'].mean(dim=['deptht','x_grid_T','y_grid_T'],skipna=True).to_netcdf(run + '_MLE/' + run + '_mean_MLE_mean_over_time_' + mask_choice + '_perVol.nc')
        
        #DS['Psi, mean in time, max in column'].to_netcdf(run + '_MLE/' + run + '_max_MLE_mean_over_time_map_' + mask_choice + '.nc')

    ##minmax = LSmap.xrLSminmax(MLD,mask.nav_lat_grid_T,mask.nav_lon_grid_T)
    ##LSmap.LSmap(MLD,DS.nav_lon_grid_T,DS.nav_lat_grid_T,minmax,'$m$','Mixed layer depth, 21 Mar 2013\n(ts criteria)','MLD_TEST')
    ##
    ##minmax = LSmap.xrLSminmax(Lf,mask.nav_lat_grid_T,mask.nav_lon_grid_T)
    ##LSmap.LSmap(Lf,DS.nav_lon_grid_T,DS.nav_lat_grid_T,minmax,'$m$?','Mixed layer front scale, $L_f$, 21 Mar 2013','Lf_on_streamfunction_test_day_fig')

    print('Done')

if __name__ == '__main__':
    #MLE('EPM155','LS',movie=False)
    MLE('EPM156','LS3k')
    MLE('EPM155','LS3k')
    #MLE('EPM155','LS3k')
    #MLE('EPM156','LS3k')
