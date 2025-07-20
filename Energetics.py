#Energetics for ANHA4 runs
#Rowan Brown
#May 7, 2024

import numpy as np 
import xarray as xr
import os
from datetime import datetime
import density

def energy(calc_type,run,mask_choice,d,window,test_run):
   
    #creating directory if doesn't already exist
    dir = run + '_EKE/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    #== OPENING AND INITIAL PROCESSING OF THE NETCDF MODEL OUTPUT FILES ==#

    #these are text files with lists of all non-empty model outputs (made with 'filepaths.py')
    #add these suffixes for EPM157
    gridU_txt = run + '_filepaths/' + run + '_gridU_filepaths.txt' #_filepaths_may2024.txt'
    gridV_txt = run + '_filepaths/' + run + '_gridV_filepaths.txt' #_filepaths_may2024.txt'
    gridT_txt = run + '_filepaths/' + run + '_gridT_filepaths.txt' #_filepaths_may2024.txt'
    gridW_txt = run + '_filepaths/' + run + '_gridW_filepaths.txt' #_filepaths_may2024.txt'

    #open the text files and get lists of the .nc output filepaths
    years_we_care_about = list(range(2008,2019))
    with open(gridU_txt) as f: lines = f.readlines()
    filepaths_gridU = [line.strip() for line in lines]#[1000:1010]
    filepaths_gridU = [fp for fp in filepaths_gridU if (datetime(2007,6,1) <= datetime.strptime(fp[-20:-9],'y%Ym%md%d') <= datetime(2019,3,1))]
    with open(gridV_txt) as f: lines = f.readlines()
    filepaths_gridV = [line.strip() for line in lines]#[1000:1010]
    filepaths_gridV = [fp for fp in filepaths_gridV if (datetime(2007,6,1) <= datetime.strptime(fp[-20:-9],'y%Ym%md%d') <= datetime(2019,3,1))]
    with open(gridT_txt) as f: lines = f.readlines()
    filepaths_gridT = [line.strip() for line in lines]#[1000:1010]
    filepaths_gridT = [fp for fp in filepaths_gridT if (datetime(2007,6,1) <= datetime.strptime(fp[-20:-9],'y%Ym%md%d') <= datetime(2019,3,1))]
    with open(gridW_txt) as f: lines = f.readlines()
    filepaths_gridW = [line.strip() for line in lines]#[1000:1010]
    filepaths_gridW = [fp for fp in filepaths_gridW if (datetime(2007,6,1) <= datetime.strptime(fp[-20:-9],'y%Ym%md%d') <= datetime(2019,3,1))]
  
    #shorten the list if we're just testing the script interactively 
    if test_run==True:
        filepaths_gridU = filepaths_gridU[500:520]
        filepaths_gridV = filepaths_gridV[500:520]
        filepaths_gridT = filepaths_gridT[500:520]
        filepaths_gridW = filepaths_gridW[500:520]

    #print lens to ensure they're the same (basically a simple manual check)
    print('Number of gridW files: ' + str(len(filepaths_gridW)))
    print('Number of gridT files: ' + str(len(filepaths_gridT)))
    print('Number of gridV files: ' + str(len(filepaths_gridV)))
    print('Number of gridU files: ' + str(len(filepaths_gridU)))

    #preprocessing (specifying variables that we need and ignoring the rest)
    #we don't need e3u e3v or e3w since we co-locate the vels. on the T grid
    preprocess_gridU = lambda ds: ds[['vozocrtx']]
    preprocess_gridV = lambda ds: ds[['vomecrty']]
    preprocess_gridW = lambda ds: ds[['vovecrtz']]
    if calc_type=='EKE' or calc_type=='T4': #i.e., if we don't need densities
        preprocess_gridT = lambda ds: ds[['e3t']]
    else: #i.e., if we need densities
        preprocess_gridT = lambda ds: ds[['e3t','votemper','vosaline']]

    #open gridT
    DST = xr.open_mfdataset(filepaths_gridT,preprocess=preprocess_gridT,engine="netcdf4")
    DST = DST.rename({'deptht': 'depth', 'y_grid_T': 'y', 'x_grid_T': 'x'})

    #== MASKS ==#

    #mask for land, bathymetry, etc. and horiz. grid dimensions
    with xr.open_dataset('masks/ANHA4_mesh_mask.nc') as DS:
        DST['e1t'] = DS.e1t[0,:,:]
        DST['e2t'] = DS.e2t[0,:,:]
        DST['e1u'] = DS.e1t[0,:,:] #need these two to calc gradients between grid T points
        DST['e2v'] = DS.e2t[0,:,:] # //
    if mask_choice == 'LS2k':
        with xr.open_dataarray('masks/mask_LS_2k.nc') as DS:
            DST = DST.assign_coords(mask=DS.astype(int).rename({'deptht':'depth','x_grid_T':'x','y_grid_T':'y'}))
    elif mask_choice == 'LS3k':
        with xr.open_dataarray('masks/mask_LS_3000.nc') as DS:
            DST = DST.assign_coords(mask=DS.astype(int).rename({'deptht':'depth','x_grid_T':'x','y_grid_T':'y'}))
    elif mask_choice == 'LS':
        with xr.open_dataarray('masks/mask_LS.nc') as DS:
            DST = DST.assign_coords(mask=DS.astype(int).rename({'deptht':'depth','x_grid_T':'x','y_grid_T':'y'}))
    elif mask_choice == 'LSCR':
        with xr.open_dataset('masks/ARGOProfiles_mask.nc') as DS:
            DST = DST.assign_coords(mask = DS.tmask.astype(int))
    else:
        print("Y'all didn't choose a mask")
        quit()
    print('Masks opened')

    #== INITIAL ANALYSES ==#

    #if you need w velocities
    if calc_type == 'T3' or calc_type == 'T1':
        DSW = xr.open_mfdataset(filepaths_gridW,preprocess=preprocess_gridW,engine="netcdf4")
        DSW = DSW.rename({'depthw': 'depth'}) #depth gets reset/interpolated onto deptht in few lines, so this is OK
        DSW = DSW.where(DST.mask.isel(depth=0).drop_vars('depth') == 1, drop=True).drop_vars(['mask','nav_lat_grid_T','nav_lon_grid_T'])
        DSW = DSW.interp(depth=DSW.depth+0.5) #Locating vel. on the T grid (necessary so that the same exact volumes are used to get the total energy conversion rates across T1--T4)
        DSW['depth'] = DST['depth']
        DSW = DSW.where(DSW.depth < d, drop=True)

    #if you're going to need horizontal velocities
    if calc_type=='EKE' or calc_type=='T2' or calc_type=='T4':
        DSU = xr.open_mfdataset(filepaths_gridU,preprocess=preprocess_gridU,engine="netcdf4")
        DSV = xr.open_mfdataset(filepaths_gridV,preprocess=preprocess_gridV,engine="netcdf4")
        DSU = DSU.rename({'depthu': 'depth'})
        DSV = DSV.rename({'depthv': 'depth'})
        if calc_type=='T2' or calc_type=='T4': #in this case, we need gradients and therefore cannot apply the mask yet
            DSU = DSU.where((DSU.x>100)&(DSU.x<250)&(DSU.y>300)&(DSU.y<500),drop=True)
            DSV = DSV.where((DSV.x>100)&(DSV.x<250)&(DSV.y>300)&(DSV.y<500),drop=True)
            if calc_type=='T4': #too many if statements---no time to fix!
                DSU = DSU.where(DSU.depth < d, drop=True)
                DSV = DSV.where(DSV.depth < d, drop=True)
        else: #EKE doesn't have gradients so we can apply the mask
            DSU = DSU.where(DST.mask.isel(depth=0).drop_vars('depth') == 1, drop=True).drop_vars(['mask','nav_lat_grid_T','nav_lon_grid_T'])
            DSV = DSV.where(DST.mask.isel(depth=0).drop_vars('depth') == 1, drop=True).drop_vars(['mask','nav_lat_grid_T','nav_lon_grid_T'])
            DSU = DSU.where(DSU.depth < d, drop=True)
            DSV = DSV.where(DSV.depth < d, drop=True)
        DSU = DSU.interp(x=DSU.x-0.5).drop_vars('x') #Co-locating vels. on the T grid
        DSV = DSV.interp(y=DSV.y-0.5).drop_vars('y') #Aug 31, 2023: x interp is updated from x+0.5 to x-0.5

    #some (final) initial processing on gridT (I'm assuming these "where" statements save computational cost)
    if calc_type=='T2' or calc_type=='T4': #i.e., keeping DST same size as DSU and DSV in the case that we need gradients 
        DST = DST.where((DST.x>100)&(DST.x<250)&(DST.y>300)&(DST.y<500),drop=True)
        if calc_type=='T4':
            DST = DST.where(DST.depth < d,drop=True)
    else: #i.e., if we're calculating EKE, T1, or T3 and therefore don't need gradients
        DST = DST.where(DST.mask == 1, drop=True)
        DST = DST.where(DST.depth < d,drop=True)

    #constants (etc.) needed in the following calcs 
    rho_0 = 1025
    #window = 5
    g = 9.834

    print('Pre-analyses completed')

    if calc_type=='EKE': ##################################################################################################################
        print('Calculating EKE')
        #EKE is calculated based on the methods from: Martínez-Moreno et al. - Global changes in oceanic mesoscale currents over the satellite altimetry record
        # => EKE = (1/2) * density_0 * (u'**2 + v'**2) [J/m**3] where u′ = u − u_mean and v′ = v − v_mean
        #recall, units are J = kg m**2 / s**2, density = kg / m**3, and vel**2 = m**2 / s**2, so density*vel**2 = kg / m s**2 = J / m**3
        #so, integrating  over volume gives total Joules

        #EKE calculations
        DSU_bar_sqr = (DSU-DSU.rolling(time_counter=window,center=True).mean())**2 
        DSV_bar_sqr = (DSV-DSV.rolling(time_counter=window,center=True).mean())**2
        EKE = (1/2) * rho_0 * (DSU_bar_sqr.vozocrtx + DSV_bar_sqr.vomecrty) 
        EKE = EKE.rename('EKE')
        
        #add grid T lat and lons and other useful quantities as coordinates (useful for plotting later)
        EKE = EKE.assign_coords(nav_lat_grid_T=DST.nav_lat_grid_T, nav_lon_grid_T=DST.nav_lon_grid_T,mask=DST.mask,e1t=DST.e1t,e2t=DST.e2t,e3t=DST.e3t)
        EKE = EKE.reset_coords(names=['e1t','e2t','e3t']) #change cell dims from coordinates to variables

        #EKE in J within each cell
        EKE['areas'] = EKE.e1t*EKE.e2t
        EKE['EKE'] = EKE.EKE*EKE.areas*EKE.e3t #EKE.EKE was originally in units J/m**3, so multiply by volue

        #masking the shelves, keeping wherever the bottom slice of EKE (across all times) is above zero
        #it mightn't be necessary to mask the shelves, since EKE is summed, not averaged in depth
        #but I guess if you don't, you could end up with slightly misleading maps
        #time counter is 10 because the first few times will be 0/nan due to the rolling window average 
        #EKE = EKE.where(EKE.EKE.isel(depth=-1).isel(time_counter=int(window/2)).drop_vars(['depth','time_counter'])>0,drop=True)
        
        #saving
        #print(EKE.EKE.sum(dim='depth',skipna=True))
        #print(EKE.areas.isel(depth=0).drop_vars('depth'))
        EKE['EKE_per_m2_in_J'] = EKE.EKE.sum(dim='depth',skipna=True)/(EKE.areas.isel(depth=0).drop_vars('depth'))
        EKE_max_EKE_per_m2_in_J_each_year_AS_OCT = EKE.EKE_per_m2_in_J.resample(time_counter='AS-OCT').mean(dim='time_counter',skipna=True)#.drop_vars('mask')
        EKE_max_EKE_per_m2_in_J_each_year_AS_OCT.to_netcdf(run + '_EKE/' + run + '_mean_EKE_map_THESIS_' + mask_choice + str(d) + '_window'+str(window)+'.nc') 
        #EKE['EKE_in_region_in_J'] = EKE.EKE.sum(dim=['depth','x','y'],skipna=True)
        #EKE.EKE_in_region_in_J.to_netcdf(run + '_EKE/' + run + '_EKE_timePlot_THESIS_' + mask_choice + str(d) + '_window'+str(window)+'.nc')
        
        #Closing statements
        print('EKE calculated')
        DST.close()
        DSU.close()
        DSV.close()    

    elif calc_type=='T3': ##################################################################################################################
        print('Calculating T3 conversion')
        #rate of energy conversion from eddy available potential energy to eddy kinetic energy 
        #Adam says unit is W/m3, so multiply by volume to get total rate 

        DSW['rho'] = density.density(DST.vosaline,DST.votemper) 
        DSW['rho_anom'] = DSW['rho']-DSW['rho'].rolling(time_counter=window,center=True).mean()
        DSW['w_anom'] = DSW['vovecrtz'] - DSW['vovecrtz'].rolling(time_counter=window,center=True).mean()
        DSW['rho_anom_X_w_anom'] = DSW['rho_anom']*DSW['w_anom'] 
        DSW['areas'] = DST.e1t*DST.e2t
        DSW['T3_in_each_cell'] = (-1)*DSW['areas']*DST.e3t*g*DSW['rho_anom_X_w_anom'].rolling(time_counter=window,center=True).mean() 
        T3_max_per_col_per_year_AS_OCT = DSW['T3_in_each_cell'].sum(dim='depth',skipna=True).resample(time_counter='AS-OCT').mean(dim='time_counter',skipna=True)
        T3_max_per_m2_per_year_AS_OCT = T3_max_per_col_per_year_AS_OCT/(DSW.areas.isel(depth=0)) #units become W/m2
        T3_max_per_m2_per_year_AS_OCT = T3_max_per_m2_per_year_AS_OCT.drop_vars(['mask','depth','nav_lat','nav_lon'])
        T3_max_per_m2_per_year_AS_OCT.to_netcdf(run + '_EKE/' + run + '_T3_mean_map_THESIS_' + mask_choice + str(d) + '_window'+str(window)+'.nc')
        #DSW['T3_in_each_cell'].sum(dim=['depth','x','y']).drop_vars('time_centered').to_netcdf(run + '_EKE/' + run + '_T3_timePlot_THESIS_' + mask_choice + str(d) + '_window'+str(window)+'.nc')
        
        #Closing statements
        print('T3 calculated')
        DST.close()
        DSW.close()

    elif calc_type=='T1': ##################################################################################################################
        print('Calculating T1 conversion')
        #rate of energy conversion from mean available potential energy to mean kinetic energy 

        DSW['rho'] = density.density(DST.vosaline,DST.votemper)
        DSW['areas'] = DST.e1t*DST.e2t
        DSW['rho_bar'] = DSW['rho'].rolling(time_counter=window,center=True).mean()
        DSW['w_bar'] = DSW['vovecrtz'].rolling(time_counter=window,center=True).mean()
        DSW['T1_in_each_cell'] = (-1)*g*DSW['areas']*DST.e3t*DSW['rho_bar']*DSW['w_bar'] 
        DSW = DSW.drop_vars(['nav_lat','nav_lon','mask','time_centered'])
        T1_per_m2 = DSW['T1_in_each_cell'].sum(dim='depth',skipna=True)/(DSW.areas.isel(depth=0).drop_vars('depth'))
        T1_max_per_m2_per_year_AS_OCT = T1_per_m2.resample(time_counter='AS-OCT').mean(dim='time_counter',skipna=True)
        T1_max_per_m2_per_year_AS_OCT.to_netcdf(run + '_EKE/' + run + '_T1_mean_map_THESIS_' + mask_choice + str(d) + '_window'+str(window)+'.nc')
        #DSW['T1_in_each_cell'].sum(dim=['depth','x','y']).to_netcdf(run + '_EKE/' + run + '_T1_timePlot_THESIS_' + mask_choice + str(d) + '_window'+str(window)+'.nc')
        
        #Closing statements
        print('T1 calculated')
        DST.close()
        DSW.close()

    elif calc_type=='T4': ##################################################################################################################
        print('Calculating T4 conversion')
        #rate of energy conversion from mean kinetic energy to eddy kinetic 
        #eq "BT" from Gou et al. --- Variability of Eddy Formation...
        #as with the other conversion terms, unit is W/m3, so multiply by volume to get total energy conversion in masked region
        #this function and the next (T2) could likely be optimized computationally but they should also work as-is

        #terms of the eq
        DSU['u_bar'] = DSU['vozocrtx'].rolling(time_counter=window,center=True).mean()
        DSV['v_bar'] = DSV['vomecrty'].rolling(time_counter=window,center=True).mean()
        DSU['u_prime'] = DSU['vozocrtx'] - DSU['u_bar']
        DSV['v_prime'] = DSV['vomecrty'] - DSV['v_bar']
        DSU['u_prime_sqr_bar'] = DSU['u_prime']**2
        DSU['u_prime_sqr_bar'] = DSU['u_prime_sqr_bar'].rolling(time_counter=window,center=True).mean()
        DSV['v_prime_sqr_bar'] = DSV['v_prime']**2
        DSV['v_prime_sqr_bar'] = DSV['v_prime_sqr_bar'].rolling(time_counter=window,center=True).mean()
        DSU['uv_prime_bar'] = DSU['u_prime']*DSV['v_prime']
        DSU['uv_prime_bar'] = DSU['uv_prime_bar'].rolling(time_counter=window,center=True).mean()
        
        #gradients
        #use e1u and e2v since these measure the zonal and meridional distances between t-points  
        d_u_bar_in_x = DSU['u_bar'].diff(dim='x',label='lower').drop_vars(['nav_lat','nav_lon'])/DST.e1u.isel(x=slice(0,-1)) # (DSU['u_bar'].isel(x=slice(0,-1)) - DSU['u_bar'].isel(x=slice(1,None)))/DST.e1u.isel(x=slice(0,-1)) 
        d_v_bar_in_y = DSV['v_bar'].diff(dim='y',label='lower').drop_vars(['nav_lat','nav_lon'])/DST.e2v.isel(y=slice(0,-1)) #.isel(y=slice(0,-1)) - DSV['v_bar'].isel(y=slice(1,None)))/DST.e2v.isel(y=slice(0,-1))
        d_u_bar_in_y = DSU['u_bar'].diff(dim='y',label='lower').drop_vars(['nav_lat','nav_lon'])/DST.e2v.isel(y=slice(0,-1)) #.sel(y=slice(0,-1)) - DSU['u_bar'].isel(y=slice(1,None)))/DST.e2v.isel(y=slice(0,-1))
        d_v_bar_in_x = DSV['v_bar'].diff(dim='x',label='lower').drop_vars(['nav_lat','nav_lon'])/DST.e1u.isel(x=slice(0,-1)) # isel(x=slice(0,-1)) - DSV['v_bar'].isel(x=slice(1,None)))/DST.e1u.isel(x=slice(0,-1))

        #making sure all terms are the correct size
        DSU = DSU.isel(x=slice(0,-1),y=slice(0,-1))
        DSV = DSV.isel(x=slice(0,-1),y=slice(0,-1))
        d_u_bar_in_x = d_u_bar_in_x.isel(y=slice(0,-1))
        d_v_bar_in_y = d_v_bar_in_y.isel(x=slice(0,-1))
        d_u_bar_in_y = d_u_bar_in_y.isel(x=slice(0,-1))
        d_v_bar_in_x = d_v_bar_in_x.isel(y=slice(0,-1))
        DST = DST.isel(x=slice(0,-1),y=slice(0,-1))

        #print(DSU['u_prime'].isel(time_counter=2, depth=10, y=30, x=25).to_numpy())
        #print(DSU['u_prime'].isel(time_counter=2, depth=10, y=30, x=26).to_numpy())
        #print(DST['e1t'].isel(depth=10, y=30, x=25).to_numpy())
        #print(d_u_bar_in_x.isel(time_counter=2, depth=10, y=30, x=25).to_numpy())

        #calculating BT / T4, masking, saving
        BT = (-1)*rho_0*( DSU['u_prime_sqr_bar']*d_u_bar_in_x + DSV['v_prime_sqr_bar']*d_v_bar_in_y + DSU['uv_prime_bar']*( d_u_bar_in_y + d_v_bar_in_x ) )
        BT = BT.where(DST.mask==1,drop=True).drop_vars(['nav_lat','nav_lon','time_centered','mask'])
        DST = DST.where(DST.mask==1,drop=True)
        areas = DST.e1t*DST.e2t
        BT_per_cell = BT*areas*DST.e3t
        BT_per_m2 = BT_per_cell.sum(dim='depth')/areas.isel(depth=0).drop_vars('depth')
        BT_max_per_m2_per_year_AS_OCT = BT_per_m2.resample(time_counter='AS-OCT').mean(dim='time_counter',skipna=True)

        #print(DST.nav_lat_grid_T.isel(x=20,y=20).to_numpy())
        #print(DST.nav_lon_grid_T.isel(x=20,y=20).to_numpy())
        #print(BT_max_per_m2_per_year_AS_OCT.nav_lat_grid_T.isel(x=20,y=20).to_numpy())
        #print(BT_max_per_m2_per_year_AS_OCT.nav_lon_grid_T.isel(x=20,y=20).to_numpy())
        #quit()
        BT_max_per_m2_per_year_AS_OCT.drop_vars('mask').to_netcdf(run + '_EKE/' + run + '_T4_mean_map_THESIS_' + mask_choice + str(d) + '_window'+str(window)+'.nc')
        #BT_per_cell.sum(dim=['depth','x','y']).to_netcdf(run + '_EKE/' + run + '_T4_timePlot_THESIS_' + mask_choice + str(d) + '_window'+str(window)+'.nc')

        #Closing statements
        print('T4 calculated')
        DST.close()
        DSU.close()
        DSV.close()

    elif calc_type=='T2': ##################################################################################################################
        print('Calculating T2 conversion')
        #rate of energy conversion from eddy available potential energy to mean availabile potential energy
        #eq "BC" from Gou et al. --- Variability of Eddy Formation...

        #print(DST)
        #print(DSU)
        #print(DSV)
        #print(DST.votemper.mean().to_numpy())
        #print(DSU.vozocrtx.mean().to_numpy())
        #print(DSV.vomecrty.mean().to_numpy())

        #need to calc buoyancy frequency, so need thickness of w-cells, ie vertical distance from t-point to t-point
        #in this case e3w[0:] gives the distances between the 1st and 2nd t-points, 2nd and 3rd t-points, and so on
        preprocess_gridW = lambda ds: ds[['e3w']]
        e3w = xr.open_mfdataset(filepaths_gridW,preprocess=preprocess_gridW,engine="netcdf4")
        e3w = e3w.where((e3w.x>100)&(e3w.x<250)&(e3w.y>300)&(e3w.y<500),drop=True).rename({'depthw':'depth','nav_lat':'nav_lat_grid_T','nav_lon':'nav_lon_grid_T'})
        e3w['depth'] = DST.depth #need to remap so that you can use as denominator in gradient eq. with numerator from DST; lat and lon are fine, because DSW lats and lons are already same as DST's
        DST['e3w'] = e3w['e3w']#.rename({'depthw':'depth','nav_lat':'nav_lat_grid_T','nav_lon':'nav_lon_grid_T'})

        #print('datasets and dataarrays:')
        #print(e3t)
        #print(DST)
        #print(DSU)
        #print(DSV)

        #print('simple calcs:')
        #print(DST.votemper.mean().to_numpy())
        #print(DSU.vozocrtx.mean().to_numpy())
        #print(DSV.vomecrty.mean().to_numpy())

        #simple "pre-calcs"
        DST['rho'] = density.density(DST.vosaline,DST.votemper)
        DST['b'] = (-1)*g*(DST['rho'] - rho_0)/rho_0
        DST['areas'] = DST.e1t*DST.e2t
        DST['rho_bar'] = DST['rho'].rolling(time_counter=window,center=True).mean()
        DST['rho_prime'] = DST['rho'] - DST['rho_bar']
        DSU['u_prime'] = DSU['vozocrtx'] - DSU['vozocrtx'].rolling(time_counter=window,center=True).mean()
        DSV['v_prime'] = DSV['vomecrty'] - DSV['vomecrty'].rolling(time_counter=window,center=True).mean()
        DSU['urho_prime'] = DSU['u_prime']*DST['rho_prime']
        DSV['vrho_prime'] = DSV['v_prime']*DST['rho_prime']
        DSU['urho_prime_bar'] = DSU['urho_prime'].rolling(time_counter=window,center=True).mean()
        DSV['vrho_prime_bar'] = DSV['vrho_prime'].rolling(time_counter=window,center=True).mean()

        #print('more complicated calcs completed (but no divisions)')


        #gradients
        N2 = (-1)*DST['b'].diff(dim='depth',label='lower')/DST['e3w'].isel(depth=slice(0,-1))# - DST['b'].isel(depth=slice(1,None)))#/DST['e3w'].isel(depth=slice(0,-1))
        rho_bar_in_x = DST['rho_bar'].diff(dim='x',label='lower')/DST.e1u.isel(x=slice(0,-1)) #.drop_vars(['nav_lon','nav_lat']) 
        rho_bar_in_y = DST['rho_bar'].diff(dim='y',label='lower')/DST.e2v.isel(y=slice(0,-1))

        #print(N2)
        

        #making sure all terms are the correct size
        N2 = N2.isel(x=slice(0,-1),y=slice(0,-1)).where(N2.depth < d, drop=True)#.where(BC.mask==1,drop=True)
        DSU = DSU.isel(x=slice(0,-1),y=slice(0,-1)).where(DSU.depth < d, drop=True)#.where(BC.mask==1,drop=True)
        DSV = DSV.isel(x=slice(0,-1),y=slice(0,-1)).where(DSV.depth < d, drop=True)#.where(BC.mask==1,drop=True)
        DST = DST.isel(x=slice(0,-1),y=slice(0,-1)).where(DST.depth < d, drop=True)#.where(BC.mask==1,drop=True)
        rho_bar_in_x = rho_bar_in_x.isel(y=slice(0,-1)).where(rho_bar_in_x.depth < d, drop=True)#.where(BC.mask==1,drop=True)
        rho_bar_in_y = rho_bar_in_y.isel(x=slice(0,-1)).where(rho_bar_in_x.depth < d, drop=True)#.where(BC.mask==1,drop=True)

        #print(N2.isel(time_counter=5,x=5,y=5).to_numpy())
        #print(DSU['urho_prime_bar'])
        #print(rho_bar_in_x)
        #print(DSV['vrho_prime_bar'])
        #print(rho_bar_in_y)
        #quit()

        #calculating T2
        N2 = N2.where(N2!=0)
        BC = (-1)*((g**2)/(N2 * rho_0))*( DSU['urho_prime_bar']*rho_bar_in_x + DSV['vrho_prime_bar']*rho_bar_in_y )

        # masking, saving
        BC = BC.where(BC.mask==1,drop=True).drop_vars(['time_centered','mask'])
        #BC = BC.where(BC.depth < d,drop=True)
        print(BC)#.mean().to_numpy())
        #print(BC.mean(skipna=True).to_numpy())
        #quit()
        DST = DST.where(DST.mask==1,drop=True)
        areas = DST.e1t*DST.e2t
        BC_per_cell = BC*areas*DST.e3t
        #print(BC_per_cell.mean().to_numpy())
        #quit()
        print(BC_per_cell.isel(time_counter=5,depth=2,x=40,y=40).to_numpy())
        #quit()

        #print('check')
        BC_per_m2 = BC_per_cell.sum(dim='depth')/areas.isel(depth=0).drop_vars('depth')
        #print(BC_per_m2.isel(time_counter=1,x=1,y=1).to_numpy())
        #print('check')
        #print(areas.isel(depth=0))
        BC_max_per_m2_per_year_AS_OCT = BC_per_m2.resample(time_counter='AS-OCT').mean(dim='time_counter',skipna=True)
        BC_max_per_m2_per_year_AS_OCT.drop_vars('mask').to_netcdf(run + '_EKE/' + run + '_T2_mean_map_THESIS_' + mask_choice + str(d) + '_window'+str(window)+'.nc')
        #BC_per_cell.sum(dim=['depth','x','y']).to_netcdf(run + '_EKE/' + run + '_T2_timePlot_THESIS_' + mask_choice + str(d) + '_window'+str(window)+'.nc')

        #Closing statements
        print('T2 calculated')
        DST.close()
        DSU.close()
        DSV.close()
        e3w.close()

if __name__ == '__main__':
  
    run = 'EPM151'
    energy('EKE',run,mask_choice='LS',d=2000,window=21,test_run=False)
    #energy('T1',run,mask_choice='LS',d=2000,window=21,test_run=False)
    energy('T2',run,mask_choice='LS',d=2000,window=21,test_run=False)
    #energy('T3',run,mask_choice='LS',d=2000,window=21,test_run=False)
    energy('T4',run,mask_choice='LS',d=2000,window=21,test_run=False)
    energy('EKE',run,mask_choice='LS',d=200,window=21,test_run=False)
    #energy('T1',run,mask_choice='LS',d=200,window=21,test_run=False)
    energy('T2',run,mask_choice='LS',d=200,window=21,test_run=False)
    #energy('T3',run,mask_choice='LS',d=200,window=21,test_run=False)
    energy('T4',run,mask_choice='LS',d=200,window=21,test_run=False)
    quit()

    energy('T2','EPM157',mask_choice='LS',d=200,window=5,test_run=False)
    energy('T2','EPM157',mask_choice='LS',d=2000,window=5,test_run=False)
    energy('T2','EPM157',mask_choice='LS',d=200,window=21,test_run=False)
    energy('T2','EPM157',mask_choice='LS',d=2000,window=21,test_run=False)
    quit()

    energy('T1','EPM157',mask_choice='LS',d=200,window=5,test_run=False)
    energy('T1','EPM157',mask_choice='LS',d=2000,window=5,test_run=False)
    energy('T1','EPM157',mask_choice='LS',d=200,window=21,test_run=False)
    energy('T1','EPM157',mask_choice='LS',d=2000,window=21,test_run=False)
     

    energy('T3','EPM157',mask_choice='LS',d=200,window=5,test_run=False)
    energy('T3','EPM157',mask_choice='LS',d=2000,window=5,test_run=False)
    energy('T3','EPM157',mask_choice='LS',d=200,window=21,test_run=False)
    energy('T3','EPM157',mask_choice='LS',d=2000,window=21,test_run=False)
    quit()


    energy('T2','EPM151',mask_choice='LS',d=200,window=5,test_run=False)
    energy('T2','EPM152',mask_choice='LS',d=200,window=5,test_run=False)
    energy('T2','EPM155',mask_choice='LS',d=200,window=5,test_run=False)
    energy('T2','EPM156',mask_choice='LS',d=200,window=5,test_run=False)
    #energy('T2','EPM157',mask_choice='LS',d=200,window=5,test_run=False)
    energy('T2','EPM158',mask_choice='LS',d=200,window=5,test_run=False)
    quit()

    #energy('T2','EPM151','LS',d=200,window=5,test_run=True)
    #quit()
    runs = ['EPM151','EPM152','EPM155','EPM156','EPM157','EPM158']
    runid = runs[5]
    energy('T2',runid,mask_choice='LS',d=200,window=5,test_run=True)
    energy('T2',runid,mask_choice='LS',d=2000,window=5,test_run=False)
    energy('T2',runid,mask_choice='LS',d=200,window=21,test_run=False)
    energy('T2',runid,mask_choice='LS',d=2000,window=21,test_run=False)
    energy('EKE',runid,mask_choice='LS',d=200,window=21,test_run=False)
    energy('EKE',runid,mask_choice='LS',d=2000,window=21,test_run=False)
    energy('T1',runid,mask_choice='LS',d=200,window=21,test_run=False)
    energy('T1',runid,mask_choice='LS',d=2000,window=21,test_run=False)
    energy('T3',runid,mask_choice='LS',d=200,window=21,test_run=False)
    energy('T3',runid,mask_choice='LS',d=2000,window=21,test_run=False)


    #for mask_choice in ['LS3k']:#,'LS']:
    #    energy('T3',runid,mask_choice,d=200,window=5,test_run=False)
    #    energy('T3',runid,mask_choice,d=2000,window=5,test_run=False)
    #    energy('T1',runid,mask_choice,d=200,window=5,test_run=False)
    #    energy('T1',runid,mask_choice,d=2000,window=5,test_run=False)
