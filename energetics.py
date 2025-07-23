# Energetics for ANHA4 runs, including EKE and Lorenz cycle
# (For supplemental figures to my MSc paper)
# Rowan Brown
# July 2025

import numpy as np 
import xarray as xr
import os
from datetime import datetime
import gsw

def energy(calc_type,run,d,window=21):
    """Calculates the EKE or one of four (T1, T2, T3, or T4) pathways in the Lorenz cycle.
    Can specify the depth you want to integrate over and the size of the rolling window (for
    calculations that require looking at an anomaly).
    calc_type : "EKE", "T1", "T2", "T3", "T4"
    run :       "EPM151", "EPM152", etc.
    d :         200, 2000 etc.
    window :    5, 21 etc."""

    if d>3000: 
        print("Depth must be less than 3,000 m, since this is the depth of the isobath mask")
        quit()
    
    #== OPENING AND INITIAL PROCESSING OF THE NETCDF MODEL OUTPUT FILES ==#

    # These are text files with lists of all non-empty model outputs
    gridU_txt = '../filepaths/'+run+'_gridU_filepaths_jul2025.txt'
    gridV_txt = '../filepaths/'+run+'_gridV_filepaths_jul2025.txt'
    gridT_txt = '../filepaths/'+run+'_gridT_filepaths_jul2025.txt'
    gridW_txt = '../filepaths/'+run+'_gridW_filepaths_jul2025.txt'

    # Open the text files and get lists of the .nc output filepaths
    years_we_care_about = list(range(2008,2019))
    start_date, end_date = datetime(2007,6,1), datetime(2019,3,1)
    with open(gridU_txt) as f: lines = f.readlines()
    filepaths_gridU = [line.strip() for line in lines]
    filepaths_gridU = [fp for fp in filepaths_gridU if (start_date <= datetime.strptime(fp[-20:-9],'y%Ym%md%d') <= end_date)]
    with open(gridV_txt) as f: lines = f.readlines()
    filepaths_gridV = [line.strip() for line in lines]
    filepaths_gridV = [fp for fp in filepaths_gridV if (start_date <= datetime.strptime(fp[-20:-9],'y%Ym%md%d') <= end_date)]
    with open(gridT_txt) as f: lines = f.readlines()
    filepaths_gridT = [line.strip() for line in lines]
    filepaths_gridT = [fp for fp in filepaths_gridT if (start_date <= datetime.strptime(fp[-20:-9],'y%Ym%md%d') <= end_date)]
    with open(gridW_txt) as f: lines = f.readlines()
    filepaths_gridW = [line.strip() for line in lines]
    filepaths_gridW = [fp for fp in filepaths_gridW if (start_date <= datetime.strptime(fp[-20:-9],'y%Ym%md%d') <= end_date)]
  
    # Print lens to ensure they're the same (basically a simple manual check)
    print('Number of gridW files: ' + str(len(filepaths_gridW)))
    print('Number of gridT files: ' + str(len(filepaths_gridT)))
    print('Number of gridV files: ' + str(len(filepaths_gridV)))
    print('Number of gridU files: ' + str(len(filepaths_gridU)))

    # Preprocessing (specifying variables that we need and ignoring the rest)
    # (We don't need e3u e3v or e3w since we co-locate the velocities on the T grid)
    preprocess_gridU = lambda ds: ds[['vozocrtx']]
    preprocess_gridV = lambda ds: ds[['vomecrty']]
    preprocess_gridW = lambda ds: ds[['vovecrtz']]
    if calc_type=='EKE' or calc_type=='T4': #i.e., if we don't need densities
        preprocess_gridT = lambda ds: ds[['e3t']]
    else: #i.e., if we need densities
        preprocess_gridT = lambda ds: ds[['e3t','votemper','vosaline']]

    # Open gridT
    DST = xr.open_mfdataset(filepaths_gridT,preprocess=preprocess_gridT,engine="netcdf4")
    DST = DST.rename({'deptht': 'depth', 'y_grid_T': 'y', 'x_grid_T': 'x'})

    # Mask for land, bathymetry, etc. and horiz. grid dimensions
    with xr.open_dataset('masks/ANHA4_mesh_mask.nc') as DS:
        DST['e1t'] = DS.e1t[0,:,:]
        DST['e2t'] = DS.e2t[0,:,:]
        DST['e1u'] = DS.e1t[0,:,:] # Needed to calc gradients between grid T points
        DST['e2v'] = DS.e2t[0,:,:] # Needed to calc gradients between grid T points
    with xr.open_dataarray('masks/mask_LS_3000.nc') as DS:
        DST = DST.assign_coords(mask=DS.astype(int).rename({'deptht':'depth','x_grid_T':'x','y_grid_T':'y'}))

    # If you need w velocities...
    if calc_type == 'T3' or calc_type == 'T1':
        DSW = xr.open_mfdataset(filepaths_gridW,preprocess=preprocess_gridW,engine="netcdf4")
        DSW = DSW.rename({'depthw': 'depth'}) # depth gets reset/interpolated onto deptht (already renamed depth) in few lines, so this is OK
        DSW = DSW.where(DST.mask.isel(depth=0).drop_vars('depth') == 1, drop=True).drop_vars(['mask','nav_lat_grid_T','nav_lon_grid_T'])
        DSW = DSW.interp(depth=DSW.depth+0.5) # Locating velocity on the T grid 
        # (The above is necessary so that the same exact volumes are used to get the total energy conversion rates across T1--T4)
        DSW['depth'] = DST['depth']
        DSW = DSW.where(DSW.depth < d, drop=True)

    # If you need horizontal velocities...
    if calc_type=='EKE' or calc_type=='T2' or calc_type=='T4':
        DSU = xr.open_mfdataset(filepaths_gridU,preprocess=preprocess_gridU,engine="netcdf4")
        DSV = xr.open_mfdataset(filepaths_gridV,preprocess=preprocess_gridV,engine="netcdf4")
        DSU = DSU.rename({'depthu': 'depth'})
        DSV = DSV.rename({'depthv': 'depth'})
        if calc_type=='T2' or calc_type=='T4': # In this case, we need gradients (i.e., bounding cells) and therefore cannot apply the mask yet
            # Cutting down grid size for less unnecessary computation
            # Note that for EKE, T1, and T3 we use the mask now; for T2 and T4 we use the mask later
            DSU = DSU.where((DSU.x>100)&(DSU.x<250)&(DSU.y>300)&(DSU.y<500),drop=True)
            DSV = DSV.where((DSV.x>100)&(DSV.x<250)&(DSV.y>300)&(DSV.y<500),drop=True)
            if calc_type=='T4': # Need to keep depths if T2
                DSU = DSU.where(DSU.depth < d, drop=True)
                DSV = DSV.where(DSV.depth < d, drop=True)
        else: # EKE calculations don't need gradients so we can apply the mask
            DSU = DSU.where(DST.mask.isel(depth=0).drop_vars('depth') == 1, drop=True).drop_vars(['mask','nav_lat_grid_T','nav_lon_grid_T'])
            DSV = DSV.where(DST.mask.isel(depth=0).drop_vars('depth') == 1, drop=True).drop_vars(['mask','nav_lat_grid_T','nav_lon_grid_T'])
            DSU = DSU.where(DSU.depth < d, drop=True)
            DSV = DSV.where(DSV.depth < d, drop=True)
        DSU = DSU.interp(x=DSU.x-0.5).drop_vars('x') # Co-locating velsocities on the T grid
        DSV = DSV.interp(y=DSV.y-0.5).drop_vars('y') 

    # Mirroring sections from above on the T grid
    if calc_type=='T2' or calc_type=='T4': # i.e., keeping DST same size as DSU and DSV in the case that we need gradients 
        DST = DST.where((DST.x>100)&(DST.x<250)&(DST.y>300)&(DST.y<500),drop=True)
        if calc_type=='T4':
            DST = DST.where(DST.depth < d,drop=True)
    else: 
        DST = DST.where(DST.mask == 1, drop=True)
        DST = DST.where(DST.depth < d,drop=True)

    # Constants needed for calculations
    rho_0 = 1025
    g = 9.80665

    print('Pre-analyses completed')

    if calc_type=='EKE':
        
        print('Calculating EKE for '+run+' down to '+str(d)+' m')
        
        # EKE is calculated based on the methods from: 
        #    Martínez-Moreno et al. - Global changes in oceanic mesoscale currents over the satellite altimetry record
        #    => EKE = (1/2) * density_0 * (u'**2 + v'**2) [J/m**3] where u′ = u − u_mean and v′ = v − v_mean
        #    recall, units are J = kg m**2 / s**2, density = kg / m**3, and vel**2 = m**2 / s**2, 
        #    so density*vel**2 = kg / m s**2 = J / m**3
        #    t.f. integrating  over volume gives total Joules

        # Calculating velocity anomalies
        DSU_bar_sqr = (DSU-DSU.rolling(time_counter=window,center=True).mean())**2 
        DSV_bar_sqr = (DSV-DSV.rolling(time_counter=window,center=True).mean())**2
        EKE = (1/2) * rho_0 * (DSU_bar_sqr.vozocrtx + DSV_bar_sqr.vomecrty) 
        EKE = EKE.rename('EKE')
        
        # Add grid T lat and lons and other useful quantities as coordinates (useful for plotting later)
        EKE = EKE.assign_coords(nav_lat_grid_T=DST.nav_lat_grid_T, 
                                nav_lon_grid_T=DST.nav_lon_grid_T,
                                mask=DST.mask,
                                e1t=DST.e1t,
                                e2t=DST.e2t,
                                e3t=DST.e3t)
        EKE = EKE.reset_coords(names=['e1t','e2t','e3t']) # Change cell dims from coordinates to variables

        # Calculating EKE (in J) within each cell
        EKE['areas'] = EKE.e1t*EKE.e2t
        EKE['EKE'] = EKE.EKE*EKE.areas*EKE.e3t # (EKE.EKE was originally in units J/m**3, so multiply by volume)

        # Note we don't need to mask the shelves or sea floor, since we've already masked the interior Lab Sea and
        # used .where to remove depths below the specified value
       
        # Also note that we could save maps now, but these aren't necessary for the paper so I will comment out this section
        #EKE['EKE_per_m2_in_J'] = EKE.EKE.sum(dim='depth',skipna=True)/(EKE.areas.isel(depth=0).drop_vars('depth'))
        #EKE_max_EKE_per_m2_in_J_each_year_AS_OCT = EKE.EKE_per_m2_in_J.resample(time_counter='AS-OCT').max(dim='time_counter',skipna=True)
        #EKE_max_EKE_per_m2_in_J_each_year_AS_OCT.to_netcdf('EKE_map_ls3k_'+run+'_depth'+str(d)+'m_window'+str(window)+'.nc') 
        
        # Saving EKE summed over the region
        EKE['EKE_in_region_in_J'] = EKE.EKE.sum(dim=['depth','x','y'],skipna=True)
        EKE.EKE_in_region_in_J.to_netcdf('EKE_time_series_ls3k_'+run+'_depth'+str(d)+'m_window'+str(window)+'.nc')
        
        # Closing
        print('EKE calculated for '+run+' down to '+str(d)+' m')
        DST.close()
        DSU.close()
        DSV.close()    

    elif calc_type=='T3': 
        
        print('Calculating T3 conversion  '+run+' down to '+str(d)+' m')
        
        # Rate of energy conversion from eddy available potential energy to eddy kinetic energy 
        # Unit is apparently W/m**3, so multiply by volume to get total rate 

        # Calculating potential density using gsw
        DST['pressure'] = gsw.p_from_z((-1)*DST['depth'],DST['nav_lat_grid_T']) # dbar, note depth needs to be negative
        DST['SA'] = gsw.SA_from_SP(DST['vosaline'],DST['pressure'],DST['nav_lon_grid_T'],DST['nav_lat_grid_T']) # unitless, i.e., g/kg 
        DST['CT'] = gsw.CT_from_pt(DST['SA'],DST['votemper']) # C
        DSW['rho'] = gsw.rho(DST['SA'],DST['CT'],0) # kg/m**3, equal to potential density if pressure = 0

        # Calculating conversion terms
        DSW['rho_anom'] = DSW['rho']-DSW['rho'].rolling(time_counter=window,center=True).mean()
        DSW['w_anom'] = DSW['vovecrtz'] - DSW['vovecrtz'].rolling(time_counter=window,center=True).mean()
        DSW['rho_anom_X_w_anom'] = DSW['rho_anom']*DSW['w_anom'] 
        DSW['areas'] = DST.e1t*DST.e2t
        DSW['T3_in_each_cell'] = (-1)*DSW['areas']*DST.e3t*g*DSW['rho_anom_X_w_anom'].rolling(time_counter=window,center=True).mean() 
        
        # As with EKE, we could save maps now, but these aren't necessary for the paper so I will comment out this section 
        #T3_max_per_col_per_year_AS_OCT = DSW['T3_in_each_cell'].sum(dim='depth',skipna=True).resample(time_counter='AS-OCT').max(dim='time_counter',skipna=True)
        #T3_max_per_m2_per_year_AS_OCT = T3_max_per_col_per_year_AS_OCT/(DSW.areas.isel(depth=0)) # Units become W/m2
        #T3_max_per_m2_per_year_AS_OCT = T3_max_per_m2_per_year_AS_OCT.drop_vars(['mask','depth','nav_lat','nav_lon'])
        #T3_max_per_m2_per_year_AS_OCT.to_netcdf('T3_map_ls3k_'+run+'_depth'+str(d)+'m_window'+str(window)+'.nc') 
        
        # Saving T3 summed over the region
        DSW['T3_in_each_cell'].sum(dim=['depth','x','y']).drop_vars('time_centered').to_netcdf('T3_time_series_ls3k_'+run+'_depth'+str(d)+'m_window'+str(window)+'.nc')

        # Closing
        print('T3 calculated for '+run+' down to '+str(d)+' m')

        DST.close()
        DSW.close()

    elif calc_type=='T1':

        print('Calculating T1 conversion for '+run+' down to '+str(d)+' m')
        
        # Rate of energy conversion from mean available potential energy to mean kinetic energy 

        # Calculating potential density using gsw
        DST['pressure'] = gsw.p_from_z((-1)*DST['depth'],DST['nav_lat_grid_T']) # dbar, note depth needs to be negative
        DST['SA'] = gsw.SA_from_SP(DST['vosaline'],DST['pressure'],DST['nav_lon_grid_T'],DST['nav_lat_grid_T']) # unitless, i.e., g/kg 
        DST['CT'] = gsw.CT_from_pt(DST['SA'],DST['votemper']) # C
        DSW['rho'] = gsw.rho(DST['SA'],DST['CT'],0) # kg/m**3, equal to potential density if pressure = 0

        # Calculating conversion terms
        DSW['areas'] = DST.e1t*DST.e2t
        DSW['rho_bar'] = DSW['rho'].rolling(time_counter=window,center=True).mean()
        DSW['w_bar'] = DSW['vovecrtz'].rolling(time_counter=window,center=True).mean()
        DSW['T1_in_each_cell'] = (-1)*g*DSW['areas']*DST.e3t*DSW['rho_bar']*DSW['w_bar'] 
        DSW = DSW.drop_vars(['nav_lat','nav_lon','mask','time_centered'])
        
        # As with T3 and EKE, we could save maps now, but these aren't necessary for the paper so I will comment out this section 
        #T1_per_m2 = DSW['T1_in_each_cell'].sum(dim='depth',skipna=True)/(DSW.areas.isel(depth=0).drop_vars('depth'))
        #T1_max_per_m2_per_year_AS_OCT = T1_per_m2.resample(time_counter='AS-OCT').max(dim='time_counter',skipna=True)
        #T1_max_per_m2_per_year_AS_OCT.to_netcdf('T1_map_ls3k_'+run+'_depth'+str(d)+'m_window'+str(window)+'.nc')

        # Saving T1 summed over the region
        DSW['T1_in_each_cell'].sum(dim=['depth','x','y']).to_netcdf('T1_time_series_ls3k_'+run+'_depth'+str(d)+'m_window'+str(window)+'.nc')
        
        # Closing
        print('T1 calculated for '+run+' down to '+str(d)+' m')
        DST.close()
        DSW.close()

    elif calc_type=='T4': 
        
        print('Calculating T4 conversionfor '+run+' down to '+str(d)+' m')
        
        # Rate of energy conversion from mean kinetic energy to eddy kinetic 
        # Eq "BT" from Gou et al. --- Variability of Eddy Formation...
        # as with the other conversion terms, unit is W/m3, so multiply by volume to get total energy conversion in masked region
        # this function and the next (T2) could likely be optimized computationally but they should also work as-is

        # Terms of the eq
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
        
        # Calculating gradients
        # (Use e1u and e2v since these measure the zonal and meridional distances between T points)
        d_u_bar_in_x = DSU['u_bar'].diff(dim='x',label='lower').drop_vars(['nav_lat','nav_lon'])/DST.e1u.isel(x=slice(0,-1)) 
        d_v_bar_in_y = DSV['v_bar'].diff(dim='y',label='lower').drop_vars(['nav_lat','nav_lon'])/DST.e2v.isel(y=slice(0,-1)) 
        d_u_bar_in_y = DSU['u_bar'].diff(dim='y',label='lower').drop_vars(['nav_lat','nav_lon'])/DST.e2v.isel(y=slice(0,-1)) 
        d_v_bar_in_x = DSV['v_bar'].diff(dim='x',label='lower').drop_vars(['nav_lat','nav_lon'])/DST.e1u.isel(x=slice(0,-1)) 

        # Making sure all terms are the correct size
        DSU = DSU.isel(x=slice(0,-1),y=slice(0,-1))
        DSV = DSV.isel(x=slice(0,-1),y=slice(0,-1))
        d_u_bar_in_x = d_u_bar_in_x.isel(y=slice(0,-1))
        d_v_bar_in_y = d_v_bar_in_y.isel(x=slice(0,-1))
        d_u_bar_in_y = d_u_bar_in_y.isel(x=slice(0,-1))
        d_v_bar_in_x = d_v_bar_in_x.isel(y=slice(0,-1))
        DST = DST.isel(x=slice(0,-1),y=slice(0,-1))

        # Calculating T4 and masking
        BT = (-1)*rho_0*( DSU['u_prime_sqr_bar']*d_u_bar_in_x + DSV['v_prime_sqr_bar']*d_v_bar_in_y + DSU['uv_prime_bar']*( d_u_bar_in_y + d_v_bar_in_x ) )
        BT = BT.where(DST.mask==1,drop=True).drop_vars(['nav_lat','nav_lon','time_centered','mask']) # Mask now because we couldn't mask earlier
        DST = DST.where(DST.mask==1,drop=True) # As above, mask now because we couldn't mask earlier
        areas = DST.e1t*DST.e2t
        BT_per_cell = BT*areas*DST.e3t
        
        # As with the other terms, we can save maps now but this isn't necessary for the paper so I'll comment out this block
        #BT_per_m2 = BT_per_cell.sum(dim='depth')/areas.isel(depth=0).drop_vars('depth')
        #BT_max_per_m2_per_year_AS_OCT = BT_per_m2.resample(time_counter='AS-OCT').max(dim='time_counter',skipna=True)
        #BT_max_per_m2_per_year_AS_OCT.drop_vars('mask').to_netcdf('T4_map_ls3k_'+run+'_depth'+str(d)+'m_window'+str(window)+'.nc') 

        # Saving T4 summed over the region
        BT_per_cell.sum(dim=['depth','x','y']).to_netcdf('T4_time_series_ls3k_'+run+'_depth'+str(d)+'m_window'+str(window)+'.nc')

        # Closing
        print('T4 calculated for '+run+' down to '+str(d)+' m')
        DST.close()
        DSU.close()
        DSV.close()

    elif calc_type=='T2':

        print('Calculating T2 conversion for '+run+' down to '+str(d)+' m')
        
        # Rate of energy conversion from eddy available potential energy to mean availabile potential energy
        # Actually, I think it goes the other direction (Rieck + Gou disagree with Von Storch on the direction)
        # See eq "BC" from Gou et al. --- Variability of Eddy Formation...

        # Need to get buoyancy frequency, so need thickness of w-cells, i.e., vertical distance from t-point to t-point
        preprocess_gridW = lambda ds: ds[['e3w']]
        e3w = xr.open_mfdataset(filepaths_gridW,preprocess=preprocess_gridW,engine="netcdf4")
        e3w = e3w.where((e3w.x>100)&(e3w.x<250)&(e3w.y>300)&(e3w.y<500),drop=True).rename({'depthw':'depth','nav_lat':'nav_lat_grid_T','nav_lon':'nav_lon_grid_T'})
        e3w['depth'] = DST.depth 
        DST['e3w'] = e3w['e3w']

        # Calculating potential density using gsw
        DST['pressure'] = gsw.p_from_z((-1)*DST['depth'],DST['nav_lat_grid_T']) # dbar, note depth needs to be negative
        DST['SA'] = gsw.SA_from_SP(DST['vosaline'],DST['pressure'],DST['nav_lon_grid_T'],DST['nav_lat_grid_T']) # unitless, i.e., g/kg 
        DST['CT'] = gsw.CT_from_pt(DST['SA'],DST['votemper']) # C
        DST['rho'] = gsw.rho(DST['SA'],DST['CT'],0) # kg/m**3, equal to potential density if pressure = 0

        # Calculating converstion terms
        DST['areas'] = DST.e1t*DST.e2t
        DST['rho_bar'] = DST['rho'].rolling(time_counter=window,center=True).mean()
        DST['rho_prime'] = DST['rho'] - DST['rho_bar']
        DSU['u_prime'] = DSU['vozocrtx'] - DSU['vozocrtx'].rolling(time_counter=window,center=True).mean()
        DSV['v_prime'] = DSV['vomecrty'] - DSV['vomecrty'].rolling(time_counter=window,center=True).mean()
        DSU['urho_prime'] = DSU['u_prime']*DST['rho_prime']
        DSV['vrho_prime'] = DSV['v_prime']*DST['rho_prime']
        DSU['urho_prime_bar'] = DSU['urho_prime'].rolling(time_counter=window,center=True).mean()
        DSV['vrho_prime_bar'] = DSV['vrho_prime'].rolling(time_counter=window,center=True).mean()

        # Calculating buoyancy frequency
        DST['b'] = (-1)*g*(DST['rho'] - rho_0)/rho_0
        N2 = (-1)*DST['b'].diff(dim='depth',label='lower')/DST['e3w'].isel(depth=slice(0,-1))
        
        # Calculating gradients
        rho_bar_in_x = DST['rho_bar'].diff(dim='x',label='lower')/DST.e1u.isel(x=slice(0,-1))
        rho_bar_in_y = DST['rho_bar'].diff(dim='y',label='lower')/DST.e2v.isel(y=slice(0,-1))

        # Making sure all terms are the correct size (including dropping below depth d, which we haven't done yet)
        N2 = N2.isel(x=slice(0,-1),y=slice(0,-1)).where(N2.depth < d, drop=True)
        DSU = DSU.isel(x=slice(0,-1),y=slice(0,-1)).where(DSU.depth < d, drop=True)
        DSV = DSV.isel(x=slice(0,-1),y=slice(0,-1)).where(DSV.depth < d, drop=True)
        DST = DST.isel(x=slice(0,-1),y=slice(0,-1)).where(DST.depth < d, drop=True)
        rho_bar_in_x = rho_bar_in_x.isel(y=slice(0,-1)).where(rho_bar_in_x.depth < d, drop=True)
        rho_bar_in_y = rho_bar_in_y.isel(x=slice(0,-1)).where(rho_bar_in_x.depth < d, drop=True)

        # Calculating T2
        N2 = N2.where(N2!=0)
        BC = (-1)*((g**2)/(N2 * rho_0))*( DSU['urho_prime_bar']*rho_bar_in_x + DSV['vrho_prime_bar']*rho_bar_in_y )

        # Masking and final calculations
        # (as with T4, we couldn't mask earlier because we needed neighbouring cells to get gradients)
        BC = BC.where(BC.mask==1,drop=True).drop_vars(['time_centered','mask'])
        DST = DST.where(DST.mask==1,drop=True)
        areas = DST.e1t*DST.e2t
        BC_per_cell = BC*areas*DST.e3t

        # As with the other terms, we can save maps now but this isn't necessary for the paper so I'll comment out this block
        #BC_per_m2 = BC_per_cell.sum(dim='depth')/areas.isel(depth=0).drop_vars('depth')
        #BC_max_per_m2_per_year_AS_OCT = BC_per_m2.resample(time_counter='AS-OCT').max(dim='time_counter',skipna=True)
        #BC_max_per_m2_per_year_AS_OCT.drop_vars('mask').to_netcdf('T2_map_ls3k_'+run+'_depth'+str(d)+'m_window'+str(window)+'.nc'

        # Saving T2 summed over the region
        BC_per_cell.sum(dim=['depth','x','y']).to_netcdf('T2_time_series_ls3k_'+run+'_depth'+str(d)+'m_window'+str(window)+'.nc')

        # Closing
        print('T2 calculated for '+run+' down to '+str(d)+' m')
        DST.close()
        DSU.close()
        DSV.close()
        e3w.close()

if __name__ == '__main__':
   
    for run in ['EPM151','EPM152','EPM155','EPM156','EPM157','EPM158']:
        #energy('EKE',run,400)
        #energy('EKE',run,2000)
        #energy('T3',run,400)
        #energy('T3',run,2000)
        #energy('T1',run,400)
        #energy('T1',run,2000)
        #energy('T4',run,400)
        #energy('T4',run,2000)
        energy('T2',run,400)
        energy('T2',run,2000)

