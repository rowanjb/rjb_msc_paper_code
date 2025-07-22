# Code to investigate the stratification in the Lab Sea
# Rowan Brown
# July 2025

import numpy as np
import pandas as pd
import xarray as xr
import cftime
from datetime import datetime
import gsw
import os

def convective_resistance(run):
    """Investigates the convective resistance in a region!"""

    print("Beginning: Convective resistance calculations for "+run)

    # Masks (for land, bathymetry, etc. and horiz. grid dimensions)
    with xr.open_dataset('masks/ANHA4_mesh_mask.nc') as DS:
        tmask = DS.tmask[0,:,:,:].rename({'z': 'deptht', 'y': 'y_grid_T', 'x': 'x_grid_T'}) 
        e1t = DS.e1t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
        e2t = DS.e2t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
    mask = xr.open_dataarray('masks/mask_LS_3000.nc').astype(int)

    # Text file of paths to non-empty model output
    gridT_txt = '../filepaths/'+run+'_gridT_filepaths_jul2025.txt'

    # Open the text files and get lists of the .nc output filepaths
    with open(gridT_txt) as f: lines = f.readlines()
    filepaths_gridT = [line.strip() for line in lines]

    # Open the files and look at e3t and votemper
    preprocess_gridT = lambda ds: ds[['e3t','votemper','vosaline','somxl010']]
    DS = xr.open_mfdataset(filepaths_gridT,preprocess=preprocess_gridT) 

    # Add horizontal cell dims
    DS[['e1t','e2t']] = e1t,e2t 

    # Need to mask the shelves/sea floor, or else the "0" temperatures are counted
    # Easy way to do this is to mask anywhere with 0 salinities, since 0 temps are plausible
    DS = DS.where(DS.vosaline>0)

    # Apply tmask (which I /think/ it for land etc.)
    DS = DS.where(tmask == 1)

    # Apply region mask
    DS.coords['mask'] = mask
    DS = DS.where(DS.mask == 1, drop=True)

    #== Begin actual calculations (note: these are based on the Introducing LAB60... paper from Clark Pennelly ==#
   
    # Note for convective resistance calculations, you need to select a reference depth
    depth = 2000 # Clark uses 2,000 m, so that's what we'll use here
    DS = DS.where(DS.deptht < depth, drop=True)

    # Gravity (why not incl. lots of decimal places?)
    g = 9.80665

    # Calculate cell areas 
    DS['areas'] = DS['e1t'].isel(time_counter=0,deptht=0)*DS['e2t'].isel(time_counter=0,deptht=0)
    
    # Calculate cell potential densities
    DS['p'] = gsw.p_from_z( (-1)*DS['deptht'], DS['nav_lat_grid_T'] ) # Requires negative depths
    DS['SA'] = gsw.SA_from_SP( DS['vosaline'], DS['p'], DS['nav_lon_grid_T'], DS['nav_lat_grid_T'] )
    DS['CT'] = gsw.CT_from_pt( DS['SA'], DS['votemper'])
    DS['pot_dens'] = DS['pot_dens'] = gsw.rho( DS['SA'], DS['CT'], 0 ) # gsw.rho with p=0 gives potential density

    # ...calculating term 1 in the double integrand
    # (Used e3t.sum instead of depth[-1] because term2 is integrated using e3t too
    DS['term1'] = DS['e3t'].sum(dim='deptht')*DS['pot_dens'].isel(deptht=-1) # [m]*[kg/m**3]

    # ...calculating term 2 in the double integrand 
    DS['term2'] = (DS['pot_dens']*DS['e3t']).sum('deptht') # [m]*[kg/m**3] = [kg/m**2]

    # ...calculating integrand
    DS['integrand'] = DS['term1'] - DS['term2'] # [m]*[kg/m**3] = [kg/m**2]

    # Finally, taking the convective resistance for the whole area
    DS['omega_cr_per_vol'] = g*DS['integrand'] # [m/s**2]*[kg/m**2] = [kg / s**2 m] = [J/m**3]
    DS['omega_cr'] = DS['omega_cr_per_vol']*DS['areas']*DS['e3t'].sum(dim='deptht') # Multiply by area and depth to get J
    DS['omega_cr_sum'] = DS['omega_cr'].sum(['x_grid_T','y_grid_T'])
    
    # Saving
    DS['omega_cr_sum'].to_netcdf('ls3k_convective_resistance_'+run+'.nc') 
    
    print('Completed: Convective resistance calculations for ' + run)

def MLD(run):
    """Creates and saves datasets of MLDs and convective 
    volumes in the Lab Sea."""

    print("Beginning: MLD calculations for "+run)

    # Masks (for land, bathymetry, etc. and horiz. grid dimensions)
    with xr.open_dataset('masks/ANHA4_mesh_mask.nc') as DS:
        tmask = DS.tmask[0,:,:,:].rename({'z': 'deptht', 'y': 'y_grid_T', 'x': 'x_grid_T'})
        e1t = DS.e1t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
        e2t = DS.e2t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
    mask = xr.open_dataarray('masks/mask_LS_3000.nc').astype(int)

    # Text file of paths to non-empty model output
    gridT_txt = '../filepaths/'+run+'_gridT_filepaths_jul2025.txt'

    # Open the text files and get lists of the .nc output filepaths
    with open(gridT_txt) as f: lines = f.readlines()
    filepaths_gridT = [line.strip() for line in lines]

    # Open the files and look at MLD 
    preprocess_gridT = lambda ds: ds[['somxlts']]
    DS = xr.open_mfdataset(filepaths_gridT,preprocess=preprocess_gridT)
    DS = DS.rename({'somxlts':'MLD'}) # Easier to handle this way

    # Add horizontal cell dims (drop depths now because they aren't needed)
    DS[['e1t','e2t']] = e1t,e2t

    # Apply tmask (which I /think/ it for land etc.)
    DS = DS.where(tmask.isel(deptht=0) == 1)

    # Add region mask
    DS.coords['mask'] = mask.isel(deptht=0)

    # Note that masking shelves likly isn't necessary because we're only looking at MLD

    #== Calculations ==#

    # We are looking at several related ideas: 
    #    (1) Full domain maps of yearly mean and max MLD,
    #    (2) 10-yr time series of convective volume (defined with z_crit=1000 m), and 
    #    (3) 10-yr time series of mean MLD in a masked region in the interior Lab Sea. 

    # (1) Starting with full domain maps of yearly mean and max MLD
    global_MLD_maps = DS['MLD'].groupby('time_counter.year').max(dim=['time_counter'],skipna=True).to_dataset().rename({'MLD':'yearly_max'})
    global_MLD_maps['yearly_mean'] = DS['MLD'].groupby('time_counter.year').mean(dim=['time_counter'], skipna=True)
    global_MLD_maps.to_netcdf('MLD_yearly_maps_full_domain_'+run+'.nc')
    print('Saved: MLD yearly maps for '+run)

    # (2) Next looking at convective volume 
    # Extracting MLDs only where deeper than 1000 m and multiplying by cell area
    # Only interested in interior Lab Sea, hence need to mask first
    DS = DS.where(DS['mask'] == 1, drop=True)
    conv_vol = DS['MLD'].where(DS['MLD']>1000)*DS['e1t']*DS['e2t'] 
    conv_vol.sum(dim=['y_grid_T','x_grid_T']).to_netcdf('1kMLD_convective_volume_'+run+'.nc')
    print('Saved: Convective volume time series analyses for '+run)

    # (3) Finally looking at time series of MLD in the interior Lab Sea
    # I'm pretty sure I originally copied this weighting technique from the xarray docs
    areas = DS.e1t*DS.e2t
    avgArea = areas.mean(dim=['y_grid_T','x_grid_T'])
    weights = areas/avgArea 
    weights = weights.fillna(0)
    MLD = DS['MLD'].weighted(weights) 
    avgMLD_region = MLD.mean(dim=['y_grid_T','x_grid_T'],skipna=True)
    avgMLD_region.to_netcdf('ls3k_MLD_mean_'+run+'.nc')
    print('Saved: MLD time series analyses for '+run)

    print('Completed: Stratification analyses for '+run)

def MLD_Argo():
    """A shortened copy of MLD().
    Creates and saves datasets of Argo MLDs in the Lab Sea."""

    print("Beginning: MLD calculations for Argo")

    # Masks (for land, bathymetry, etc. and horiz. grid dimensions)
    with xr.open_dataset('masks/ANHA4_mesh_mask.nc') as DS:
        tmask = DS.tmask[0,:,:,:].isel(z=0)
        e1t = DS.e1t[0,:,:]
        e2t = DS.e2t[0,:,:]
    mask = xr.open_dataarray('masks/mask_LS_3000.nc').astype(int).isel(deptht=0)
    mask = mask.rename({'x_grid_T':'x','y_grid_T':'y'})

    # In the Argo gridding script, I cut down the size as follows
    e1t = e1t.where( (e1t.x>100) & (e1t.x<250) & (e1t.y>300) & (e1t.y<500), drop=True)
    e2t = e2t.where( (e2t.x>100) & (e2t.x<250) & (e2t.y>300) & (e2t.y<500), drop=True)
    mask = mask.where( (mask.x>100) & (mask.x<250) & (mask.y>300) & (mask.y<500), drop=True)
    tmask = tmask.where( (tmask.x>100) & (tmask.x<250) & (tmask.y>300) & (tmask.y<500), drop=True)

    # Open Argo data
    Argo_fp = 'Argo_mld_ANHA4_LabSea.nc'
    DS = xr.open_mfdataset(Argo_fp).set_coords(['nav_lat','nav_lon']).rename({'nav_lat':'nav_lat_grid_T','nav_lat':'nav_lon_grid_T'})

    # Add horizontal cell dims (drop depths now because they aren't needed)
    DS[['e1t','e2t']] = e1t,e2t

    # Apply tmask (which I /think/ it for land etc.)
    DS = DS.where(tmask == 1)

    # Add region mask
    DS.coords['mask'] = mask

    #== Calculations ==#

    # We are looking at several related ideas: 
    #    (1) Full domain maps of yearly mean and max MLD,
    #    (2) 10-yr time series of mean MLD in a masked region in the interior Lab Sea. 

    # (1) Starting with full domain maps of yearly mean and max MLD
    Argo_MLD_maps = DS['da_mld'].groupby('date.year').max(dim=['date'],skipna=True).to_dataset().rename({'da_mld':'yearly_max'})
    Argo_MLD_maps['yearly_mean'] = DS['da_mld'].groupby('date.year').mean(dim=['date'], skipna=True)
    Argo_MLD_maps.to_netcdf('MLD_yearly_maps_full_domain_Argo.nc')
    print('Saved: Argo MLD yearly maps')

    # (2) Now mask and look at the time series of MLD
    DS = DS.where(DS['mask'] == 1, drop=True)
    areas = DS.e1t*DS.e2t
    avgArea = areas.mean(dim=['y','x'])
    weights = areas/avgArea
    weights = weights.fillna(0)
    MLD = DS['da_mld'].weighted(weights)
    avgMLD_region = MLD.mean(dim=['y','x'],skipna=True)
    avgMLD_region.to_netcdf('ls3k_MLD_mean_Argo.nc')
    print('Saved: MLD time series analyses for Argo')

    print('Completed: Stratification analyses for Argo')

def MLD_LAB60():
    """Copy of MLD() but for LAB60.
    Creates and saves datasets of MLDs and convective
    volumes in the Lab Sea."""

    

if __name__ == '__main__':
    #MLD_LAB60
    
    #MLD_Argo()
    #for run in ['EPM151','EPM152','EPM155','EPM156','EPM157','EPM158']:
    #    MLD(run)
    #    convective_resistance(run)
