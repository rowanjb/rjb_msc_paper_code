# For understanding the MLE streamfunction
# Rowan Brown
# July 2025

import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

def MLE(run):
    """For understanding the MLE stremfunction.
    Saves datasets that can be used to make a sort of "climatology" and maps.
    Requires diagnostic output of the stremfunction components.
    For these calculations, we're going to look at the full Lab Sea, but just the interior."""

    print("Beginning: MLE calculations for "+run)

    # Masks (for land, bathymetry, etc. and horiz. grid dimensions)
    with xr.open_dataset('masks/ANHA4_mesh_mask.nc') as DS:
        tmask = DS.tmask[0,:,:,:].rename({'z': 'deptht', 'y': 'y_grid_T', 'x': 'x_grid_T'}) 
        e1t = DS.e1t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
        e2t = DS.e2t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
    mask = xr.open_dataarray('masks/mask_LS.nc').astype(int)

    # Text file of paths to non-empty model output
    gridT_txt = '../filepaths/'+run+'_gridT_filepaths_jul2025.txt'

    # Open the text files and get lists of the .nc output filepaths
    with open(gridT_txt) as f: lines = f.readlines()
    filepaths_gridT = [line.strip() for line in lines]
    
    # Note we only have diagnostics for six years, so we might as well only open the necessary files
    start, end = datetime(2012,1,1), datetime(2017,12,31)
    filepaths_gridT = [fp for fp in filepaths_gridT if (start <= datetime.strptime(fp[-20:-9],'y%Ym%md%d') <= end)]

    # Open the files and look at e3t and and the MLE variables
    preprocess_gridT = lambda ds: ds[['e3t','MLE Lf','i-mle streamfunction','j-mle streamfunction']]
    DS = xr.open_mfdataset(filepaths_gridT,preprocess=preprocess_gridT)
   
    # Interpolate the streamfunction components
    iMLE = DS['i-mle streamfunction'].interp(x_grid_U=DS.x_grid_U-0.5).drop_vars(['x_grid_U','nav_lat_grid_U','nav_lon_grid_U'])
    iMLE = iMLE.rename({'depthu':'deptht','x_grid_U':'x_grid_T','y_grid_U':'y_grid_T'})
    jMLE = DS['j-mle streamfunction'].interp(y_grid_V=DS.y_grid_V-0.5).drop_vars(['y_grid_V','nav_lat_grid_V','nav_lon_grid_V'])
    jMLE = jMLE.rename({'depthv':'deptht', 'x_grid_V':'x_grid_T','y_grid_V':'y_grid_T'})
   
    # Calculate the "magnitude" of the streamfunction 
    DS['Psi'] = (iMLE**2 + jMLE**2)**0.5
    DS = DS.drop_vars(['i-mle streamfunction','j-mle streamfunction','nav_lon_grid_U','nav_lat_grid_U'])
    DS = DS.drop_vars(['nav_lon_grid_V','nav_lat_grid_V','depthu','depthv'])

    # Add horizontal cell dims
    DS[['e1t','e2t']] = e1t,e2t

    # Need to mask the shelves/sea floor, or else the "0" temperatures are counted
    # Easy way to do this is to mask anywhere with 0 salinities, since 0 temps are plausible
    vosaline = xr.open_dataset(filepaths_gridT[0])['vosaline'].isel(time_counter=0)
    DS['vosaline'] = vosaline
    DS = DS.where(DS.vosaline>0)

    # Apply tmask (which I /think/ it for land etc.)
    DS = DS.where(tmask == 1)

    # Apply region mask
    DS.coords['mask'] = mask
    DS = DS.where(DS.mask == 1, drop=True)

    # Calculate volumes and weights
    DS['volume'] = DS.e1t*DS.e3t*DS.e2t # Volume of each cell
    DS['weights'] = DS['volume']/DS['volume'].mean(['deptht','y_grid_T','x_grid_T'],skipna=True)

    # Final calcs and saving data that we want
    DS['Psi_weighted'] = DS['Psi']*DS['weights'] # Can now take mean
    DS['Psi_mean'] = DS['Psi_weighted'].mean(['deptht','y_grid_T','x_grid_T'], skipna=True)
    DS['Psi_mean'].to_netcdf('MLE_psi_time_series_'+run+'.nc')
    DS['Psi_max_map'] = DS['Psi'].max(dim='deptht')
    psi_yearly_max_map = DS['Psi_max_map'].resample(time_counter='AS-OCT').max(dim='time_counter',skipna=True)
    psi_yearly_max_map.to_netcdf('MLE_psi_max_map_'+run+'.nc')
    
    print("Completed: MLE calculations for "+run)

if __name__ == '__main__':
    MLE('EPM155')
    MLE('EPM156')
