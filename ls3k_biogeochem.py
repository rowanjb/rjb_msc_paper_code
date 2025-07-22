# Looks at oxygen and CO2 fluxes
# Rowan Brown
# 25 Oct 2023

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime,timedelta
from functools import reduce
import matplotlib.pyplot as plt

def bling(run):
    """Calculates the co2 and oxygen contents in the interior Labrador Sea."""
    # Note: Original units are mol/m**3 for both

    print("Beginning: Oxygen and CO2 calculations for "+run)

    # Masks (for land, bathymetry, etc. and horiz. grid dimensions)
    with xr.open_dataset('masks/ANHA4_mesh_mask.nc') as DS:
        tmask = DS.tmask[0,:,:,:].rename({'z': 'deptht', 'y': 'y_grid_T', 'x': 'x_grid_T'}) 
        e1t = DS.e1t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
        e2t = DS.e2t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
    mask = xr.open_dataarray('masks/mask_LS_3000.nc').astype(int)

    # Text file of paths to non-empty model output
    gridT_txt = '../filepaths/'+run+'_gridT_filepaths_bling.txt' 

    # Open the text files and get lists of the .nc output filepaths
    with open(gridT_txt) as f: lines = f.readlines()
    filepaths_gridT = [line.strip() for line in lines]

    # Open the files and look at e3t and votemper
    preprocess_gridT = lambda ds: ds[['e3t','vooxy','vodic']]
    DS = xr.open_mfdataset(filepaths_gridT,preprocess=preprocess_gridT) 

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

    #== Content calculations ==#

    # Calculate volumes
    DS['volumes'] = DS.e1t*DS.e3t*DS.e2t # Volume of each cell
    
    # Calculate total mols in each cell
    DS['ox_total'] = DS['vooxy']*DS['volumes']
    DS['dic_total'] = DS['vodic']*DS['volumes']

    # Average in space and save
    # i.e., avg conc  = total mols  in the region / total volume of the region
    DS['ox_avg_conc'] = DS['ox_total'].sum(dim=['x_grid_T','y_grid_T','deptht'])/DS['volumes'].sum(dim=['x_grid_T','y_grid_T','deptht'])
    DS['dic_avg_conc'] = DS['dic_total'].sum(dim=['x_grid_T','y_grid_T','deptht'])/DS['volumes'].sum(dim=['x_grid_T','y_grid_T','deptht'])
    DS = DS.assign_attrs({'description': 'Oxygen and CO2 data in the interior Lab Sea, defined by the 3,000 m isobath',
                    'title': 'Oxygen and CO2 data in the interior Lab Sea'})
    DS[['ox_avg_conc','dic_avg_conc']].to_netcdf('ls3k_biogeochem_'+run+'.nc')
    
    print('Completed: Oxygen and CO2 calculations for ' + run)


def bling_plot():
    """Plotting function for my MSc paper"""
    print('to come...')

if __name__ == '__main__':
    bling('EPM158')
