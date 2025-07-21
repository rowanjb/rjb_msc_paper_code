# Produces heat content and temperature 
# Rowan Brown
# July 2025

import numpy as np 
import pandas as pd
import xarray as xr
import os
import gsw
from datetime import datetime,timedelta
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def heat_and_salt_content(run): 
    """Calculates heat and salt content in the interior Lab Sea."""

    print("Beginning: Heat and salt content calculations for "+run)

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
    preprocess_gridT = lambda ds: ds[['e3t','votemper','vosaline']]
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

    #== Heat content calculations ==# 

    # Constant needed calculations
    refT = -2 # Reference temperature [C]  

    # Heat content calculations
    DS['volumes'] = DS.e1t*DS.e3t*DS.e2t # Volume of each cell

    # Get pressure from depth (simplification, but minor)
    DS['p'] = gsw.p_from_z( (-1)*DS['deptht'], DS['nav_lat_grid_T'] ) # Requires negative depths

    # Get absolute salinity and conservative temperature (necessary for using gsw.rho)
    DS['SA'] = gsw.SA_from_SP( DS['vosaline'], DS['p'], DS['nav_lon_grid_T'], DS['nav_lat_grid_T'] )
    DS['CT'] = gsw.CT_from_pt( DS['SA'], DS['votemper'])

    # Get the heat capacity
    DS['cp'] = gsw.cp_t_exact( DS['SA'], gsw.t_from_CT( DS['SA'], DS['CT'], DS['p'] ), DS['p'] )

    # Get potential density (gsw.rho with p=0 gives potential density)
    DS['pot_dens'] = gsw.rho( DS['SA'], DS['CT'], 0 )

    # Finally, calculate heat content 
    # Units:    J      = m**3          * kg/m**3        * J/kgC    *   C
    DS['heat_content'] = DS['volumes'] * DS['pot_dens'] * DS['cp'] * ( DS['votemper'] - refT )

    # Take the sum in space and save
    DS['heat_content'].sum(['deptht','y_grid_T','x_grid_T']).to_netcdf('ls3k_heat_content_'+run+'.nc')

    print('completed: Heat content saved for ' + run)

    #== Salt content calculations ==#

    # Get in-situ density (need in-situ density, not potential density as with heat)
    DS['insit_dens'] = gsw.rho( DS['SA'], DS['CT'], DS['p']) # kg/m**3

    # Finally, calculate salt content
    # Units:     g     = m**3          * g/kg           * kg/m**3
    DS['salt_content'] = DS['volumes'] * DS['vosaline'] * DS['insit_dens']

    # Take the sum in space 
    DS['salt_content'].sum(['deptht','y_grid_T','x_grid_T']).to_netcdf('ls3k_salt_content_'+run+'.nc')

    print('completed: Salt content saved for ' + run)

if __name__ == '__main__':
    for run in ['EPM151','EPM152','EPM155','EPM156','EPM157','EPM158']:
        heat_and_salt_content(run)
