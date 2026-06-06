# This is a new script in response to recommendations from reviewers
# The goal is to calculate N2 profiles for each timestep throughout
# the interior control region, including N2_heat and N2_salt, to
# better characterise how the stability is changing

import xarray as xr
import gsw
import numpy as np


def calc_stability(run):
    """Saves .nc files of stability (N2, N2_heat, and N2_salt)."""

    print("Beginning: Stability calculations for "+run)

    # Text file of paths to non-empty model output
    gridT_txt = '../filepaths/'+run+'_gridT_filepaths.txt'
    with open(gridT_txt) as f:
        lines = f.readlines()
    fps = [line.strip() for line in lines]

    # Open the files
    preprocess_gridT = lambda ds: ds[['e3t','votemper','vosaline']]
    ds = xr.open_mfdataset(fps, preprocess=preprocess_gridT, engine="netcdf4")
    ds = ds.rename({"x_grid_T": "x", "y_grid_T": "y"})

    # Mask for land, bathymetry, etc. and horiz. grid dimensions
    with xr.open_dataset('masks/ANHA4_mesh_mask.nc') as DS:
        ds['e1t'] = DS.e1t[0,:,:]
        ds['e2t'] = DS.e2t[0,:,:]
    with xr.open_dataarray('masks/mask_LS_3000.nc') as DS:
        ds = ds.assign_coords(mask=DS.astype(int).rename({'x_grid_T':'x','y_grid_T':'y'}))

    # We can cut down to the interior Lab Sea already
    ds = ds.where(ds['mask']==1, drop=True)

    # First we need some specific units and vars to get N2
    ds['p'] = gsw.p_from_z(-ds['deptht'], ds['nav_lat_grid_T'])
    ds['SA'] = gsw.SA_from_SP(ds['vosaline'], ds['p'], ds['nav_lon_grid_T'], ds['nav_lat_grid_T'])
    ds['CT'] = gsw.CT_from_pt(ds['SA'], ds['votemper'])
    ds['a'] = gsw.alpha(ds['SA'], ds['CT'], ds['p'])
    ds['b'] = gsw.beta(ds['SA'], ds['CT'], ds['p'])

    # For comparison
    #Nsquared, p = gsw.Nsquared(ds['SA'].isel(x=30, y=30, time_counter=0), ds['CT'].isel(x=30, y=30, time_counter=0), ds['p'].isel(x=30, y=30), ds['nav_lat_grid_T'].isel(x=30, y=30))
    #print("Nsquared:")
    #print(Nsquared)

    # Now we need vertical gradients
    dz = ds['deptht'].diff('deptht')
    dSA = ds['SA'].diff('deptht') 
    dCT = ds['CT'].diff('deptht')
    dSAdz = dSA/dz
    dCTdz = -dCT/dz

    # Note the negative above in dCTdz was found to be necessary after comparing 
    # N2_temp + N2_salt to Nsquared (calculated with the gsw function)

    # We also need to get the correct new depths and a and b
    z = (ds['deptht'].to_numpy()[:-1] + ds['deptht'].to_numpy()[1:]) / 2
    a = ds['a'].interp(deptht=z)
    b = ds['b'].interp(deptht=z)

    # Note we need the gradients to have the right deptht values
    dSAdz['deptht'] = z
    dCTdz['deptht'] = z

    # Now we can calculate the stability
    g = 9.80665
    N2_temp = g*a*dCTdz
    N2_salt = g*b*dSAdz
    N2 = N2_temp + N2_salt

    # Save it
    ds2 = xr.Dataset({"N2_temp": N2_temp, "N2_salt": N2_salt, "N2": N2})
    ds2 = ds2.mean(['x', 'y'], skipna=True)
    ds2.to_netcdf('stability_LS3000_'+run+'.nc')

    print("Finished: Stability calculations for "+run)
    

if __name__ == "__main__":
    for run in ['EPM157']:
        calc_stability(run)
