# Simple little script for making masks
# These masks are full-depth, i.e., they have a depth-dimensions
# Rowan Brown

import xarray as xr
import numpy as np

def open_ANHA4_ex_file():
    """Opens an example ANHA4 output file."""

    # Directory of filepath txt files
    fp_dir = '../filepaths/'

    # Example filepath
    with open(fp_dir + 'EPM155_gridT_filepaths_Jan2024.txt') as f: lines = f.readlines()
    example_fp = [line.strip() for line in lines][10]
    
    # Open the file and return it
    return xr.open_dataset(example_fp)

def mask_LS_specify_depth(depth=3000):
    """Mask the interior Lab Sea following a specified isobath."""

    testFile = open_ANHA4_ex_file()
    exact_depth = testFile.deptht.sel(deptht=3000,method="nearest") # Getting the depth of the nearest layer
    testFile = testFile.isel(time_counter=0) # Basically getting rid of a dimension
    vosaline = testFile.vosaline # Just looking at vosaline; where salinity=0 we have land, and where salinity!=0 we have water
    vosaline = vosaline.where(vosaline > 0) # Getting rid of values in land, where salinity=0

    # Only looking at the lab sea
    northLat = 66
    westLon = -65
    southLat = 53
    eastLon = -43
    vosaline = vosaline.where(vosaline.nav_lat_grid_T < northLat)
    vosaline = vosaline.where(vosaline.nav_lon_grid_T < eastLon)
    vosaline = vosaline.where(vosaline.nav_lat_grid_T > southLat)
    vosaline = vosaline.where(vosaline.nav_lon_grid_T > westLon)

    # Getting the mask 
    vosaline = vosaline.where(vosaline > 0) # Getting rid of values in land, where salinity=0
    exact_depth_slice = vosaline.where(vosaline.deptht == exact_depth, drop=True) # Getting a single layer at the deptht nearest spec'd val
    exact_depth_slice = exact_depth_slice.isel(deptht=0) # Basically getting rid of a dimension 
    exact_depth_slice = exact_depth_slice.notnull() # Looking at where there are values and where there aren't (because we made shelves/land=NaN earlier)
    mask, vosaline = xr.broadcast(exact_depth_slice, vosaline) # Getting the slice into the right shape (basically so there are ~50 depth slices instead of 1)
    mask = mask.drop_vars(['time_centered','time_counter']) # Dropping some coordinates
    mask.attrs = {'description': 'ANHA4 mask of Labrador Sea wherever the depth is greater than '+str(depth)+' m'} # Add descriptive attribute
    mask = mask.rename('mask_LS_' + str(depth)) # Renaming the dataarray
    mask.to_netcdf('masks/mask_LS_'+str(depth)+'.nc') #saving the mask

def mask_full_lab_sea():
    """For simply getting the full Lab Sea region, disregarding land and outside of bounds."""
    
    testFile = open_ANHA4_ex_file()
    exact_depth = testFile.deptht.sel(deptht=3000,method="nearest") # Getting the depth of the nearest layer
    testFile = testFile.isel(time_counter=0) # Basically getting rid of a dimension
    vosaline = testFile.vosaline # Just looking at vosaline; where salinity=0 we have land, and where salinity!=0 we have water
    vosaline = vosaline.where(vosaline > 0) # Getting rid of values in land, where salinity=0

    # Only looking at the lab sea
    northLat = 66
    westLon = -65
    southLat = 53
    eastLon = -43
    vosaline = vosaline.where(vosaline.nav_lat_grid_T < northLat)
    vosaline = vosaline.where(vosaline.nav_lon_grid_T < eastLon)
    vosaline = vosaline.where(vosaline.nav_lat_grid_T > southLat)
    vosaline = vosaline.where(vosaline.nav_lon_grid_T > westLon)

    # Getting the mask
    vosaline = vosaline.where(vosaline > 0) # Getting rid of values in land, where salinity=0
    exact_depth_slice = vosaline.where(vosaline.deptht == exact_depth, drop=True) # Getting a single layer at the deptht nearest spec'd val
    exact_depth_slice = exact_depth_slice.isel(deptht=0) # Basically getting rid of a dimension
    exact_depth_slice = exact_depth_slice.notnull() # Looking at where there are values and where there aren't (because we made shelves/land=NaN earlier)
    mask, vosaline = xr.broadcast(exact_depth_slice, vosaline) # Getting the slice into the right shape (basically so there are ~50 depth slices instead of 1)

    maskLS = vosaline.notnull() #convert to bool, basically (note that shelves are masked)
    maskLS = maskLS.rename('mask_LS') #rename from vosaline
    maskLS = maskLS.drop_vars(['time_centered','time_counter']) #remove coordinates
    maskLS.attrs = {'description': 'ANHA4 mask of Labrador Sea between 53N & 66N and 43W & 65W'} #add descriptive attribute 
    maskLS.to_netcdf('masks/mask_LS.nc') #saving the mask

if __name__=="__main__":
    mask_LS_specify_depth()
    mask_full_lab_sea()
