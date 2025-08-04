# Creates netcdf of Argo data from mixedlayer.ucsd.edu
# Puts mld on the ANHA4 grid within the Lab Sea (can be modified for different areas)
# Uses the algorithm and threshold mlds
# Rowan Brown
# July 2025

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import griddata

def regrid_Argo(region="LabSea"):
    """Regrids the ARGO MLD data so that it can be compared to ANHA4 output.
    region is a string ("LabSea" or "NorthAtlantic") referring to whether I am looking at the Lab Sea only or the wider North Atlantic."""

    # Open netCDF files 
    ds = xr.open_dataset('Argo_mixedlayers_all_04142022.nc') # downloaded from mixedlayer.ucsd.edu
    mesh = xr.open_dataset('masks/mesh_hgr_ANHA4.nc') # standard ANHA4 grid

    # Do we want to mask the general Lab Sea region
    if region=="LabSea": # If we don't want the full NA, we might as well save time now by excluding some points
        ds = ds.where( (ds.profilelat>50) & (ds.profilelat<65) & (ds.profilelon<-45) & (ds.profilelon>-65), drop=True)
        mesh = mesh.where( (mesh.x>100) & (mesh.x<250) & (mesh.y>300) & (mesh.y<500), drop=True) 
    elif region=="NorthAtlantic": # If we do want the wider NA, we can only exclude certain points
        # Define NA as much larger than the LS, more like equator <-> Fram Strait and Gulf of Mexico <-> Baltic
        # We also only need 01.01.2008--31.12.2017, so might as well cut that down now too
        # Note the Argo profiledates are saved as days since Jan-1-0000
        ds = ds.where( (ds.profilelat>-1) & (ds.profilelat<81) & (ds.profilelon<15) & (ds.profilelon>-100), drop=True) # Generally defining the North Atlantic
        mesh = mesh.where( (mesh.nav_lat>-2) & (mesh.nav_lat<82) & (mesh.nav_lon<16) & (mesh.nav_lon>-101), drop=True) # Slightly bigger than for the Argo dataset
        # Note we'll properly mask the NA later
        start_date = datetime(2008,1,1,0,0,0) - datetime(1,1,1,0,0,0) - timedelta(days=365) # First legal datetime is Jan 1, 0001
        end_date = datetime(2018,1,1,0,0,0) - datetime(1,1,1,0,0,0) - timedelta(days=365)
        ds = ds.where( (ds.profiledate>start_date.days) & (ds.profiledate<end_date.days), drop=True)
    else:
        print("region string is illegal; must be either 'LabSea' or 'NorthAtlantic'")
        quit()
    print("General region masked")

    # Binning the ARGO data in time (based on matlab datevec, I suppose)
    start_datetime = datetime(1,1,1,0,0,0) # Jan 1, year 1 (need to subtract 365 w/r/t profiledate, which measures from Jan 1, year 0)
    profiledates = ds.profiledate.to_numpy() # List of profiledate dates measured as days from Jan 1, year 0
    rounded_profiledate = lambda profiledate : 5*round(profiledate/5) # Function for binning every 5 days
    rounded_profiledates = [rounded_profiledate(profiledate) for profiledate in profiledates]
    dtdate = lambda profiledate : start_datetime + timedelta(days=(profiledate-365)) # Function for converting to datetime objects
    dtdates = [dtdate(rounded_profiledate) for rounded_profiledate in rounded_profiledates]

    # Initializing output dataset
    ARGO = mesh[['nav_lat','nav_lon']] # Starting off using the grid lats and lons
    ARGO.attrs['TimeStamp'] = 'August 2025'
    ARGO.attrs['file_name'] = 'ARGO_mld_ANHA4_'+region
    ARGO.attrs['description'] = 'Argo MLD data on the ANHA4 grid in the '+region
    ARGO.attrs['variables'] = 'Density algorithm and density threshold'
    ARGO.attrs['source'] = 'http://mixedlayer.ucsd.edu, accessed November 2023'
    datecoord = sorted(np.unique(dtdates)) # Creating coordinate from dates (looking at every 5 days, chronologically)
    ARGO = ARGO.assign_coords({'date': datecoord})
    for var in ['da_mld','dt_mld','num_profiles']: # Initializing variables for mld
        ARGO[var] = (['y','x','date'], np.empty((mesh.sizes['y'],mesh.sizes['x'],ARGO.sizes['date'])))

    # Loading into memory
    grid_lats = mesh.nav_lat.to_numpy()
    grid_lons = mesh.nav_lon.to_numpy()

    # Looping through and populating ARGO output dataset, slow but works
    print("Beginning loop through Argo points")
    for i in range(ds.sizes['iNPROF']): # Go through each Argo data point
        
        # Load the lat-lon coordinate for the Argo data point
        lat = ds.profilelat.isel(iNPROF=i).to_numpy()
        lon = ds.profilelon.isel(iNPROF=i).to_numpy()
        
        # Finding the "distance" between the the Argo data point and each grid cell
        abslat = np.abs(grid_lats - lat)
        abslon = np.abs(grid_lons - lon)
        distances = (abslat**2 + abslon**2 )**0.5

        # Finding the shortest distance and the closest grid cell
        shortest_distance = np.min(distances)
        [idy],[idx] = np.where( distances == shortest_distance )
        
        # Loading the number of Argo profiles in the same cell during the same period
        count = ARGO['num_profiles'].loc[dict(x=idx,y=idy,date=dtdates[i])].to_numpy()
        
        # Save the MLDs
        # Either adding the mld to the appropriate cell and date or taking the avg. with any other profiles that are already saved in that cell and date
        # Do it this way to avoid over-weighting areas that are highly sampled
        for mld in ['da_mld','dt_mld']: # (both variables)
            old_value = ARGO[mld].loc[dict(x=idx,y=idy,date=dtdates[i])]
            ARGO[mld].loc[dict(x=idx,y=idy,date=dtdates[i])] = (old_value*count + ds[mld][i])/(count+1) 
        # Incrementing the count to keep track of any cells and dates with multiple data points
        ARGO['num_profiles'].loc[dict(x=idx,y=idy,date=dtdates[i])] += 1 
        print(str(dtdates[i]) + ' done')

    if region=="LabSea":
        # In this case, we will save the 5-daily output for the full Argo time series, since the smaller LabSea domain makes the
        # resultant files not-too-big, even though we're saving with a fairly high time resolution
        print('Max profiles in one cell at one time: ' + str(ARGO.num_profiles.max().to_numpy())) # Just interesting
        ARGO.to_netcdf('Argo_mld_ANHA4_LabSea.nc')
        print('Argo MLD data saved for the Lab Sea')
    elif region=="NorthAtlantic":
        # In this case, the domain is bigger so we won't save 5-daily output for the full Argo time series
        # First, we've already cut down to just the period of interest from 2008 to 2018
        # We've also cut down the mesh to the general North Altantic, although we will properly mask it later
        # Now we'll reample monthly and save
        ARGO.resample(time_counter='1M').mean(dim=['date'],skipna=True)
        print('Max profiles in one cell at one time: ' + str(ARGO.num_profiles.max().to_numpy())) # Just interesting
        ARGO.to_netcdf('Argo_mld_ANHA4_NorthAtlantic.nc')
        print('Argo MLD data saved for the North Atlantic')     

if __name__=="__main__":
    regrid_Argo(region="LabSea")
    regrid_Argo(region="NorthAtlantic")
