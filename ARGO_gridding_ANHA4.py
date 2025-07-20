#creates netcdf of Argo data from mixedlayer.ucsd.edu
#puts mld on the ANHA4 grid within the Lab Sea (can be modified for different areas)
#looks at the algorithm and threshold mlds

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import griddata

#open netCDF files 
ds = xr.open_dataset('Argo_mixedlayers_all_04142022.nc') #downloaded from mixedlayer.ucsd.edu
mesh = xr.open_dataset('mesh_hgr_ANHA4.nc') #standard ol' ANHA4 grid

#masking the Lab Sea, either using coordinates or x-y grid locations
ds = ds.where( (ds.profilelat>50) & (ds.profilelat<65) & (ds.profilelon<-45) & (ds.profilelon>-65), drop=True)
mesh = mesh.where( (mesh.x>100) & (mesh.x<250) & (mesh.y>300) & (mesh.y<500), drop=True)

#binning the ARGO data in time CHECK AGAINST MATLAB DATEVEC 
start_datetime = datetime(1,1,1,0,0,0) #Jan 1, year 1 (WILL NEED TO SUBTRACT 365 FROM PROFILEDATE, which measures from Jan 1, year 0)
profiledates = ds.profiledate.to_numpy() #list of dates measured as days from Jan 1, year 0
rounded_profiledate = lambda profiledate : 5*round(profiledate/5) #function for binning every 5 days
rounded_profiledates = [rounded_profiledate(profiledate) for profiledate in profiledates]
dtdate = lambda profiledate : start_datetime + timedelta(days=(profiledate-356)) #function for converting to datetime objects
dtdates = [dtdate(rounded_profiledate) for rounded_profiledate in rounded_profiledates]

#initializing output dataset
ARGO = mesh[['nav_lat','nav_lon']] #starting off using the grid lats and lons
ARGO.attrs['TimeStamp'] = 'December 2023'
ARGO.attrs['file_name'] = 'ARGO_mld_ANHA4_LabSea (or something like that)'
ARGO.attrs['description'] = 'Argo MLD data on the ANHA4 grid in the lab sea'
ARGO.attrs['variables'] = 'Density algorithm and density threshold'
ARGO.attrs['source'] = 'http://mixedlayer.ucsd.edu, accessed November 2023'
datecoord = sorted(np.unique(dtdates)) #creating coordinate from dates (looking at every 5 days, chronologically)
ARGO = ARGO.assign_coords({'date': datecoord})
for var in ['da_mld','dt_mld','num_profiles']: #initializing variables for mld
    ARGO[var] = (['y','x','date'], np.empty((mesh.sizes['y'],mesh.sizes['x'],ARGO.sizes['date'])))

#loading into memory
grid_lats = mesh.nav_lat.to_numpy()
grid_lons = mesh.nav_lon.to_numpy()

#ds.da_mld[8] = 100
#print(ds.profilelat[8].to_numpy())
#dtdates[8] = dtdates[0]
#ds.profilelat[8] = ds.profilelat[0]
#ds.profilelon[8] = ds.profilelon[0]
#print(ds.profilelat[8].to_numpy())

#populating ARGO output dataset
for i in range(ds.sizes['iNPROF']): #looping through each Argo data point (could do this using matrix operations, but this is easy to follow)
    lat = ds.profilelat.isel(iNPROF=i).to_numpy() #loading the lat-lon coordinate for the Argo data point
    lon = ds.profilelon.isel(iNPROF=i).to_numpy()
    abslat = np.abs(grid_lats - lat) #finding the "distance" between the the Argo data point and each grid cell
    abslon = np.abs(grid_lons - lon)
    distances = (abslat**2 + abslon**2 )**0.5
    shortest_distance = np.min(distances) #finding the shortest distance and the closest grid cell
    [idy],[idx] = np.where( distances == shortest_distance )
    count = ARGO['num_profiles'].loc[dict(x=idx,y=idy,date=dtdates[i])].to_numpy() #loading the number of Argo profiles in same cell at same time
    for mld in ['da_mld','dt_mld']:
        old_value = ARGO[mld].loc[dict(x=idx,y=idy,date=dtdates[i])] #either adding the mld to the cell and date or taking the avg. with the any other...
        ARGO[mld].loc[dict(x=idx,y=idy,date=dtdates[i])] = (old_value*count + ds[mld][i])/(count+1) #...profiles already saved in that cell and date
    ARGO['num_profiles'].loc[dict(x=idx,y=idy,date=dtdates[i])] += 1 #incrementing the count to keep track of any cells and dates with multiple data points
    print(str(dtdates[i]) + ' done')

print('Max profiles in one cell at one time: ' + str(ARGO.num_profiles.max().to_numpy()))
#ARGO = ARGO.drop_vars('num_profiles')
ARGO.to_netcdf('ARGO_mld_ANHA4_LabSea.nc')
