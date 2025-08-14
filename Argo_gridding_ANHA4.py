# Creates netcdf of Argo data from mixedlayer.ucsd.edu
# Puts mld on the ANHA4 grid
# Uses the algorithm and threshold mlds
# Rowan Brown
# July 2025

import xarray as xr
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td


def regrid_Argo(region="LabSea"):
    """Regrids the ARGO MLD data so that it can be compared to
    ANHA4 output.
    region is a string ("LabSea" or "NorthAtlantic") referring to
    whether I am looking at the Lab Sea only or the wider North
    Atlantic."""

    # Open netCDF files (argo data from mixedlayer.ucoloursd.edu)
    ds = xr.open_dataset('Argo_mixedlayers_all_04142022.nc')
    mesh = xr.open_dataset('masks/mesh_hgr_ANHA4.nc')  # standard ANHA4 grid

    # Do we want to mask the general Lab Sea region
    if region == "LabSea":
        # If we don't want the full NA, we might as well save time now
        ds = ds.where(
            (ds.profilelat > 50) & (ds.profilelat < 65) &
            (ds.profilelon < -45) & (ds.profilelon > -65),
            drop=True
        )
        mesh = mesh.where( 
            (mesh.x > 100) & (mesh.x < 250) &
            (mesh.y > 300) & (mesh.y < 500),
            drop=True
        )
    elif region == "NorthAtlantic":
        # If we do want the wider NA, we can only exclude certain points
        # Define NA as much larger than the LS, more like
        # equator <-> Fram Strait and Gulf of Mexico <-> Baltic
        # We also only need 01.01.2008--31.12.2017, so might as well cut that
        # Note the Argo profiledates are saved as days since Jan-1-0000
        # Also note that we'll properly mask the NA later
        ds = ds.where(
            (ds.profilelat > -1) & (ds.profilelat < 81) &
            (ds.profilelon < 15) & (ds.profilelon > -100),
            drop=True
        )
        # First legal dt is Jan 1, 0001
        # Basically converts into a td from 1-1-0000
        start_date = (
            dt(2008, 1, 1, 0, 0, 0) - dt(1, 1, 1, 0, 0, 0) + td(days=365)
        )
        end_date = (
            dt(2018, 1, 1, 0, 0, 0) - dt(1, 1, 1, 0, 0, 0) + td(days=365)
        )
        ds = ds.where(
            (ds.profiledate > start_date.days) &
            (ds.profiledate < end_date.days),
            drop=True
        )
    else:
        print("region string is illegal; must be either "
              "'LabSea' or 'NorthAtlantic'")
        quit()
    print("General region masked")

    # Binning the ARGO data in time (based on matlab datevec, I suppose)
    # Jan 1, year 1 (need to subtract 365 w/r/t profiledate, which measures
    # from year 0)
    start_dt = dt(1, 1, 1, 0, 0, 0)
    profiledates = ds.profiledate.to_numpy()  # days from Jan 1, year 0
    # Function for binning every 5 days:
    def rounded_profiledate(profiledate): return 5*round(profiledate/5)
    rounded_profiledates = [
        rounded_profiledate(profiledate)
        for profiledate
        in profiledates
    ]
    # Function for converting to dt objects:
    def dtdate(profiledate): return start_dt + td(days=(profiledate-365))
    dtdates = [
        dtdate(rounded_profiledate)
        for rounded_profiledate
        in rounded_profiledates
    ]

    # Initializing output dataset
    ARGO = mesh[['nav_lat', 'nav_lon']]  # Starting off using the grid coords
    ARGO.attrs['TimeStamp'] = 'August 2025'
    ARGO.attrs['file_name'] = 'ARGO_mld_ANHA4_'+region
    ARGO.attrs['description'] = 'Argo MLD data on the ANHA4 grid in '+region
    ARGO.attrs['variables'] = 'Density algorithm and density threshold'
    ARGO.attrs['source'] = 'http://mixedlayer.ucoloursd.edu'
    datecoord = sorted(np.unique(dtdates))  # Creating coordinate from dates
    ARGO = ARGO.assign_coords({'date': datecoord})
    for var in ['da_mld', 'dt_mld', 'num_profiles']:  # Initializing mld vars
        ARGO[var] = (
            ['y', 'x', 'date'],
            np.empty((mesh.sizes['y'], mesh.sizes['x'], ARGO.sizes['date']))
        )

    # Loading into memory
    grid_lats = mesh.nav_lat.to_numpy()
    grid_lons = mesh.nav_lon.to_numpy()

    # Looping through and populating ARGO output dataset, slow but works
    print("Beginning loop through Argo points")
    for i in range(ds.sizes['iNPROF']):  # Go through each Argo data point
        
        # Load the lat-lon coordinate for the Argo data point
        lat = ds.profilelat.isel(iNPROF=i).to_numpy()
        lon = ds.profilelon.isel(iNPROF=i).to_numpy()
        
        # Finding the "distance" between the the Argo point and each grid cell
        abslat = np.abs(grid_lats - lat)
        abslon = np.abs(grid_lons - lon)
        distances = (abslat**2 + abslon**2)**0.5

        # Finding the shortest distance and the closest grid cell
        shortest_distance = np.nanmin(distances)
        try:
            [idy], [idx] = np.where(distances == shortest_distance)
        except ValueError:
            # If except it's because the point is equidistant to two cells
            ids = np.where(distances == shortest_distance)
            idy = ids[0][0]
            idx = ids[1][0]

        # Loading the number of Argo profiles in the same cell during
        # the same period
        count = ARGO['num_profiles'].loc[
            dict(x=idx, y=idy, date=dtdates[i])
        ].to_numpy()

        # Save the MLDs
        # Either adding the mld to the appropr cell and date or taking the avg
        # with any other profiles that are already saved in that cell and date
        # Do it this way to avoid over-weighting areas that are highly sampled
        for mld in ['da_mld', 'dt_mld']:  # (both variables)
            old_value = ARGO[mld].loc[dict(x=idx, y=idy, date=dtdates[i])]
            ARGO[mld].loc[
                dict(x=idx, y=idy, date=dtdates[i])
            ] = (old_value*count + ds[mld][i])/(count+1)
        # Incrementing the count to keep track of any cells and dates with
        # multiple data points
        ARGO['num_profiles'].loc[dict(x=idx, y=idy, date=dtdates[i])] += 1
        print(str(dtdates[i]) + ' done')

    # Save
    print('Max profiles in one cell at one time: ' +
          str(ARGO.num_profiles.max().to_numpy()))
    ARGO.to_netcdf('Argo_mld_ANHA4_'+region+'.nc')
    print('Argo MLD data saved for the '+region)
    # Note if the region is the whole NA, the resultant file is around 7G


if __name__ == "__main__":
    # regrid_Argo(region="LabSea")
    regrid_Argo(region="NorthAtlantic")
