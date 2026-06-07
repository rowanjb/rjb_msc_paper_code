# Modified copy of Argo_gridding_ANHA4.py
# Reviewer suggested estimating uncertainty in Argo calculations.

# For the 221 m Argo mean MLD in the interior Lab Sea, the new
# approach is to find each Argo float during winter in the
# Lab Sea and then save that specific float to a dataset. The stddev
# can then be calculated (note the mean is the same as before).
#  => This is the function "calc_Argo_uncertainty_LabSea"

# For comparison I'll also do the same calculations applied to the
# model, i.e., find each model value that corresponds to a float and
# save that to a new nc; then the same stddev calcs can be applied. I
# don't need this for the paper, though.
#  => This is the function "compare_to_ANHA4"

# The stddev (and mean) calculations are done with the another function.
#  => "simple_MLD_mean_std_dev"

# Regarding the supplemental figure, it would be good to know the error
# between the model and Argo, but comparing individual floats to model
# points is fraught with difficulties and not physically appropriate
# anyway. Instead, we will follow this algorithm:
#  1) Identify all Argo floats in the region of interest, and save
#     their MLD, grid location, and time (as before, multiple float
#     values with a 5-day window are meaned---this is already done
#     in "Argo_gridd_ANHA4.py")
#  2) Identify corresponding model values in space and time
#  4) Take the 10-year mean of the Argo data and calculate the error
#     and mean between the model points and the Argo mean
#  => This is the function "calculate_Argo_model_differences_NorthAtlantic"
#  5) Map it (this is done in the script "figure_MLDs_supplemental.py")

# Rowan Brown, Weddell Sea, Mar 2026

import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
from cftime import DatetimeNoLeap


def calc_Argo_uncertainty_LabSea():
    """Calculates the uncertainty relating to Argo-based mean MLDs.
    Works a bit differently than our previous treatment of Argo.
    Compared to gridding them (as we have already done)
    the current approach makes it possible to calculate std dev
    (and compare individual floats to the model without risking
    aliasing)."""

    # Open netCDF files (argo data from mixedlayer.ucoloursd.edu)
    ds = xr.open_dataset('Argo_mixedlayers_all_04142022.nc')
    mesh = xr.open_dataset('masks/mesh_hgr_ANHA4.nc')
    mask = xr.open_dataarray('masks/mask_LS_3000.nc').astype(int).isel(deptht=0)
    mesh['mask'] = mask.rename({"y_grid_T": "y", "x_grid_T": "x"})

    # Cut down size of datasets to save memory
    ds = ds.where(
        (ds.profilelat > 50) & (ds.profilelat < 65) &
        (ds.profilelon < -45) & (ds.profilelon > -65),
        drop=True
    )

    # Getting the ARGO date and grid locagion
    # Jan 1, year 1 (need to subtract 365 w/r/t profiledate, which measures
    # from year 0)
    start_dt = dt(1, 1, 1, 0, 0, 0)
    profiledates = ds.profiledate.to_numpy()  # days from Jan 1, year 0

    # Function for converting to Timestamp objects:
    def dtdate(d):
        return (start_dt + td(days=(d-365)))
    dtdates = [dtdate(d) for d in profiledates]

    # Loading into memory
    grid_lats = mesh.nav_lat.to_numpy()
    grid_lons = mesh.nav_lon.to_numpy()

    # In the end we want a 1D dataset with coord iNPROF and vars
    mld = []
    gridy = []
    nav_lat = []
    gridx = []
    nav_lon = []
    date = []
    iNPROF = []  # i.e., coord

    # Looping through and populating ARGO output dataset
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

        # Save the MLD if it fits in our time and region
        m = dtdates[i].month
        y = dtdates[i].year
        if (y in range(2008, 2018)) and (m in [12, 1, 2, 3, 4]):
            if mesh['mask'].sel(x=idx, y=idy).to_numpy() == 1:
                mld.append(ds['da_mld'].sel(iNPROF=i).data)
                nav_lat.append(mesh['nav_lat'].sel(x=idx, y=idy).data)
                nav_lon.append(mesh['nav_lon'].sel(x=idx, y=idy).data)
                gridx.append(idx)
                gridy.append(idy)
                date.append(dtdates[i])
                iNPROF.append(i)
                print(ds['da_mld'].sel(iNPROF=i).data)

    # Initializing output dataset
    ARGO = xr.Dataset(
        data_vars=dict(
            mld=(["iNPROF"], mld),
            nav_lat=(["iNPROF"], nav_lat),
            nav_lon=(["iNPROF"], nav_lon),
            gridx=(["iNPROF"], gridx),
            gridy=(["iNPROF"], gridy),
            date=(["iNPROF"], date),
        ),
        coords=dict(
            iNPROF=("iNPROF", iNPROF),
        ),
        attrs=dict(
            Timestamp = "March 2026",
            file_name = "Argo_mld_LabSea.nc",
            description = ("Density algorithm MLD of all floats during DJFMA" +
                           " 2008-2017 inclusive in the interior Lab Sea"),
            source = "https://mixedlayer.ucsd.edu",
        ),
    )

    # Turns out we actually need cftime
    ARGO = ARGO.where(  # Drop Feb 29
        ~((ARGO['date'].dt.month == 2) & (ARGO['date'].dt.day == 29)),
        drop=True
    )
    date = []
    for d in ARGO['date'].to_numpy():
        d = pd.Timestamp(d)
        date.append(DatetimeNoLeap(
            d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond
        ))
    ARGO['date'] = (['iNPROF'], date)

    # Save
    ARGO.to_netcdf("Argo_mld_LabSea.nc")
    print("File saved")


def compare_to_ANHA4(run):
    """Since we have the individual float data in the interior Lab Sea,
    we can compare to the nearest neighbour ANHA4 points and compute
    some statistics."""

    # Open the Argo data
    print("Opening Argo dataset")
    ARGO = xr.open_dataset("Argo_mld_LabSea.nc")

    # Open the ANHA4 data in the normal way
    # Masks (for land, bathymetry, etc. and horiz. grid dimensions)
    with xr.open_dataset('masks/ANHA4_mesh_mask.nc') as DS:
        tmask = DS.tmask[0, :, :, :].rename(
            {'z': 'deptht', 'y': 'y_grid_T', 'x': 'x_grid_T'})
        e1t = DS.e1t[0, :, :].rename(
            {'y': 'y_grid_T', 'x': 'x_grid_T'})
        e2t = DS.e2t[0, :, :].rename(
            {'y': 'y_grid_T', 'x': 'x_grid_T'})
    mask = xr.open_dataarray('masks/mask_LS_3000.nc').astype(int)

    # Text file of paths to non-empty model output
    gridT_txt_nibi = '../filepaths/'+run+'_gridT_filepaths_jul2025.txt'
    gridT_txt = '../filepaths/'+run+'_gridT_filepaths.txt'

    # Open the text files and get lists of the .nc output filepaths
    try:
        with open(gridT_txt_nibi) as f:
            lines = f.readlines()
    except FileNotFoundError:
        with open(gridT_txt) as f:
            lines = f.readlines()
    filepaths_gridT = [line.strip() for line in lines]

    # Open the files and look at e3t and votemper
    print("Opening "+run+" dataset")
    preprocess_gridT = lambda ds: ds[['somxlts']]
    DS = xr.open_mfdataset(filepaths_gridT, preprocess=preprocess_gridT)

    # Add horizontal cell dims
    DS[['e1t', 'e2t']] = e1t, e2t

    # Find nearest model data and save it
    mld = []
    tc = []
    print("Identifying model points")
    for i in ARGO['iNPROF'].to_numpy():
        argo_date = ARGO['date'].sel(iNPROF=i)
        argox = int(ARGO['gridx'].sel(iNPROF=i).data)
        argoy = int(ARGO['gridy'].sel(iNPROF=i).data)
        mld.append(
            DS["somxlts"].sel(
            time_counter=argo_date,
            method='nearest',
        ).sel(
            x_grid_T=argox,
            y_grid_T=argoy,
        ).to_numpy())
        tc.append(
            DS["time_counter"].sel(
            time_counter=argo_date,
            method='nearest',
        ).item())
        print(ARGO['mld'].sel(iNPROF=i).to_numpy(), mld[-1])
    ARGO[run] = (["iNPROF"], mld)
    ARGO['time_counter'] = (["iNPROF"], tc)

    # Now we also want the pseudo monthly climatology
    print("Calculating 2008--2018 climatology")
    da = DS['somxlts'].sel(
        time_counter=slice(
            DatetimeNoLeap(2008, 1, 1),
            DatetimeNoLeap(2018, 1, 1)
        )).groupby('time_counter.month').mean('time_counter')
    ARGO['mld_clim'] = da
    ARGO.to_netcdf("Argo_mld_LabSea_"+run+".nc")
    print("Finished saving "+run)


def simple_MLD_mean_std_dev():
    """Simply calculate the mean and std dev for the
    Argo and ANHA4 runs."""

    print("Calculating means and std devs...")

    # Calculating the mean and std dev
    def calc(ds):
        m = ds['mld'].mean().to_numpy()
        sd = ds['mld'].std().to_numpy()
        return m, sd

    # Open the Argo file
    ARGO = xr.open_dataset("Argo_mld_LabSea.nc")
    m, sd = calc(ARGO)
    print("Argo mean and std dev: "+str(m)+', '+str(sd))

    # Opening the ANHA4 files
    def open_MLD(run):
        ds = xr.open_dataset("Argo_mld_LabSea_"+run+".nc")
        ds = ds.drop_vars("mld").rename({run: "mld"})
        return ds

    runs = ['EPM151', 'EPM152', 'EPM155', 'EPM156', 'EPM157', 'EPM158']
    for run in runs:
        ds = open_MLD(run)
        m, sd = calc(ds)
        print(run+" mean and std dev: "+str(m)+', '+str(sd))


def calculate_Argo_model_differences_NorthAtlantic():

    ds = xr.open_dataset("Argo_mld_ANHA4_NorthAtlantic.nc")
    ds = ds.set_coords(['nav_lat', 'nav_lon'])
    da = ds['da_mld']
    da = da.where(da > 0)
    da = da.rename({'date': 'time'}).convert_calendar("noleap", use_cftime=True)
    da_1d = da.stack(da_mld=("y", "x", "time")).dropna("da_mld")
    da_1d = da_1d.reset_index('da_mld')

    runs = ['EPM151', 'EPM152', 'EPM155', 'EPM156', 'EPM157', 'EPM158']
    for run in runs:

        # Text file of paths to non-empty model output
        gridT_txt_nibi = '../filepaths/'+run+'_gridT_filepaths_jul2025.txt'
        gridT_txt = '../filepaths/'+run+'_gridT_filepaths.txt'

        # Open the text files and get lists of the .nc output filepaths
        try:
            with open(gridT_txt_nibi) as f:
                lines = f.readlines()
        except FileNotFoundError:
            with open(gridT_txt) as f:
                lines = f.readlines()
        filepaths_gridT = [line.strip() for line in lines]

        # Now open the model data
        print("Opening "+run+" dataset")
        preprocess_gridT = lambda ds: ds[['somxlts']]
        ds = xr.open_mfdataset(filepaths_gridT, preprocess=preprocess_gridT)

        mld_1d = ds['somxlts'].isel(
            y_grid_T=da_1d['y'],
            x_grid_T=da_1d['x'],
        ).sel(time_counter=da_1d['time'], method='nearest')

        mld_1d.to_netcdf("Argo_mld_NorthAtlantic_"+run+".nc")
        print("Modelled MLDs at Argo points saved for "+run)

    da_1d.to_netcdf("Argo_mld_NorthAtlantic.nc")
    print("MLDs at Argo points saved")


if __name__ == "__main__":

    # Calculating the stddev in the Lab Sea Argo data
    #calc_Argo_uncertainty_LabSea()
    #compare_to_ANHA4(run)
    #simple_MLD_mean_std_dev()

    # Calculating the model-Argo error in the North Atlantic
    calculate_Argo_model_differences_NorthAtlantic()
