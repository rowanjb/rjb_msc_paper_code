# Maps and cross-sections of velocity and density comparing LAB60 and ANHA4
# Effectively baselining the ANHA4 runs agains a high-resolution run
# Consider also looking at WOA
# Rowan Brown
# July 2025

import xarray as xr
import os
import numpy as np
import xarray as xr
from datetime import datetime
from metpy.interpolate import geodesic
import pyproj
import xoak
import sys
import gsw

def ANHA4_hydrography_maps(run):
    """Creates monthly maps of surface mean velocities, temperatures, and salinities, for evaluation against LAB60."""

    print("Beginning: Hydrography calculations (maps) for "+run)

    # Masks (for land, bathymetry, etc. and horiz. grid dimensions)
    with xr.open_dataset('masks/ANHA4_mesh_mask.nc') as DS:
        tmask = DS.tmask[0,:,:,:].rename({'z': 'deptht', 'y': 'y_grid_T', 'x': 'x_grid_T'})
        e1t = DS.e1t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
        e2t = DS.e2t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
    mask = xr.open_dataarray('masks/mask_LS.nc').astype(int)
    
    # Text file of paths to non-empty model output
    gridT_txt = '../filepaths/'+run+'_gridT_filepaths_jul2025.txt'
    gridU_txt = '../filepaths/'+run+'_gridU_filepaths_jul2025.txt'
    gridV_txt = '../filepaths/'+run+'_gridV_filepaths_jul2025.txt'

    # Open the text files and get lists of the .nc output filepaths
    with open(gridT_txt) as f: lines = f.readlines()
    filepaths_gridT = [line.strip() for line in lines]
    with open(gridU_txt) as f: lines = f.readlines()
    filepaths_gridU = [line.strip() for line in lines]
    with open(gridV_txt) as f: lines = f.readlines()
    filepaths_gridV = [line.strip() for line in lines]

    # Open the files
    preprocess_gridT = lambda ds: ds[['e3t','votemper','vosaline','somxlts']]
    DST = xr.open_mfdataset(filepaths_gridT,preprocess=preprocess_gridT)
    preprocess_gridU = lambda ds: ds[['vozocrtx']]
    DSU = xr.open_mfdataset(filepaths_gridU,preprocess=preprocess_gridU)
    preprocess_gridV = lambda ds: ds[['vomecrty']]
    DSV = xr.open_mfdataset(filepaths_gridV,preprocess=preprocess_gridV)

    # Add horizontal cell dims
    DST[['e1t','e2t']] = e1t,e2t

    # Interpolate to get velocities onto the T grid
    DSU = DSU.interp(x=DSU.x-0.5).drop_vars('x') # Co-locating velsocities on the T grid
    DSV = DSV.interp(y=DSV.y-0.5).drop_vars('y')

    # Add velocities to the T grid
    DST['U'] = DSU['vozocrtx'].rename({'x':'x_grid_T','y':'y_grid_T','depthu':'deptht'})
    DST['V'] = DSV['vomecrty'].rename({'x':'x_grid_T','y':'y_grid_T','depthv':'deptht'})

    # Need to mask the shelves/sea floor, or else the "0" temperatures are counted
    # Easy way to do this is to mask anywhere with 0 salinities, since 0 temps are plausible
    DST = DST.where(DST.vosaline>0)

    # Apply tmask (which I /think/ it for land etc.)
    DST = DST.where(tmask == 1)

    # Apply region mask
    DST.coords['mask'] = mask
    DST = DST.where(DST.mask == 1, drop=True)

    #== Calculations ==#

    # Define the surface
    depth = 400 # 400 is for consistency with the energetics calculations
    DST = DST.where(DST['deptht'] < depth, drop=True)

    # Take monthly means 
    #DST = DST.groupby('time_counter.month').mean(dim=['time_counter'],skipna=True)
    DST = DST.resample(time_counter='1M').mean(dim=['time_counter'],skipna=True)

    # Use weights to equal-out the differences in cell thickness
    DST['weights'] = DST['e3t']/DST['e3t'].mean(dim='deptht')
    DST['weights'] = DST['weights'].fillna(0) # Probably unnecessary
   
    # Final calculations and saving
    DST['votemper_surface_mean'] = DST['votemper'].weighted(DST['weights']).mean('deptht',skipna=True)
    DST['vosaline_surface_mean'] = DST['vosaline'].weighted(DST['weights']).mean('deptht',skipna=True)
    DST['U_surface_mean'] = DST['U'].weighted(DST['weights']).mean('deptht',skipna=True)
    DST['V_surface_mean'] = DST['V'].weighted(DST['weights']).mean('deptht',skipna=True)
    DST[['votemper_surface_mean', 'vosaline_surface_mean', 'U_surface_mean', 'V_surface_mean']].drop_vars(['nav_lon','nav_lat']).to_netcdf('hydrography_maps_monthly_'+run+'.nc')

    print('Completed: Hydrography calculations (maps) for '+run)

    # Closing
    DST.close()
    DSU.close()
    DSV.close()

def LAB60_hydrography_maps():
    """Duplicate of the ANHA4_hydrography_maps function but for LAB60."""

def ANHA4_hydrography_sections(run):
    """Creates monthly sections of mean temperatures, salinities, densities, and MLEp (where applicable) for evaluation against LAB60.
    Also for showing how the MLEp works AND checking its effects along the shelves."""

    print("Beginning: Cross section calculations for "+run)
    
    # Masks (for land, bathymetry, etc. and horiz. grid dimensions)
    with xr.open_dataset('masks/ANHA4_mesh_mask.nc') as DS:
        tmask = DS.tmask[0,:,:,:].rename({'z': 'deptht', 'y': 'y_grid_T', 'x': 'x_grid_T'})
        e1t = DS.e1t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
        e2t = DS.e2t[0,:,:].rename({'y': 'y_grid_T', 'x': 'x_grid_T'})

    # Text file of paths to non-empty model output
    gridT_txt = '../filepaths/'+run+'_gridT_filepaths_jul2025.txt'
    with open(gridT_txt) as f: lines = f.readlines()
    filepaths_gridT = [line.strip() for line in lines]#[500:1100:110]

    # Open the files
    preprocess_gridT = lambda ds: ds[['votemper','vosaline','somxlts']]
    DS = xr.open_mfdataset(filepaths_gridT,preprocess=preprocess_gridT)

    # Mask the general region of the Lab Sea
    DS = DS.where( (DS['y_grid_T'] > 250) & (DS['y_grid_T'] < 500) & (DS['x_grid_T'] > 100) & (DS['x_grid_T'] < 250), drop=True)

    # Take monthly means to cut down on computing power
    #DS = DS.groupby('time_counter.month').mean(dim=['time_counter'],skipna=True)
    DS = DS.resample(time_counter='1M').mean(dim=['time_counter'],skipna=True)

    # It we're looking at EPM155 or EPM156, then we also want the sf output     
    if run in ['EPM155','EPM156']:
            
        # Note we only have diagnostics for six years, so we might as well only open the necessary files
        start, end = datetime(2012,1,1), datetime(2017,12,31)
        mle_filepaths_gridT = [fp for fp in filepaths_gridT if (start <= datetime.strptime(fp[-20:-9],'y%Ym%md%d') <= end)]
        mle_preprocess_gridT = lambda ds: ds[['MLE Lf','i-mle streamfunction','j-mle streamfunction']]
        DS_mle = xr.open_mfdataset(mle_filepaths_gridT,preprocess=mle_preprocess_gridT)

        # Interpolate the streamfunction components
        iMLE = DS_mle['i-mle streamfunction'].interp(x_grid_U=DS_mle.x_grid_U-0.5).drop_vars(['x_grid_U','nav_lat_grid_U','nav_lon_grid_U'])
        iMLE = iMLE.rename({'depthu':'deptht','x_grid_U':'x_grid_T','y_grid_U':'y_grid_T'})
        jMLE = DS_mle['j-mle streamfunction'].interp(y_grid_V=DS_mle.y_grid_V-0.5).drop_vars(['y_grid_V','nav_lat_grid_V','nav_lon_grid_V'])
        jMLE = jMLE.rename({'depthv':'deptht', 'x_grid_V':'x_grid_T','y_grid_V':'y_grid_T'})

        # Calculate the "magnitude" of the streamfunction 
        # Note our end result will be the 3D value of the SF (the "magnitude"), not some projected quantity
        DS_mle['Psi'] = (iMLE**2 + jMLE**2)**0.5
        DS_mle = DS_mle.drop_vars(['i-mle streamfunction','j-mle streamfunction','nav_lon_grid_U','nav_lat_grid_U'])
        DS_mle = DS_mle.drop_vars(['nav_lon_grid_V','nav_lat_grid_V','depthu','depthv'])

        # Mask the general region of the Lab Sea
        DS_mle = DS_mle.where( (DS_mle['y_grid_T'] > 250) & (DS_mle['y_grid_T'] < 500) & (DS_mle['x_grid_T'] > 100) & (DS_mle['x_grid_T'] < 250), drop=True)

        # Take monthly means to cut down on computing power
        #DS_mle = DS_mle.groupby('time_counter.month').mean(dim=['time_counter'],skipna=True)    
        DS_mle = DS_mle.resample(time_counter='1M').mean(dim=['time_counter'],skipna=True)

        # Combine the MLE variables with the standard variables
        DS = DS.merge(DS_mle, join='outer')
        DS_mle.close()
        print("MLE data merged")

    #== Now to take the cross sections ==#

    # Start and end coordinates of the AR7W cells
    vertices_lon = [-56.458036, -48.036965]
    vertices_lat = [53.410189, 60.733433]

    # For calculating distances etc 
    DS.xoak.set_index(['nav_lat_grid_T', 'nav_lon_grid_T'], 'scipy_kdtree')
    wgs84 = pyproj.CRS(4326)

    # Define the path
    start = (vertices_lat[0], vertices_lon[0]) # Starting point
    end   = (vertices_lat[1], vertices_lon[1]) # Ending end
    path = geodesic(wgs84, start, end, steps=160) # Getting "steps" number of points on the path between start and end 
    # Note 40 is a rough guess at the number of cells spanning the Lab Sea
    # With 160 steps, I believe there would be a good representation of the coarse-ness of the 1/4-deg grid

    # Getting the cross section (i.e., values at each poin on the path and at each depth)
    cross = DS.xoak.sel( 
            nav_lat_grid_T=xr.DataArray(path[:, 1], dims='index', attrs=DS.nav_lat_grid_T.attrs),
            nav_lon_grid_T=xr.DataArray(path[:, 0], dims='index', attrs=DS.nav_lon_grid_T.attrs)
    )

    # Getting distances from starting point 
    # Can also get angles but I don't need that now so I will comment out these lines
    geod = pyproj.Geod(ellps='sphere') # Use pyproj.Geod class
    distances = []
    #forward_azimuth = []
    for i in path:
            az12, __, dist = geod.inv(vertices_lon[0],vertices_lat[0],i[0],i[1])
            distances = np.append(distances,dist)
            #forward_azimuth = np.append(forward_azimuth, az12)
    #forward_azimuth[0] = forward_azimuth[1] # The first element is always 180
    #forward_azimuth = np.radians(forward_azimuth)

    cross = cross.assign_coords(dists=('index',distances)) # Add distances to the cross section 
    #cross = cross.assign_coords(forward_azimuth=('index',forward_azimuth)) # and forward azimuth

    # Finally just adding density
    cross['P'] = gsw.p_from_z( (-1)*cross['deptht'], cross['nav_lat_grid_T'] ) # Requires negative depths
    cross['SA'] = gsw.SA_from_SP( cross['vosaline'], cross['P'], cross['nav_lon_grid_T'], cross['nav_lat_grid_T'] )
    cross['CT'] = gsw.CT_from_pt( cross['SA'], cross['votemper'] )
    cross['pot_dens'] = gsw.rho( cross['SA'], cross['CT'], 0 ) # gsw.rho with p=0 gives potential density

    # And save
    cross = cross.drop_vars(['P','SA','CT'])
    cross.to_netcdf('hydrography_sections_monthly_'+run+'.nc')

    print("Completed: Cross section calculations for "+run)
    DS.close()

def LAB60_hydrography_sections():
    """Duplicate of the ANHA4_hydrography_sections function minus the MLEp output."""


if __name__=="__main__":
    for run in ['EPM155','EPM156','EPM157','EPM158']:
        ANHA4_hydrography_maps(run)
        #ANHA4_hydrography_sections(run)
