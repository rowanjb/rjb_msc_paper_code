# Creates cross section netcdfs that can be plotted later
# Rowan Brown, 14 Aug 2025

import xarray as xr
import numpy as np
from metpy.interpolate import geodesic
import pyproj
import xoak
import gsw


def ANHA4_xsection_maker(run, date):
    """Creates example cross section nc files with isopycnals and the
    MLE SF."""

    print("Beginning: Cross section calculations for "+run+", "+date)

    # Start and end coordinates of the AR7W cells
    vertices_lon = [-56.458036, -48.036965]
    vertices_lat = [53.410189, 60.733433]

    # Text file of paths to non-empty model output
    gridT_txt = '../filepaths/'+run+'_gridT_filepaths.txt'
    with open(gridT_txt) as f:
        lines = f.readlines()
    filepaths_gridT = [line.strip() for line in lines]

    # Open the files
    [fp] = [fp for fp in filepaths_gridT if date in fp]
    vars = ['vosaline',
            'votemper',
            'MLE Lf',
            'i-mle streamfunction',
            'j-mle streamfunction']
    ds = xr.open_dataset(fp)[vars]

    # Interpolate the streamfunction components onto same grid
    iMLE = ds['i-mle streamfunction'].interp(
        x_grid_U=ds.x_grid_U-0.5).drop_vars(
        ['x_grid_U', 'nav_lat_grid_U', 'nav_lon_grid_U'])
    iMLE = iMLE.rename({'depthu': 'deptht', 'x_grid_U': 'x_grid_T',
                        'y_grid_U': 'y_grid_T'})
    jMLE = ds['j-mle streamfunction'].interp(
        y_grid_V=ds.y_grid_V-0.5).drop_vars(
        ['y_grid_V', 'nav_lat_grid_V', 'nav_lon_grid_V'])
    jMLE = jMLE.rename({'depthv': 'deptht', 'x_grid_V': 'x_grid_T',
                        'y_grid_V': 'y_grid_T'})

    # Calculate the "magnitude" of the streamfunction
    # Note our end result will be the 3D value of the SF
    # (the "magnitude"), not some projected quantity
    ds['Psi'] = (iMLE**2 + jMLE**2)**0.5
    ds = ds.drop_vars(['i-mle streamfunction',
                       'j-mle streamfunction',
                       'nav_lon_grid_U',
                       'nav_lat_grid_U'])
    ds = ds.drop_vars(['nav_lon_grid_V',
                       'nav_lat_grid_V',
                       'depthu',
                       'depthv'])

    # For calculating distances etc
    ds.xoak.set_index(['nav_lat_grid_T', 'nav_lon_grid_T'], 'scipy_kdtree')
    wgs84 = pyproj.CRS(4326)

    # Define the path
    start = (vertices_lat[0], vertices_lon[0])  # Starting point
    end = (vertices_lat[1], vertices_lon[1])  # Ending end
    # Getting "steps" number of points on the path between start and end
    path = geodesic(wgs84, start, end, steps=1000)
    # Note 40 is a rough guess at the number of cells spanning the Lab Sea
    # With 1000 steps, I believe there would be an excellent representation of
    # the coarse-ness of the 1/4-deg grid

    # Getting the cross section (i.e., values at each point on the
    # path and at each depth)
    cross = ds.xoak.sel(
        nav_lat_grid_T=xr.DataArray(
            path[:, 1], dims='index', attrs=ds.nav_lat_grid_T.attrs),
        nav_lon_grid_T=xr.DataArray(
            path[:, 0], dims='index', attrs=ds.nav_lon_grid_T.attrs)
    )

    # Getting distances from starting point
    # Can also get angles but I don't need that now so I will comment
    # out these lines
    geod = pyproj.Geod(ellps='sphere')  # Use pyproj.Geod class
    distances = []
    # forward_azimuth = [] # Don't need currently
    for i in path:
        az12, __, dist = geod.inv(
            vertices_lon[0], vertices_lat[0], i[0], i[1])
        distances = np.append(distances, dist)
        # forward_azimuth = np.append(forward_azimuth, az12)
    # The first element is always 180
    # forward_azimuth[0] = forward_azimuth[1]
    # forward_azimuth = np.radians(forward_azimuth)

    # Add distances to the cross section
    cross = cross.assign_coords(dists=('index', distances))
    # and forward azimuth
    # cross=cross.assign_coords(forward_azimuth=('index',forward_azimuth))

    # Finally just adding density
    cross['P'] = gsw.p_from_z(
        (-1)*cross['deptht'],  # (Requires negative depths)
        cross['nav_lat_grid_T'])
    cross['SA'] = gsw.SA_from_SP(
        cross['vosaline'],
        cross['P'],
        cross['nav_lon_grid_T'],
        cross['nav_lat_grid_T'])
    cross['CT'] = gsw.CT_from_pt(
        cross['SA'],
        cross['votemper'])
    cross['pot_dens'] = gsw.rho(
        cross['SA'],
        cross['CT'],
        0)  # gsw.rho with p=0 gives potential density

    cross.to_netcdf('cross_section_'+run+'_'+date+'.nc')

    print("Completed: Cross section calculations for "+run+", "+date)


if __name__ == "__main__":
    for run in ['EPM155','EPM156']:
        for date in ['y2013m05d15', 'y2013m07d04']:
            ANHA4_xsection_maker(run, date)

